#!/usr/bin/env python3

import itertools
import logging
import re
from typing import Literal

import click
from beartype import beartype

from chainscope.api_utils.open_ai_utils import \
    process_batch_results as process_openai_batch_results
from chainscope.api_utils.open_ai_utils import submit_openai_batch
from chainscope.typing import *

PROMPT = """Please classify the following question into CLEAR or AMBIGUOUS.

Guidelines:
- Choose CLEAR if the question only admits one interpretation and one correct answer.
- Choose AMBIGUOUS if the question admits more than one interpretation, which could lead to different answers.

Format output:
<analysis>Your analysis of the question</analysis>
<classification>CLEAR/AMBIGUOUS</classification>

Question: `{question}`"""


@beartype
def extract_classification(response: str) -> tuple[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"], str | None]:
    """Extract classification and analysis from response.
    
    Returns:
        tuple: (classification, analysis)
            - classification: CLEAR, AMBIGUOUS, or FAILED_EVAL
            - analysis: The analysis string or None if failed to extract
    """
    try:
        analysis_match = re.search(r"<analysis>(.*?)(?:</analysis>|<classification>)", response, re.DOTALL)
        classification_match = re.search(
            r"<classification>(.*?)</classification>", response, re.DOTALL
        )

        if not analysis_match:
            logging.warning(f"Could not parse analysis: {response}")
            analysis = None
        else:
            analysis = analysis_match.group(1).strip()
            if not analysis:
                logging.warning(f"Got an empty analysis")
                analysis = None

        if not classification_match:
            logging.warning(f"Could not parse classification: {response}")
            classification = "FAILED_EVAL"
        else:
            classification_str = classification_match.group(1).strip()
            classification_types: dict[str, Literal["CLEAR", "AMBIGUOUS"]] = {
                "CLEAR": "CLEAR",
                "AMBIGUOUS": "AMBIGUOUS",
            }

            if classification_str not in classification_types:
                logging.warning(f"Invalid classification value: {classification_str}")
                classification = "FAILED_EVAL"
            else:
                classification = classification_types[classification_str]

        return classification, analysis

    except Exception as e:
        logging.error(f"Error extracting parsing ambiguity eval response: {e}")
        return "FAILED_EVAL", None


@beartype
def submit_batch(
    qs_dataset: QsDataset,
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    num_evals: int = 5,
) -> OpenAIBatchInfo:
    """Submit a batch of questions for ambiguity evaluation."""
    # Create prompts for each question, num_evals times
    prompt_by_qrid = {}
    for qid, question in qs_dataset.question_by_qid.items():
        for eval_idx in range(num_evals):
            qr_id = QuestionResponseId(qid=qid, uuid=f"ambiguity_eval_{eval_idx}")
            prompt = PROMPT.format(question=question.q_str)
            logging.info(f"Sending prompt for question {qid} (eval {eval_idx}): `{prompt}`")
            prompt_by_qrid[qr_id] = prompt

    # Submit batch using OpenAI batch API
    batch_info = submit_openai_batch(
        prompt_by_qrid=prompt_by_qrid,
        instr_id="",
        ds_params=qs_dataset.params,
        evaluated_model_id="ambiguity_eval",
        evaluated_sampling_params=sampling_params,
        evaluator_model_id=evaluator_model_id,
    )
    return batch_info


@beartype
def process_batch(batch_info: OpenAIBatchInfo) -> AmbiguityEval:
    """Process a batch of responses and create an AmbiguityEval object."""
    # Initialize data structures for multiple evaluations
    ambiguity_by_qid: dict[str, list[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]]] = {}
    analysis_by_qid: dict[str, list[str | None]] = {}
    final_ambiguity_by_qid: dict[str, Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]] = {}

    # Process the batch
    results = process_openai_batch_results(batch_info)
    
    # Group results by qid
    for qr_id, response in results:
        qid = qr_id.qid
        classification, analysis = extract_classification(response)
        
        # Initialize lists for this qid if not already present
        if qid not in ambiguity_by_qid:
            ambiguity_by_qid[qid] = []
            analysis_by_qid[qid] = []
        
        # Add this evaluation's results
        ambiguity_by_qid[qid].append(classification)
        analysis_by_qid[qid].append(analysis)

    # Determine final ambiguity for each question
    for qid in ambiguity_by_qid.keys():
        if any(result == "AMBIGUOUS" for result in ambiguity_by_qid[qid]):
            final_ambiguity_by_qid[qid] = "AMBIGUOUS"
        elif all(result == "CLEAR" for result in ambiguity_by_qid[qid]):
            final_ambiguity_by_qid[qid] = "CLEAR"
        else:
            final_ambiguity_by_qid[qid] = "FAILED_EVAL"

    return AmbiguityEval(
        ambiguity_by_qid=ambiguity_by_qid,
        analysis_by_qid=analysis_by_qid,
        final_ambiguity_by_qid=final_ambiguity_by_qid,
        model_id=batch_info.evaluator_model_id or batch_info.evaluated_model_id,
        instr_id=batch_info.instr_id,
        ds_params=batch_info.ds_params,
        sampling_params=batch_info.evaluated_sampling_params,
    )


@click.group()
def cli() -> None:
    """Evaluate questions for ambiguity using OpenAI's batch API."""
    pass


@cli.command()
@click.option("--evaluator-model-id", default="gpt-4o")
@click.option("--temperature", default=0.7)
@click.option("--top-p", default=0.9)
@click.option("--max-tokens", default=1000)
@click.option("-n", "--num-evals", default=10, help="Number of evaluations per question")
@click.option("--test", is_flag=True, help="Test mode: only process 10 questions from first dataset")
@click.option("-v", "--verbose", is_flag=True)
def submit(
    evaluator_model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_evals: int,
    test: bool,
    verbose: bool,
) -> None:
    """Submit batches of questions for ambiguity evaluation."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_tokens,
    )

    # Find all question files
    question_files = list(DATA_DIR.glob("questions/*/*.yaml"))
    logging.info(f"Found {len(question_files)} question files")

    if test:
        question_files = question_files[:1]
        logging.info("Test mode: using only first dataset")

    for question_file in question_files:
        try:
            qs_dataset = QsDataset.load_from_path(question_file)
            
            if test:
                # Take only first 10 questions
                test_questions = dict(itertools.islice(qs_dataset.question_by_qid.items(), 10))
                qs_dataset.question_by_qid = test_questions
                logging.info(f"Test mode: using {len(test_questions)} questions")

            batch_info = submit_batch(
                qs_dataset=qs_dataset,
                evaluator_model_id=evaluator_model_id,
                sampling_params=sampling_params,
                num_evals=num_evals,
            )
            logging.info(f"Submitted batch {batch_info.batch_id} for {question_file}")
            logging.info(f"Batch info saved to {batch_info.save()}")
        except Exception as e:
            logging.error(f"Error processing {question_file}: {e}")


@cli.command()
@click.option("-v", "--verbose", is_flag=True)
def process(verbose: bool) -> None:
    """Process all batches of ambiguity evaluation responses."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Find all batch info files
    batch_files = list(DATA_DIR.glob("openai_batches/**/ambiguity_eval*.yaml"))
    logging.info(f"Found {len(batch_files)} batch files to process")

    for batch_path in batch_files:
        try:
            batch_info = OpenAIBatchInfo.load(batch_path)
            ambiguity_eval = process_batch(batch_info)
            saved_path = ambiguity_eval.save()
            logging.info(f"Processed batch {batch_info.batch_id}")
            logging.info(f"Results saved to {saved_path}")
        except Exception as e:
            logging.error(f"Error processing {batch_path}: {e}")


if __name__ == "__main__":
    cli()
