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
- Choose CLEAR if the question only admits one interpretation
- Choose AMBIGUOUS if the question admits more than one interpretation, which could lead to different answers.

Format output:
<explanation>You analysis of the question</explanation>
<classification>CLEAR/AMBIGUOUS</classification>

Question: `{question}`"""


@beartype
def extract_classification(response: str) -> tuple[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"], str | None]:
    """Extract classification and explanation from response.
    
    Returns:
        tuple: (classification, explanation)
            - classification: CLEAR, AMBIGUOUS, or FAILED_EVAL
            - explanation: The explanation string or None if failed to extract
    """
    try:
        explanation_match = re.search(r"<explanation>(.*?)</explanation>", response, re.DOTALL)
        classification_match = re.search(
            r"<classification>(.*?)</classification>", response, re.DOTALL
        )

        if not explanation_match:
            logging.warning(f"Could not parse explanation: {response}")
            explanation = None
        else:
            explanation = explanation_match.group(1).strip()
            if not explanation:
                logging.warning(f"Got an empty explanation")
                explanation = None

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

        return classification, explanation

    except Exception as e:
        logging.error(f"Error extracting parsing ambiguity eval response: {e}")
        return "FAILED_EVAL", None


@beartype
def submit_batch(
    qs_dataset: QsDataset,
    evaluator_model_id: str,
    sampling_params: SamplingParams,
) -> OpenAIBatchInfo:
    """Submit a batch of questions for ambiguity evaluation."""
    # Create prompts for each question
    prompt_by_qrid = {}
    for qid, question in qs_dataset.question_by_qid.items():
        qr_id = QuestionResponseId(qid=qid, uuid="ambiguity_eval")
        prompt = PROMPT.format(question=question.q_str)
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
    results = process_openai_batch_results(batch_info)
    
    ambiguity_by_qid: dict[str, Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]] = {}
    explanation_by_qid: dict[str, str | None] = {}

    for qr_id, response in results:
        classification, explanation = extract_classification(response)
        ambiguity_by_qid[qr_id.qid] = classification
        explanation_by_qid[qr_id.qid] = explanation

    return AmbiguityEval(
        ambiguity_by_qid=ambiguity_by_qid,
        explanation_by_qid=explanation_by_qid,
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
@click.option("--test", is_flag=True, help="Test mode: only process 10 questions from first dataset")
def submit(
    evaluator_model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    test: bool,
) -> None:
    """Submit batches of questions for ambiguity evaluation."""

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
