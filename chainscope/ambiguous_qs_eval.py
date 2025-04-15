import logging
import re
from typing import Literal

from beartype import beartype

from chainscope.api_utils.open_ai_utils import generate_oa_response_sync
from chainscope.api_utils.open_ai_utils import \
    process_batch_results as process_openai_batch_results
from chainscope.api_utils.open_ai_utils import submit_openai_batch
from chainscope.rag import RAGValue
from chainscope.typing import *


@beartype
def extract_classification(
    response: str,
) -> tuple[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"], str | None]:
    """Extract classification and analysis from response.

    Returns:
        tuple: (classification, analysis)
            - classification: CLEAR, AMBIGUOUS, or FAILED_EVAL
            - analysis: The analysis string or None if failed to extract
    """
    try:
        analysis_match = re.search(
            r"<analysis>(.*?)(?:</analysis>|<classification>)", response, re.DOTALL
        )
        classification_match = re.search(
            r"<classification>(.*?)</classification>", response, re.DOTALL
        )

        if not analysis_match:
            logging.warning(f"Could not parse analysis: {response}")
            analysis = None
        else:
            analysis = analysis_match.group(1).strip()
            if not analysis:
                logging.warning("Got an empty analysis")
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


def build_prompt_for_ambiguous_eval(
    question: str,
    x_name: str,
    y_name: str,
    rag_values_by_entity_name: dict[str, list[RAGValue]] | None,
) -> str:
    """Build a prompt for ambiguous evaluation."""
    prompt_template = """Please classify the following question into CLEAR or AMBIGUOUS. Please also use the provided RAG values to help you make your decision.

Guidelines:
- Choose CLEAR if the question only admits one interpretation and one correct answer. Additionally, the RAG values should not overlap between the two entities.
- Choose AMBIGUOUS if the question admits more than one interpretation, which could lead to different answers, or if the RAG values overlap between the two entities.

Format output:
<analysis>Your analysis of the question</analysis>
<classification>CLEAR/AMBIGUOUS</classification>

Question: `{question}`"""

    prompt = prompt_template.format(question=question)

    if rag_values_by_entity_name is not None:
        for entity_name in [x_name, y_name]:
            if entity_name in rag_values_by_entity_name:
                prompt += f"\n\nRAG values for `{entity_name}`:"
                for rag_value in rag_values_by_entity_name[entity_name]:
                    prompt += f"\n - `{rag_value.value}`"

    return prompt


@beartype
def evaluate_single_question(
    q_str: str,
    x_name: str,
    y_name: str,
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    num_evals: int = 10,
    short_circuit_on_ambiguous: bool = True,
    rag_values_by_entity_name: dict[str, list[RAGValue]] | None = None,
) -> tuple[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"], str | None]:
    """Evaluate a single question for ambiguity synchronously.

    Args:
        q_str: The question string to evaluate
        x_name: The name of the first entity
        y_name: The name of the second entity
        evaluator_model_id: The model ID to use for evaluation
        sampling_params: The sampling parameters for the model
        num_evals: Number of evaluations to perform (default: 10)
        short_circuit_on_ambiguous: Whether to short circuit and return AMBIGUOUS if any evaluation is AMBIGUOUS (default: True)
        rag_values_by_entity_name: A dictionary mapping entity names to RAG values (default: None)
    Returns:
        tuple: (classification, analysis)
            - classification: CLEAR, AMBIGUOUS, or FAILED_EVAL
            - analysis: The analysis string or None if failed to extract
    """
    prompt = build_prompt_for_ambiguous_eval(q_str, x_name, y_name, rag_values_by_entity_name)
    logging.info(f"Using prompt `{prompt}`")
    
    classifications: list[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]] = []
    analyses: list[str | None] = []
    
    for _ in range(num_evals):
        response = generate_oa_response_sync(
            prompt=prompt,
            model_id=evaluator_model_id,
            temperature=sampling_params.temperature,
            max_new_tokens=sampling_params.max_new_tokens,
        )
        if response is None:
            logging.warning(f"Got None response for question: {q_str}")
            classifications.append("FAILED_EVAL")
            analyses.append(None)
            continue
            
        classification, analysis = extract_classification(response)
        classifications.append(classification)
        analyses.append(analysis)
        if short_circuit_on_ambiguous and classification == "AMBIGUOUS":
            return classification, analysis
    
    # Use same logic as process_batch to determine final classification
    if any(result == "AMBIGUOUS" for result in classifications):
        final_classification = "AMBIGUOUS"
    elif all(result == "CLEAR" for result in classifications):
        final_classification = "CLEAR"
    else:
        final_classification = "FAILED_EVAL"
    
    final_analysis = next(item[0] for item in zip(analyses, classifications) if item[1] == final_classification)
    
    return final_classification, final_analysis


@beartype
def submit_batch(
    qs_dataset: QsDataset,
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    num_evals: int = 10,
) -> OpenAIBatchInfo:
    """Submit a batch of questions for ambiguity evaluation."""
    # Create prompts for each question, num_evals times
    prompt_by_qrid = {}
    for qid, question in qs_dataset.question_by_qid.items():
        for eval_idx in range(num_evals):
            qr_id = QuestionResponseId(qid=qid, uuid=f"ambiguity_eval_{eval_idx}")
            prompt = build_prompt_for_ambiguous_eval(question.q_str, question.x_name, question.y_name, None)
            logging.info(
                f"Sending prompt for question {qid} (eval {eval_idx}): `{prompt}`"
            )
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