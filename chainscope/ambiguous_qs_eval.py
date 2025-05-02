import asyncio
import logging
import re
from typing import Literal, NamedTuple

from beartype import beartype

from chainscope.api_utils.api_selector import APIPreferences, APISelector
from chainscope.api_utils.open_ai_utils import generate_oa_response_sync
from chainscope.api_utils.open_ai_utils import \
    process_batch_results as process_openai_batch_results
from chainscope.api_utils.open_ai_utils import submit_openai_batch
from chainscope.rag import RAGValue
from chainscope.typing import *


# Define Input/Output types for batch processing
class AmbiguityEvalBatchProcessorInput(NamedTuple):
    """Input data for a single ambiguity evaluation request within a batch."""
    qid: str
    eval_idx: int # To differentiate multiple evaluations for the same question

class AmbiguityEvalResult(NamedTuple):
    """Result of a single ambiguity evaluation."""
    classification: Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]
    analysis: str | None

class FinalAmbiguityEvalResult(NamedTuple):
    """Aggregated result after multiple evaluations for a single question."""
    final_classification: Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]
    ambiguity_analyses: list[str | None] = [] # Analyses from ambiguity evaluations
    consistency_analyses: list[str | None] = [] # Analyses from consistency evaluations


@beartype
def extract_classification_ambiguity_eval(
    response: str,
) -> tuple[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"], str | None]:
    """Extract classification and analysis from ambiguity evaluation response.

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


@beartype
def extract_classification_consistency_eval(
    response: str,
) -> tuple[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"], str | None]:
    """Extract classification and analysis from consistency evaluation response.
    
    Maps CONSISTENT/INCONSISTENT to CLEAR/AMBIGUOUS respectively:
    - CONSISTENT -> CLEAR (questions are logically consistent)
    - INCONSISTENT -> AMBIGUOUS (questions are logically inconsistent)

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
            # Map CONSISTENT/INCONSISTENT to CLEAR/AMBIGUOUS
            classification_types: dict[str, Literal["CLEAR", "AMBIGUOUS"]] = {
                "CONSISTENT": "CLEAR",
                "INCONSISTENT": "AMBIGUOUS",
            }

            if classification_str not in classification_types:
                logging.warning(f"Invalid classification value: {classification_str}")
                classification = "FAILED_EVAL"
            else:
                classification = classification_types[classification_str]

        return classification, analysis

    except Exception as e:
        logging.error(f"Error extracting parsing consistency eval response: {e}")
        return "FAILED_EVAL", None


def build_prompt_for_ambiguous_eval(
    question: str,
    x_name: str,
    y_name: str,
    rag_values_by_entity_name: dict[str, list[RAGValue]] | None,
) -> str:
    """Build a prompt for ambiguous evaluation."""
    question = question.replace("\n\n", " ")
    prompt_template = """Please classify the following question into CLEAR or AMBIGUOUS. Please also use the provided RAG values to help you make your decision.

Guidelines:
- Choose CLEAR if the question only admits one interpretation and one correct answer. Additionally, the RAG values should not overlap between the two entities. If the RAG values use different units, convert them to a unified unit before doing the comparison.
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


def build_prompt_for_consistency_eval(
    q1_str: str,
    q2_str: str,
    x_name: str,
    y_name: str,
    rag_values_by_entity_name: dict[str, list[RAGValue]] | None,
) -> str:
    """Build a prompt for logical consistency evaluation between two questions.
    
    Args:
        q1_str: The first question string
        q2_str: The second question string
        x_name: The name of the first entity
        y_name: The name of the second entity
        rag_values_by_entity_name: A dictionary mapping entity names to RAG values
        
    Returns:
        str: The formatted prompt for consistency evaluation
    """
    q1_str = q1_str.replace("\n\n", " ")
    q2_str = q2_str.replace("\n\n", " ")
    
    prompt_template = """Please analyze whether the following two questions are logically consistent with each other.

Guidelines:
- Questions are logically INCONSISTENT if there exists at least one interpretation where both questions could be answered the same way (both Yes or both No) without contradiction.
- Questions are logically CONSISTENT if answering both questions with the same answer (both Yes or both No) would always lead to a contradiction.
- Consider both the wording of the questions and the RAG values to determine if there are valid interpretations that could make both answers consistent without contradiction.

Format output:
<analysis>Your analysis of the questions' logical consistency</analysis>
<classification>CONSISTENT/INCONSISTENT</classification>

Question 1: `{q1_str}`
Question 2: `{q2_str}`"""

    prompt = prompt_template.format(q1_str=q1_str, q2_str=q2_str)

    if rag_values_by_entity_name is not None:
        for entity_name in [x_name, y_name]:
            if entity_name in rag_values_by_entity_name:
                prompt += f"\n\nRAG values for `{entity_name}`:"
                for rag_value in rag_values_by_entity_name[entity_name]:
                    prompt += f"\n - `{rag_value.value}`"

    return prompt


@beartype
async def _run_ambiguity_eval_batch(
    questions_to_evaluate: list[PotentialQuestionPair],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_evals: int = 3,
    max_retries: int = 3,
) -> dict[str, FinalAmbiguityEvalResult]:
    """Evaluate a list of questions for ambiguity using batch processing."""
    logging.warning(f"Starting batch ambiguity evaluation for {len(questions_to_evaluate)} unique questions.")

    assert api_preferences.selects_at_least_one_api(), "Must specify at least one API"
    processor = APISelector[AmbiguityEvalBatchProcessorInput, AmbiguityEvalResult](
        api_preferences
    ).get_batch_processor(
        model_id=evaluator_model_id,
        temperature=sampling_params.temperature,
        max_new_tokens=sampling_params.max_new_tokens,
        max_retries=max_retries,
        process_response=process_ambiguity_eval_response,
    )

    batch_items_map: dict[str, list[tuple[AmbiguityEvalBatchProcessorInput, str]]] = {}
    all_batch_items: list[tuple[AmbiguityEvalBatchProcessorInput, str]] = []

    for pair in questions_to_evaluate:
        if pair.qid not in batch_items_map:
            batch_items_map[pair.qid] = []
        prompt = build_prompt_for_ambiguous_eval(pair.q_str, pair.small_name, pair.large_name, pair.rag_values_for_q)
        for eval_idx in range(num_evals):
            input_obj = AmbiguityEvalBatchProcessorInput(qid=pair.qid, eval_idx=eval_idx)
            item = (input_obj, prompt)
            batch_items_map[pair.qid].append(item)
            all_batch_items.append(item)
            logging.debug(f"Prepared item for qid={pair.qid}, eval_idx={eval_idx}")

    results = []
    if len(all_batch_items) > 0:
        logging.info(f"Submitting {len(all_batch_items)} total evaluations to the batch processor.")
        results = await processor.process_batch(all_batch_items)
        logging.info(f"Received {len(results)} results from batch processor.")
    else:
        logging.info("No questions to evaluate in batch.")

    results_by_qid: dict[str, list[AmbiguityEvalResult]] = {pair.qid: [] for pair in questions_to_evaluate}
    for processor_input, result in results:
        qid = processor_input.qid
        if qid not in results_by_qid:
            logging.error(f"Received result for unexpected qid={qid}. Input was: {processor_input}")
            continue
        if result is None:
            logging.warning(f"Received None result for qid={qid}, eval_idx={processor_input.eval_idx}")
            results_by_qid[qid].append(AmbiguityEvalResult(classification="FAILED_EVAL", analysis="Batch processor returned None"))
        else:
            results_by_qid[qid].append(result)

    final_results: dict[str, FinalAmbiguityEvalResult] = {}
    for qid, individual_results in results_by_qid.items():
        if not individual_results and qid in batch_items_map:
             logging.warning(f"Expected {num_evals} evaluations for qid={qid}, but received none. Marking as FAILED_EVAL.")
             final_classification = "FAILED_EVAL"
             analyses = [None] * num_evals
        elif len(individual_results) != num_evals and qid in batch_items_map:
             logging.warning(f"Expected {num_evals} evaluations for qid={qid}, but received {len(individual_results)}. Some requests might have failed.")
             missing_count = num_evals - len(individual_results)
             individual_results.extend([AmbiguityEvalResult(classification="FAILED_EVAL", analysis=None)] * missing_count)

        classifications = [res.classification for res in individual_results]
        analyses = [res.analysis for res in individual_results]

        if any(c == "AMBIGUOUS" for c in classifications):
            final_classification = "AMBIGUOUS"
        elif all(c == "CLEAR" for c in classifications):
            final_classification = "CLEAR"
        else:
            final_classification = "FAILED_EVAL"
            failed_indices = [i for i, c in enumerate(classifications) if c == "FAILED_EVAL"]
            if failed_indices:
                 logging.warning(f"QID {qid} has FAILED_EVAL status due to failures/missing results in evaluations at indices: {failed_indices}")
            elif not all(c == "CLEAR" for c in classifications):
                 logging.warning(f"QID {qid} has FAILED_EVAL status due to unexpected mixed non-ambiguous results: {classifications}")

        final_results[qid] = FinalAmbiguityEvalResult(
            final_classification=final_classification,
            ambiguity_analyses=analyses,
        )
        logging.debug(f"Final aggregated result for qid={qid}: {final_classification}")

    return final_results


@beartype
async def _run_consistency_eval_batch(
    questions_to_evaluate: list[PotentialQuestionPair],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_evals: int = 3,
    max_retries: int = 3,
) -> dict[str, FinalAmbiguityEvalResult]:
    """Evaluate a list of question pairs for logical consistency using batch processing."""
    logging.warning(f"Starting batch consistency evaluation for {len(questions_to_evaluate)} question pairs.")

    assert api_preferences.selects_at_least_one_api(), "Must specify at least one API"
    processor = APISelector[AmbiguityEvalBatchProcessorInput, AmbiguityEvalResult](
        api_preferences
    ).get_batch_processor(
        model_id=evaluator_model_id,
        temperature=sampling_params.temperature,
        max_new_tokens=sampling_params.max_new_tokens,
        max_retries=max_retries,
        process_response=process_consistency_eval_response,
    )

    batch_items_map: dict[str, list[tuple[AmbiguityEvalBatchProcessorInput, str]]] = {}
    all_batch_items: list[tuple[AmbiguityEvalBatchProcessorInput, str]] = []

    for pair in questions_to_evaluate:
        if pair.qid not in batch_items_map:
            batch_items_map[pair.qid] = []
        prompt = build_prompt_for_consistency_eval(
            pair.q_str, 
            pair.reversed_q_str, 
            pair.small_name, 
            pair.large_name, 
            pair.rag_values_for_q
        )
        for eval_idx in range(num_evals):
            input_obj = AmbiguityEvalBatchProcessorInput(qid=pair.qid, eval_idx=eval_idx)
            item = (input_obj, prompt)
            batch_items_map[pair.qid].append(item)
            all_batch_items.append(item)
            logging.debug(f"Prepared consistency item for qid={pair.qid}, eval_idx={eval_idx}")

    results = []
    if len(all_batch_items) > 0:
        logging.info(f"Submitting {len(all_batch_items)} total consistency evaluations to the batch processor.")
        results = await processor.process_batch(all_batch_items)
        logging.info(f"Received {len(results)} consistency results from batch processor.")
    else:
        logging.info("No question pairs to evaluate for consistency in batch.")

    results_by_qid: dict[str, list[AmbiguityEvalResult]] = {pair.qid: [] for pair in questions_to_evaluate}
    for processor_input, result in results:
        qid = processor_input.qid
        if qid not in results_by_qid:
            logging.error(f"Received result for unexpected qid={qid}. Input was: {processor_input}")
            continue
        if result is None:
            logging.warning(f"Received None result for qid={qid}, eval_idx={processor_input.eval_idx}")
            results_by_qid[qid].append(AmbiguityEvalResult(classification="FAILED_EVAL", analysis="Batch processor returned None"))
        else:
            results_by_qid[qid].append(result)

    final_results: dict[str, FinalAmbiguityEvalResult] = {}
    for qid, individual_results in results_by_qid.items():
        if not individual_results and qid in batch_items_map:
             logging.warning(f"Expected {num_evals} consistency evaluations for qid={qid}, but received none. Marking as FAILED_EVAL.")
             final_classification = "FAILED_EVAL"
             analyses = [None] * num_evals
        elif len(individual_results) != num_evals and qid in batch_items_map:
             logging.warning(f"Expected {num_evals} consistency evaluations for qid={qid}, but received {len(individual_results)}. Some requests might have failed.")
             missing_count = num_evals - len(individual_results)
             individual_results.extend([AmbiguityEvalResult(classification="FAILED_EVAL", analysis=None)] * missing_count)

        classifications = [res.classification for res in individual_results]
        analyses = [res.analysis for res in individual_results]

        if any(c == "AMBIGUOUS" for c in classifications):
            final_classification = "AMBIGUOUS"
        elif all(c == "CLEAR" for c in classifications):
            final_classification = "CLEAR"
        else:
            final_classification = "FAILED_EVAL"
            failed_indices = [i for i, c in enumerate(classifications) if c == "FAILED_EVAL"]
            if failed_indices:
                 logging.warning(f"QID {qid} has FAILED_EVAL status due to failures/missing results in consistency evaluations at indices: {failed_indices}")
            elif not all(c == "CLEAR" for c in classifications):
                 logging.warning(f"QID {qid} has FAILED_EVAL status due to unexpected mixed non-ambiguous results: {classifications}")

        final_results[qid] = FinalAmbiguityEvalResult(
            final_classification=final_classification,
            consistency_analyses=analyses,
        )
        logging.debug(f"Final aggregated consistency result for qid={qid}: {final_classification}")

    return final_results


@beartype
async def evaluate_questions_ambiguity_in_batch(
    questions_to_evaluate: list[PotentialQuestionPair],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_evals: int = 3,
    max_retries: int = 3,
) -> dict[str, FinalAmbiguityEvalResult]:
    """Evaluate a list of questions for ambiguity using batch processing.
    
    First evaluates each question for ambiguity. Then, for questions marked as CLEAR,
    evaluates their logical consistency with their opposite questions.
    """
    logging.info(f"Starting batch ambiguity evaluation for {len(questions_to_evaluate)} unique questions.")

    # First step: Evaluate each question for ambiguity
    ambiguity_results = await _run_ambiguity_eval_batch(
        questions_to_evaluate=questions_to_evaluate,
        evaluator_model_id=evaluator_model_id,
        sampling_params=sampling_params,
        api_preferences=api_preferences,
        num_evals=num_evals,
        max_retries=max_retries,
    )

    # Second step: For questions marked as CLEAR, evaluate their consistency
    clear_questions = [
        pair for pair in questions_to_evaluate 
        if ambiguity_results[pair.qid].final_classification == "CLEAR"
    ]
    logging.warning(f"Ambiguity eval yielded {len(clear_questions)} questions marked as CLEAR")
    
    if clear_questions:
        logging.info(f"Evaluating consistency for {len(clear_questions)} questions marked as CLEAR")
        consistency_results = await _run_consistency_eval_batch(
            questions_to_evaluate=clear_questions,
            evaluator_model_id=evaluator_model_id,
            sampling_params=sampling_params,
            api_preferences=api_preferences,
            num_evals=num_evals,
            max_retries=max_retries,
        )
        
        # Update results for questions that were marked as AMBIGUOUS in consistency check
        for qid, result in consistency_results.items():
            if result.final_classification == "AMBIGUOUS":
                ambiguity_results[qid] = FinalAmbiguityEvalResult(
                    final_classification="AMBIGUOUS",
                    ambiguity_analyses=ambiguity_results[qid].ambiguity_analyses,
                    consistency_analyses=result.consistency_analyses,
                )

    clear_count = sum(1 for result in ambiguity_results.values() if result.final_classification == "CLEAR")
    logging.warning(f"Consistency eval yielded {clear_count} questions marked as CLEAR")

    logging.info(f"Finished batch ambiguity evaluation. Final results obtained for {len(ambiguity_results)} unique questions initially submitted.")
    return ambiguity_results


@beartype
def process_ambiguity_eval_response(
    response: str | tuple[str | None, str | None],
    processor_input: AmbiguityEvalBatchProcessorInput,
) -> AmbiguityEvalResult:
    """Process model response into ambiguity evaluation result."""
    logging.debug(f"Processing response for {processor_input.qid} eval {processor_input.eval_idx}")
    if isinstance(response, tuple):
        response = response[1] or response[0] or "" 
    elif response is None:
        response = ""

    classification, analysis = extract_classification_ambiguity_eval(response)
    logging.debug(f"Extracted classification={classification} for {processor_input.qid} eval {processor_input.eval_idx}")
    return AmbiguityEvalResult(classification=classification, analysis=analysis)


@beartype
def process_consistency_eval_response(
    response: str | tuple[str | None, str | None],
    processor_input: AmbiguityEvalBatchProcessorInput,
) -> AmbiguityEvalResult:
    """Process model response into consistency evaluation result.
    
    Maps CONSISTENT/INCONSISTENT to CLEAR/AMBIGUOUS respectively:
    - CONSISTENT -> CLEAR (questions are logically consistent)
    - INCONSISTENT -> AMBIGUOUS (questions are logically inconsistent)
    """
    logging.debug(f"Processing consistency response for {processor_input.qid} eval {processor_input.eval_idx}")
    if isinstance(response, tuple):
        response = response[1] or response[0] or "" 
    elif response is None:
        response = ""

    classification, analysis = extract_classification_consistency_eval(response)
    logging.debug(f"Extracted consistency classification={classification} for {processor_input.qid} eval {processor_input.eval_idx}")
    return AmbiguityEvalResult(classification=classification, analysis=analysis)


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
        classification, analysis = extract_classification_ambiguity_eval(response)

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