import asyncio
import hashlib
import logging

from chainscope.ambiguous_qs_eval import (FinalAmbiguityEvalResult,
                                          evaluate_questions_in_batch)
from chainscope.api_utils.api_selector import APIPreferences
from chainscope.rag import RAGValue
from chainscope.typing import *


def make_yes_no_question_pair(
    template: str,
    open_ended_template: str,
    yes_question_by_qid: dict[str, Question],
    no_question_by_qid: dict[str, Question],
    x_name: str,
    y_name: str,
    x_value: int | float,
    y_value: int | float,
):
    """Generate a pair of complementary YES/NO questions by swapping the order of compared items.

    For each pair of items (x,y), generates:
    - YES question comparing x to y
    - NO question comparing y to x
    Questions are stored in the provided dictionaries keyed by their SHA256 hash.

    Args:
        template: Question template with {x} and {y} placeholders
        open_ended_template: Open-ended version of the question template
        yes_question_by_qid: Dict to store YES questions
        no_question_by_qid: Dict to store NO questions
        x_name: Name of the first item
        y_name: Name of the second item
        x_value: Value of the first item
        y_value: Value of the second item
    """
    # Generate YES question (x compared to y)
    yes_q_str = template.format(x=x_name, y=y_name)
    yes_q_str_open_ended = open_ended_template.format(x=x_name, y=y_name)
    yes_qid = hashlib.sha256(yes_q_str.encode()).hexdigest()
    yes_question_by_qid[yes_qid] = Question(
        q_str=yes_q_str,
        q_str_open_ended=yes_q_str_open_ended,
        x_name=x_name,
        y_name=y_name,
        x_value=x_value,
        y_value=y_value,
    )

    # Generate NO question (y compared to x)
    no_q_str = template.format(x=y_name, y=x_name)
    no_q_str_open_ended = open_ended_template.format(x=y_name, y=x_name)
    no_qid = hashlib.sha256(no_q_str.encode()).hexdigest()
    no_question_by_qid[no_qid] = Question(
        q_str=no_q_str,
        q_str_open_ended=no_q_str_open_ended,
        x_name=y_name,
        y_name=x_name,
        x_value=y_value,
        y_value=x_value,
    )


def _filter_entities_by_popularity(
    properties: Properties,
    prop_id: str,
    entity_popularity_filter: int | None,
) -> Properties:
    """Filter entities based on popularity threshold."""
    if entity_popularity_filter is None:
        return properties

    try:
        prop_eval = PropEval.load_id(prop_id)
        properties.value_by_name = {
            entity_name: entity_value
            for entity_name, entity_value in properties.value_by_name.items()
            if prop_eval.popularity_by_entity_name[entity_name] >= entity_popularity_filter
        }
        assert len(properties.value_by_name) > 1, f"Not enough entities left after filtering by popularity of {entity_popularity_filter} for {prop_id}"
        logging.info(f"After filtering by popularity, we have {len(properties.value_by_name)} entities for {prop_id}")
        return properties
    except FileNotFoundError:
        raise ValueError(f"Entity popularity filter set to {entity_popularity_filter} but prop eval not found for {prop_id}")

def _filter_entities_by_rag_values(
    properties: Properties,
    rag_values_map: dict[str, list[RAGValue]] | None,
    prop_id: str,
) -> Properties:
    """Filter entities that have RAG values."""
    if rag_values_map is None:
        return properties

    properties.value_by_name = {
        entity_name: entity_value
        for entity_name, entity_value in properties.value_by_name.items()
        if entity_name in rag_values_map and len(rag_values_map[entity_name]) > 0
    }
    logging.info(f"After filtering by entities with RAG values, we have {len(properties.value_by_name)} entities for {prop_id}")
    return properties

def _generate_potential_pairs(
    properties: Properties,
    all_sorted_values: list[tuple[str, int | float]],
    min_percent_value_diff: float | None,
    rag_values_map: dict[str, list[RAGValue]] | None,
) -> tuple[list[tuple[tuple[str, int | float], tuple[str, int | float]]], list[tuple[str, str, str, str, dict[str, list[RAGValue]] | None]], dict[str, tuple[tuple[str, int | float], tuple[str, int | float]]]]:
    """Generate potential pairs and prepare questions for ambiguity evaluation."""
    potential_small_large_pairs = []
    questions_for_ambiguity_eval = []
    pair_info_by_qid: dict[str, tuple[tuple[str, int | float], tuple[str, int | float]]] = {}

    # Calculate value range for percentage difference filtering
    if min_percent_value_diff is not None:
        min_val = all_sorted_values[0][1]
        max_val = all_sorted_values[-1][1]
        value_range = max_val - min_val
        min_absolute_diff = value_range * (min_percent_value_diff / 100)
        logging.info(f"Value range: {value_range}, minimum required difference: {min_absolute_diff}")

    logging.info(f"Generating potential pairs for ambiguity evaluation...")
    for small_idx, (small_name, small_value) in enumerate(all_sorted_values):
        logging.info(f"Generating questions for entity `{small_name}` ({small_value}), index {small_idx}/{len(all_sorted_values)}")
        
        for large_idx, (large_name, large_value) in enumerate(all_sorted_values[small_idx + 1:]):
            logging.info(f"Comparing {small_name} ({small_value}) and {large_name} ({large_value}), index {large_idx}/{len(all_sorted_values) - small_idx - 1}")
            
            if small_value == large_value:
                logging.info(f"Skipping {small_name} and {large_name} because values are equal ({small_value})")
                continue
            
            if min_percent_value_diff is not None and abs(large_value - small_value) < min_absolute_diff:
                logging.info(
                    f"Skipping {small_name} ({small_value}) and {large_name} ({large_value}) "
                    f"because difference ({abs(large_value - small_value)}) is less than "
                    f"minimum required ({min_absolute_diff})"
                )
                continue

            current_pair = ((small_name, small_value), (large_name, large_value))
            potential_small_large_pairs.append(current_pair)

            q_str_for_eval = properties.gt_question.format(x=small_name, y=large_name)
            qid_for_eval = hashlib.sha256(q_str_for_eval.encode()).hexdigest()

            rag_values_for_q = None
            if rag_values_map:
                rag_values_for_q = {
                    name: rag_values_map.get(name, [])
                    for name in [small_name, large_name]
                }
            
            if qid_for_eval not in pair_info_by_qid:
                questions_for_ambiguity_eval.append(
                    (qid_for_eval, q_str_for_eval, small_name, large_name, rag_values_for_q)
                )
                pair_info_by_qid[qid_for_eval] = current_pair

    logging.info(f"Generated {len(potential_small_large_pairs)} potential pairs before ambiguity filtering.")
    return potential_small_large_pairs, questions_for_ambiguity_eval, pair_info_by_qid

def _filter_by_ambiguity(
    potential_small_large_pairs: list[tuple[tuple[str, int | float], tuple[str, int | float]]],
    questions_for_ambiguity_eval: list[tuple[str, str, str, str, dict[str, list[RAGValue]] | None]],
    pair_info_by_qid: dict[str, tuple[tuple[str, int | float], tuple[str, int | float]]],
    remove_ambiguous: bool,
    evaluator_model_id: str,
    evaluator_sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_ambiguity_evals: int,
    max_ambiguity_eval_retries: int,
) -> list[tuple[tuple[str, int | float], tuple[str, int | float]]]:
    """Filter pairs based on ambiguity evaluation."""
    if not remove_ambiguous or not questions_for_ambiguity_eval:
        return potential_small_large_pairs

    logging.info(f"Evaluating {len(questions_for_ambiguity_eval)} unique questions for ambiguity using {evaluator_model_id}...")
    ambiguity_results: dict[str, FinalAmbiguityEvalResult] = asyncio.run(evaluate_questions_in_batch(
        questions_to_evaluate=questions_for_ambiguity_eval,
        evaluator_model_id=evaluator_model_id,
        sampling_params=evaluator_sampling_params,
        api_preferences=api_preferences,
        num_evals=num_ambiguity_evals,
        max_retries=max_ambiguity_eval_retries,
    ))

    non_ambiguous_pairs = []
    for qid, result in ambiguity_results.items():
        if result.final_classification == "CLEAR":
            non_ambiguous_pairs.append(pair_info_by_qid[qid])
        else:
            original_q_str = next((q[1] for q in questions_for_ambiguity_eval if q[0] == qid), "UNKNOWN Q STR")
            logging.info(f"Skipping question corresponding to qid '{qid}' (`{original_q_str}`) due to ambiguity evaluation result: {result.final_classification}")

    logging.info(f"Ambiguity evaluation results: {len(non_ambiguous_pairs)} CLEAR pairs out of {len(potential_small_large_pairs)} potential pairs.")
    return non_ambiguous_pairs

def _filter_by_max_comparisons(
    pairs: list[tuple[tuple[str, int | float], tuple[str, int | float]]],
    max_comparisons: int,
) -> list[tuple[tuple[str, int | float], tuple[str, int | float]]]:
    """Filter pairs to have at most max_comparisons per small value."""
    comparisons_per_small_value = {}
    filtered_pairs = []
    for pair in pairs:
        small_name, _ = pair[0]
        if small_name not in comparisons_per_small_value:
            comparisons_per_small_value[small_name] = 0
        comparisons_per_small_value[small_name] += 1
        if comparisons_per_small_value[small_name] <= max_comparisons:
            filtered_pairs.append(pair)

    logging.info(f"After filtering by max_comparisons {max_comparisons}, {len(filtered_pairs)} pairs remain.")
    return filtered_pairs

def _sample_pairs(
    pairs: list[tuple[tuple[str, int | float], tuple[str, int | float]]],
    target_n: int,
) -> list[tuple[tuple[str, int | float], tuple[str, int | float]]]:
    """Sample n pairs from the available pairs."""
    total_pairs = len(pairs)
    if total_pairs == 0:
        logging.warning("No pairs generated after filtering. Skipping generation.")
        return []

    n = min(target_n, total_pairs)
    if n == 0:
        logging.info("No pairs available to sample.")
        return []

    step = total_pairs / n
    indices = [int(i * step) for i in range(n)]
    unique_indices = sorted(list(set(indices)))
    if len(unique_indices) < n:
        logging.warning(f"Generated non-unique indices ({len(indices)} requested, {len(unique_indices)} unique). Using unique indices.")
        indices = unique_indices

    sampled_pairs = [pairs[i] for i in indices]
    logging.info(f"Sampling {len(indices)} pairs from the {total_pairs} available pairs.")
    return sampled_pairs

def _generate_datasets(
    sampled_pairs: list[tuple[tuple[str, int | float], tuple[str, int | float]]],
    properties: Properties,
    prop_id: str,
    max_comparisons: int,
    dataset_suffix: str | None,
) -> dict[tuple[Literal["gt", "lt"], Literal["YES", "NO"]], QsDataset]:
    """Generate final datasets from the sampled pairs."""
    datasets = {}
    for comparison in ["gt", "lt"]:
        template = properties.gt_question if comparison == "gt" else properties.lt_question
        open_ended_template = (
            properties.gt_open_ended_question
            if comparison == "gt"
            else properties.lt_open_ended_question
        )
        yes_question_by_qid = {}
        no_question_by_qid = {}
        
        for pair in sampled_pairs:
            (small_name, small_value), (large_name, large_value) = pair
            if comparison == "lt":
                x_name, y_name = small_name, large_name
                x_value, y_value = small_value, large_value
            else:
                x_name, y_name = large_name, small_name
                x_value, y_value = large_value, small_value
                
            make_yes_no_question_pair(
                template,
                open_ended_template,
                yes_question_by_qid,
                no_question_by_qid,
                x_name=x_name,
                y_name=y_name,
                x_value=x_value,
                y_value=y_value,
            )

        datasets[(comparison, "YES")] = QsDataset(
            question_by_qid=yes_question_by_qid,
            params=DatasetParams(
                prop_id=prop_id,
                comparison=comparison,
                answer="YES",
                max_comparisons=max_comparisons,
                suffix=dataset_suffix,
            ),
        )

        datasets[(comparison, "NO")] = QsDataset(
            question_by_qid=no_question_by_qid,
            params=DatasetParams(
                prop_id=prop_id,
                comparison=comparison,
                answer="NO",
                max_comparisons=max_comparisons,
                suffix=dataset_suffix,
            ),
        )

    logging.info(f"Generated {len(sampled_pairs)} YES/NO pairs for each comparison type (gt/lt).")
    return datasets

def gen_qs(
    prop_id: str,
    n: int,
    max_comparisons: int,
    entity_popularity_filter: int | None,
    min_percent_value_diff: float | None,
    dataset_suffix: str | None,
    remove_ambiguous: bool,
    non_overlapping_rag_values: bool,
    api_preferences: APIPreferences,
    evaluator_model_id: str,
    evaluator_sampling_params: SamplingParams,
    num_ambiguity_evals: int = 1,
    max_ambiguity_eval_retries: int = 1,
) -> dict[tuple[Literal["gt", "lt"], Literal["YES", "NO"]], QsDataset]:
    """Generate comparative questions for a given property.

    For each comparison type (greater than 'gt' and less than 'lt'), generates n pairs of YES/NO questions.
    The questions are generated by:
    1. Filtering entities by how well-known they are if entity_popularity_filter is set
    2. Sorting all values for the property
    3. Creating pairs of items where one value is greater than the other
    4. Taking n evenly spaced pairs to ensure good coverage of the value range
    5. For each pair, generating both YES and NO questions by swapping the order

    Args:
        prop_id: ID of the property to generate questions for
        n: Target number of question pairs to generate for each comparison type
        max_comparisons: Maximum number of comparisons to generate for each item during initial pair generation
        entity_popularity_filter: Minimum popularity rank for entities
        min_percent_value_diff: Minimum required percentage difference between values
        dataset_suffix: Optional suffix for dataset parameters
        remove_ambiguous: Whether to filter out questions deemed ambiguous by an LLM evaluator
        non_overlapping_rag_values: Whether to use RAG values in ambiguity check prompt
        api_preferences: API preferences for the ambiguity evaluator LLM calls
        evaluator_model_id: Model ID for the ambiguity evaluator LLM
        evaluator_sampling_params: Sampling parameters for the ambiguity evaluator LLM
        num_ambiguity_evals: Number of times to evaluate each question for ambiguity
        max_ambiguity_eval_retries: Maximum number of retries for ambiguity evaluation
    Returns:
        Dictionary mapping (comparison, answer) pairs to generated datasets
    """
    properties = Properties.load(prop_id)
    logging.info(f"Generating questions for {prop_id}, aiming for {n} pairs per comparison type.")
    logging.info(f"We have {len(properties.value_by_name)} entities for {prop_id}")

    properties = _filter_entities_by_popularity(
        properties=properties,
        prop_id=prop_id,
        entity_popularity_filter=entity_popularity_filter
    )

    rag_values_map = None
    if remove_ambiguous and non_overlapping_rag_values:
        rag_eval_path = DATA_DIR / "prop_rag_eval" / "T0.0_P0.9_M1000" / f"{prop_id}.yaml"
        rag_eval = PropRAGEval.load(rag_eval_path)
        rag_values_map = rag_eval.values_by_entity_name
        properties = _filter_entities_by_rag_values(
            properties=properties,
            rag_values_map=rag_values_map,
            prop_id=prop_id
        )

    all_sorted_values = sorted(properties.value_by_name.items(), key=lambda x: x[1])
    
    potential_pairs, questions_for_eval, pair_info = _generate_potential_pairs(
        properties=properties,
        all_sorted_values=all_sorted_values,
        min_percent_value_diff=min_percent_value_diff,
        rag_values_map=rag_values_map
    )
    
    non_ambiguous_pairs = _filter_by_ambiguity(
        potential_small_large_pairs=potential_pairs,
        questions_for_ambiguity_eval=questions_for_eval,
        pair_info_by_qid=pair_info,
        remove_ambiguous=remove_ambiguous,
        evaluator_model_id=evaluator_model_id,
        evaluator_sampling_params=evaluator_sampling_params,
        api_preferences=api_preferences,
        num_ambiguity_evals=num_ambiguity_evals,
        max_ambiguity_eval_retries=max_ambiguity_eval_retries
    )
    
    filtered_pairs = _filter_by_max_comparisons(
        pairs=non_ambiguous_pairs,
        max_comparisons=max_comparisons
    )
    sampled_pairs = _sample_pairs(
        pairs=filtered_pairs,
        target_n=n
    )
    
    return _generate_datasets(
        sampled_pairs=sampled_pairs,
        properties=properties,
        prop_id=prop_id,
        max_comparisons=max_comparisons,
        dataset_suffix=dataset_suffix
    )
