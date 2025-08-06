import asyncio
import hashlib
import logging
from datetime import datetime

import yaml

from chainscope.ambiguous_qs_eval import (
    FinalAmbiguityEvalResult, evaluate_questions_ambiguity_in_batch)
from chainscope.api_utils.api_selector import APIPreferences
from chainscope.rag import RAGValue
from chainscope.typing import *

location_min_comparison_limits = {
    "zip": 1,
    "college": 1,
    "structure": 1,
    "city": 1,
    "populated": 10, # Populated area
    "natural": 10, # Natural area
    "county": 10,
}


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
    min_popularity: int | None,
    max_popularity: int | None,
) -> Properties:
    """Filter entities based on popularity threshold."""
    if min_popularity is None and max_popularity is None:
        # No filtering by popularity
        return properties
    
    try:
        prop_eval = PropEval.load_id(prop_id)
        if min_popularity is not None:
            properties.value_by_name = {
                entity_name: entity_value
                for entity_name, entity_value in properties.value_by_name.items()
                if prop_eval.popularity_by_entity_name[entity_name] >= min_popularity
            }
        if max_popularity is not None:
            properties.value_by_name = {
                entity_name: entity_value
                for entity_name, entity_value in properties.value_by_name.items()
                if prop_eval.popularity_by_entity_name[entity_name] <= max_popularity
            }
        assert len(properties.value_by_name) > 1, f"Not enough entities left after filtering by popularity of {min_popularity} and {max_popularity} for {prop_id}"
        logging.warning(f"After filtering by popularity, we have {len(properties.value_by_name)} entities for {prop_id}")
        return properties
    except FileNotFoundError:
        raise ValueError(f"Entity popularity filter set to {min_popularity} but prop eval not found for {prop_id}")

def _filter_entities_by_name(
    properties: Properties,
    prop_id: str,
) -> Properties:
    """Filter entities that don't have a name-surname pair."""
    if "person" in prop_id:
        # Check that we have at least name-surname for each entity, to avoid ambiguous entities like just "Albert" or "Adolf"
        # We'll miss some cases like "Charlemagne", "Napoleon", "Plato", "Aristotle", etc., but it's better than generating bad questions
        properties.value_by_name = {
            entity_name: entity_value
            for entity_name, entity_value in properties.value_by_name.items()
            if " " in entity_name
        }
    
    # If we have two entities with the same prefix up to a opening parenthesis, remove both
    # This is to avoid cases like "Lake Victoria" and "Lake Victoria (Victoria)"
    # Also, they are probably ambiguous or the same entity
    # Make a copy of the original entities
    
    # Process entities in a loop to find ones to remove
    to_remove = set()
    entity_names = set(properties.value_by_name.keys())
    for entity_name in entity_names:
        if entity_name.startswith("("):
            logging.info(f"Removing {entity_name} because it starts with a parenthesis")
            to_remove.add(entity_name)
            continue

        for other_entity_name in entity_names - to_remove:
            if entity_name == other_entity_name:
                continue
            if "(" in other_entity_name:
                prefix = other_entity_name[:other_entity_name.find("(")]
                if entity_name.startswith(prefix):
                    logging.info(f"Removing {entity_name} and {other_entity_name} because they have the same prefix up to a opening parenthesis")
                    to_remove.add(entity_name)
                    to_remove.add(other_entity_name)

    # Filter out the entities that need to be removed
    properties.value_by_name = {
        name: value 
        for name, value in properties.value_by_name.items()
        if name not in to_remove
    }
    
    logging.warning(f"After filtering by name, we have {len(properties.value_by_name)} entities for {prop_id}")
    return properties

def _filter_entities_by_rag_values(
    properties: Properties,
    rag_values_map: dict[str, list[RAGValue]] | None,
    prop_id: str,
    min_rag_values_count: int | None,
) -> Properties:
    """Filter entities that have RAG values."""
    if rag_values_map is None or min_rag_values_count is None:
        return properties

    entities = list(properties.value_by_name.keys())
    for entity_name in entities:
        rag_values = rag_values_map.get(entity_name, [])
        if len(rag_values) < min_rag_values_count:
            logging.info(f"Removing {entity_name} because it has {len(rag_values)} RAG values but min_rag_values_count is {min_rag_values_count}")
            del properties.value_by_name[entity_name]

    logging.warning(f"After filtering by entities with RAG values, we have {len(properties.value_by_name)} entities for {prop_id}")
    return properties

def _are_valid_values_for_property(
    small_name: str,
    small_value: int | float,
    large_name: str,
    large_value: int | float,
    prop_id: str,
) -> bool:
    """Check if the values are valid for the property."""
    places = ["city", "county", "college", "natural", "structure", "zip"]
    if any(place in prop_id for place in places):
        # if there is a comma in any of the names, that's probably the state (if it's two letters)
        # make sure that they are different
        small_state = None
        large_state = None
        if "," in small_name:
            small_state = small_name.split(",")[1].strip()
            if len(small_state) == 2:
                small_state = small_state.upper()
            else:
                small_state = None
        if "," in large_name:
            large_state = large_name.split(",")[1].strip()
            if len(large_state) == 2:
                large_state = large_state.upper()
            else:
                large_state = None
        if small_state is not None and large_state is not None and small_state == large_state:
            # If the states are the same, we don't want to compare them
            return False

    # We set specific minimums when comparing locations, depending on their size.
    if "-long" in prop_id: # checks for longitude comparisons
        # the values should not be too far apart (more than 120 degrees)
        valid = abs(small_value - large_value) <= 120
        # the values should not be close to the limit of -180 or 180, where it gets fuzzy what's west or east of what.
        valid = valid and abs(small_value) < 150 and abs(large_value) < 150
        # check that the values have the minimum required difference
        required_diff = next((limit for location_type, limit in location_min_comparison_limits.items() if location_type in prop_id), None)
        assert required_diff is not None, f"No minimum required difference set for location comparison {prop_id}"
        valid = valid and abs(small_value - large_value) >= required_diff
        return valid
    elif "-lat" in prop_id: # checks for latitude comparisons
        # check that the values have the minimum required difference
        required_diff = next((limit for location_type, limit in location_min_comparison_limits.items() if location_type in prop_id), None)
        assert required_diff is not None, f"No minimum required difference set for location comparison {prop_id}"
        return abs(small_value - large_value) >= required_diff
    elif "age" in prop_id:
        # check that the values have at least 5 years between them
        return abs(small_value - large_value) >= 5
    elif "release" in prop_id:
        # check that the values have at least 2 years between them
        # the year is the first 4 digits of the value
        small_year = int(str(small_value)[:4])
        large_year = int(str(large_value)[:4])
        return abs(small_year - large_year) >= 2
    
    return True

def _convert_to_comparable_value(value: int | float, prop_id: str) -> int | float:
    """Convert YYMMDD format to days if the property is a date property, otherwise return the value as is."""
    date_props = ["release", "pubdate"]
    if not any(date_prop in prop_id for date_prop in date_props):
        return value

    # Convert YYMMDD format (e.g., "19570719") to days
    try:
        yymmdd_str = str(int(value))  # Convert to int first to handle float values
        year = int(yymmdd_str[:4])
        month = int(yymmdd_str[4:6])
        day = int(yymmdd_str[6:8])

        date = datetime(year, month, day)
        base_date = datetime(1900, 1, 1)
    except Exception as e:
        logging.error(f"Failed to convert value {value} to date: {e}")
        raise

    return (date - base_date).days

def _get_value_range(
    prop_id: str,
    properties: Properties,
) -> tuple[list[tuple[str, int | float]], int | float]:
    """Get the range of values for the property."""
    # Convert all values to days if needed and sort
    all_sorted_values: list[tuple[str, int | float]] = sorted(
        [(name, _convert_to_comparable_value(value, prop_id)) for name, value in properties.value_by_name.items()],
        key=lambda x: x[1]
    )
    min_val = all_sorted_values[0][1]
    logging.info(f"Minimum value for {prop_id}: {min_val}")
    max_val = all_sorted_values[-1][1]
    logging.info(f"Maximum value for {prop_id}: {max_val}")
    value_range = max_val - min_val
    logging.info(f"Value range for {prop_id}: {value_range}")
    return all_sorted_values, value_range

def _generate_potential_pairs(
    prop_id: str,
    properties: Properties,
    min_fraction_value_diff: float | None,
    max_fraction_value_diff: float | None,
    rag_values_map: dict[str, list[RAGValue]] | None,
) -> list[PotentialQuestionPair]:
    """Generate potential pairs and all info needed for ambiguity evaluation and sampling."""
    potential_pairs: list[PotentialQuestionPair] = []

    # Calculate value range for fraction difference filtering
    all_sorted_values, value_range = _get_value_range(prop_id, properties)
    min_absolute_diff = None
    max_absolute_diff = None
    if min_fraction_value_diff is not None:
        min_absolute_diff = value_range * min_fraction_value_diff
        logging.info(f"Minimum required difference for {prop_id}: {min_absolute_diff}")
    if max_fraction_value_diff is not None:
        max_absolute_diff = value_range * max_fraction_value_diff
        logging.info(f"Maximum allowed difference for {prop_id}: {max_absolute_diff}")

    if "-lat" in prop_id or "-long" in prop_id:
        required_diff = next((limit for location_type, limit in location_min_comparison_limits.items() if location_type in prop_id), None)
        if required_diff is not None:
            # Add the required difference as a base minimum/maximum
            # Otherwise we might end up with no pairs at all since these two requirements will produce an empty set
            if min_absolute_diff is not None:
                min_absolute_diff += required_diff
            if max_absolute_diff is not None:
                max_absolute_diff += required_diff
            logging.info(f"Setting min_absolute_diff for {prop_id} to {min_absolute_diff} and max_absolute_diff to {max_absolute_diff} due to location comparison")

    logging.info(f"Generating potential pairs for ambiguity evaluation...")
    for small_idx, (small_name, small_value) in enumerate(all_sorted_values):
        logging.info(f"Generating questions for entity `{small_name}` ({small_value}), index {small_idx}/{len(all_sorted_values)}")
        for large_idx, (large_name, large_value) in enumerate(all_sorted_values[small_idx + 1:]):
            logging.info(f"Comparing {small_name} ({small_value}) and {large_name} ({large_value}), index {large_idx}/{len(all_sorted_values) - small_idx - 1}")
            value_diff = abs(large_value - small_value)
            if value_diff == 0:
                logging.info(f"Skipping {small_name} and {large_name} because values are equal ({small_value})")
                continue

            if min_absolute_diff is not None and value_diff < min_absolute_diff:
                logging.info(
                    f"Skipping {small_name} ({small_value}) and {large_name} ({large_value}) "
                    f"because difference ({value_diff}) is less than "
                    f"minimum required ({min_absolute_diff})"
                )
                continue

            if max_absolute_diff is not None and value_diff > max_absolute_diff:
                logging.info(
                    f"Skipping {small_name} ({small_value}) and {large_name} ({large_value}) "
                    f"because difference ({value_diff}) is greater than "
                    f"maximum allowed ({max_absolute_diff})"
                )
                logging.info(f"Skipping the rest of the pairs for {small_name} since we have already surpassed the max_absolute_diff")
                break

            if not _are_valid_values_for_property(
                small_name=small_name,
                small_value=small_value,
                large_name=large_name,
                large_value=large_value,
                prop_id=prop_id,
            ):
                logging.info(f"Skipping {small_name} ({small_value}) and {large_name} ({large_value}) because values are not valid for {prop_id}")
                continue

            q_str = properties.gt_question.format(x=small_name, y=large_name)
            reversed_q_str = properties.gt_question.format(x=large_name, y=small_name)
            qid = hashlib.sha256(q_str.encode()).hexdigest()
            rag_values_for_q = None
            if rag_values_map:
                rag_values_for_q = {
                    name: rag_values_map.get(name, [])
                    for name in [small_name, large_name]
                }
            potential_pairs.append(PotentialQuestionPair(
                qid=qid,
                q_str=q_str,
                reversed_q_str=reversed_q_str,
                small_name=small_name,
                small_value=small_value,
                large_name=large_name,
                large_value=large_value,
                rag_values_for_q=rag_values_for_q,
            ))
    logging.warning(f"Generated {len(potential_pairs)} potential pairs before ambiguity filtering.")
    return potential_pairs

def _sample_pairs(
    pairs: list[PotentialQuestionPair],
    target_n: int,
) -> list[PotentialQuestionPair]:
    """Sample n pairs from the available pairs."""
    total_pairs = len(pairs)
    assert total_pairs > 0, "No pairs available to sample."

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
    sampled_pairs: list[PotentialQuestionPair],
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
            if comparison == "lt":
                x_name, y_name = pair.small_name, pair.large_name
                x_value, y_value = pair.small_value, pair.large_value
            else:
                x_name, y_name = pair.large_name, pair.small_name
                x_value, y_value = pair.large_value, pair.small_value
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
                comparison=comparison,  # type: ignore
                answer="YES",
                max_comparisons=max_comparisons,
                suffix=dataset_suffix,
            ),
        )
        datasets[(comparison, "NO")] = QsDataset(
            question_by_qid=no_question_by_qid,
            params=DatasetParams(
                prop_id=prop_id,
                comparison=comparison,  # type: ignore
                answer="NO",
                max_comparisons=max_comparisons,
                suffix=dataset_suffix,
            ),
        )
    logging.info(f"Generated {len(sampled_pairs)} YES/NO pairs for each comparison type (gt/lt).")
    return datasets

def _get_ambiguity_eval_cache_path(prop_id: str) -> Path:
    """Get the path to the ambiguity evaluation cache for a property."""
    return DATA_DIR / "ambiguity_eval" / "gen_qs_cache" / f"{prop_id}.yaml"

def _load_ambiguity_cache(prop_id: str) -> dict[tuple[str, str], str]:
    """Load ambiguity evaluation cache for a property."""
    cache_path = _get_ambiguity_eval_cache_path(prop_id)
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            # Convert string keys back to tuples
            raw_cache = yaml.safe_load(f) or {}
            return {tuple(k.split("|")): v for k, v in raw_cache.items()}
    except Exception as e:
        logging.error(f"Error loading ambiguity cache from {cache_path}: {e}")
        return {}

def _save_ambiguity_cache(prop_id: str, cache: dict[tuple[str, str], str]) -> None:
    """Save ambiguity evaluation cache for a property."""
    cache_path = _get_ambiguity_eval_cache_path(prop_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Convert tuple keys to strings for YAML serialization
        serializable_cache = {f"{k[0]}|{k[1]}": v for k, v in cache.items()}
        with open(cache_path, "w") as f:
            yaml.safe_dump(serializable_cache, f)
    except Exception as e:
        logging.error(f"Error saving ambiguity cache to {cache_path}: {e}")

def gen_qs(
    prop_id: str,
    n: int,
    max_comparisons: int,
    min_popularity: int | None,
    max_popularity: int | None,
    min_fraction_value_diff: float | None,
    max_fraction_value_diff: float | None,
    dataset_suffix: str | None,
    remove_ambiguous: Literal["no", "enough-comparisons", "enough-pairs"],
    non_overlapping_rag_values: bool,
    min_rag_values_count: int | None,
    api_preferences: APIPreferences,
    evaluator_model_id: str,
    evaluator_sampling_params: SamplingParams,
    num_ambiguity_evals: int = 1,
    max_ambiguity_eval_retries: int = 1,
) -> dict[tuple[Literal["gt", "lt"], Literal["YES", "NO"]], QsDataset]:
    """Generate comparative questions for a given property.

    For each comparison type (greater than 'gt' and less than 'lt'), generates n pairs of YES/NO questions.
    The questions are generated by:
    1. Filtering entities by how well-known they are if min_popularity is set
    2. Sorting all values for the property
    3. Creating pairs of items where one value is greater than the other
    4. Iteratively evaluating ambiguity in batches, only as needed, until enough non-ambiguous pairs are found for each entity
    5. Taking n evenly spaced pairs to ensure good coverage of the value range
    6. For each pair, generating both YES and NO questions by swapping the order

    Args:
        prop_id: ID of the property to generate questions for
        n: Target number of question pairs to generate for each comparison type
        max_comparisons: Maximum number of comparisons to generate for each item during initial pair generation
        min_popularity: Minimum popularity rank for entities
        max_popularity: Maximum popularity rank for entities
        min_fraction_value_diff: Minimum required fraction difference between values
        max_fraction_value_diff: Maximum required fraction difference between values
        min_rag_values_count: Minimum number of RAG values that each entity should have
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
    logging.info(f"Before filtering, there are {len(properties.value_by_name)} entities for {prop_id}")

    properties = _filter_entities_by_popularity(
        properties=properties,
        prop_id=prop_id,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
    )
    if len(properties.value_by_name) == 0:
        logging.warning(f"No entities left after filtering by popularity. Skipping generation.")
        return {}

    properties = _filter_entities_by_name(
        properties=properties,
        prop_id=prop_id
    )
    if len(properties.value_by_name) == 0:
        logging.warning(f"No entities left after filtering by popularity and name. Skipping generation.")
        return {}

    rag_values_map = None
    if remove_ambiguous and non_overlapping_rag_values:
        rag_eval_path = DATA_DIR / "prop_rag_eval" / "T0.0_P0.9_M1000" / f"{prop_id}.yaml"
        logging.info(f"Loading RAG evaluation from {rag_eval_path}")
        rag_eval = PropRAGEval.load(rag_eval_path)
        rag_values_map = rag_eval.values_by_entity_name
        properties = _filter_entities_by_rag_values(
            properties=properties,
            rag_values_map=rag_values_map,
            prop_id=prop_id,
            min_rag_values_count=min_rag_values_count,
        )

    if len(properties.value_by_name) == 0:
        logging.warning(f"No entities left after filtering by popularity, name, and RAG values. Skipping generation.")
        return {}

    potential_pairs = _generate_potential_pairs(
        properties=properties,
        prop_id=prop_id,
        min_fraction_value_diff=min_fraction_value_diff,
        max_fraction_value_diff=max_fraction_value_diff,
        rag_values_map=rag_values_map
    )

    # Iterative ambiguity evaluation: only evaluate as many pairs as needed to satisfy max_comparisons for each entity
    if remove_ambiguous != "no":
        qid_to_pair = {p.qid: p for p in potential_pairs}
        unevaluated_qids = set(p.qid for p in potential_pairs)
        accepted_pairs: list[PotentialQuestionPair] = []
        # Initialize comparison counts for all entities to 0
        comparisons_per_entity = {entity_name: 0 for entity_name in properties.value_by_name.keys()}
        batch_size = max(n, 100)  # Evaluate in batches of at least n or 100

        # Load ambiguity cache
        ambiguity_cache = _load_ambiguity_cache(prop_id)
        logging.info(f"Loaded {len(ambiguity_cache)} cached ambiguity evaluations for {prop_id}")

        def get_entity_priority_score(qid: str) -> int:
            """Calculate priority score for a pair based on its entities' comparison counts.
            Lower score = higher priority (entities need more comparisons)"""
            pair = qid_to_pair[qid]
            small_comparisons = comparisons_per_entity[pair.small_name]
            large_comparisons = comparisons_per_entity[pair.large_name]
            # Prioritize pairs where both entities need more comparisons
            return min(small_comparisons, large_comparisons)

        def can_add_pair(pair: PotentialQuestionPair) -> bool:
            """Check if adding this pair would exceed max_comparisons for either entity"""
            small_comparisons = comparisons_per_entity[pair.small_name]
            large_comparisons = comparisons_per_entity[pair.large_name]
            return small_comparisons < max_comparisons and large_comparisons < max_comparisons

        while unevaluated_qids:
            logging.warning(f"Starting iterative ambiguity evaluation with {len(unevaluated_qids)} unevaluated pairs.")

            # Sort unevaluated pairs by priority and take the top batch_size
            sorted_qids = sorted(unevaluated_qids, key=get_entity_priority_score)
            batch_qids = sorted_qids[:batch_size]

            # Check if any of the pairs in the batch contain entities that still need comparisons
            any_useful_pairs = False
            for qid in batch_qids:
                pair = qid_to_pair[qid]
                if can_add_pair(pair):
                    any_useful_pairs = True
                    break
            
            if not any_useful_pairs:
                logging.warning("No pairs in batch contain entities needing more comparisons. Breaking loop.")
                break

            # Filter out questions that are already in the cache
            questions_to_evaluate = []
            for qid in batch_qids:
                pair = qid_to_pair[qid]
                entity_pair = (pair.small_name, pair.large_name)
                if entity_pair in ambiguity_cache:
                    # Use cached result
                    if ambiguity_cache[entity_pair] == "CLEAR":
                        if can_add_pair(pair):
                            comparisons_per_entity[pair.small_name] += 1
                            comparisons_per_entity[pair.large_name] += 1
                            accepted_pairs.append(pair)
                    unevaluated_qids.remove(qid)
                else:
                    # Add to evaluation batch
                    questions_to_evaluate.append(pair)

            if not questions_to_evaluate:
                continue

            ambiguity_results: dict[str, FinalAmbiguityEvalResult] = asyncio.run(evaluate_questions_ambiguity_in_batch(
                questions_to_evaluate=questions_to_evaluate,
                evaluator_model_id=evaluator_model_id,
                sampling_params=evaluator_sampling_params,
                api_preferences=api_preferences,
                num_evals=num_ambiguity_evals,
                max_retries=max_ambiguity_eval_retries,
            ))

            # Process results and update cache
            for qid, result in ambiguity_results.items():
                if result is not None:
                    pair = qid_to_pair[qid]
                    entity_pair = (pair.small_name, pair.large_name)
                    # Save result to cache
                    ambiguity_cache[entity_pair] = result.final_classification
                    if result.final_classification == "CLEAR":
                        if can_add_pair(pair):
                            comparisons_per_entity[pair.small_name] += 1
                            comparisons_per_entity[pair.large_name] += 1
                            accepted_pairs.append(pair)
                unevaluated_qids.remove(qid)

            # Save cache after each batch
            _save_ambiguity_cache(prop_id, ambiguity_cache)

            # Check stopping conditions based on remove_ambiguous value
            if remove_ambiguous == "enough-comparisons":
                if all(v >= max_comparisons for v in comparisons_per_entity.values()):
                    logging.warning("All entities have enough comparisons. Stopping evaluation.")
                    break
            elif remove_ambiguous == "enough-pairs":
                if len(accepted_pairs) >= n:
                    logging.warning(f"Found enough pairs ({len(accepted_pairs)} >= {n}). Stopping evaluation.")
                    break

            # Log progress
            num_entities = len(properties.value_by_name)
            num_with_enough = sum(v >= max_comparisons for v in comparisons_per_entity.values())
            num_missing = num_entities - num_with_enough
            logging.warning(f"{num_missing} entities are still missing max_comparisons after this batch.")
            logging.warning(f"We have {len(accepted_pairs)} accepted pairs so far.")
            # Print all entities that need more comparisons
            logging.info("Entities needing more comparisons:")
            for entity, count in comparisons_per_entity.items():
                if count < max_comparisons:
                    logging.info(f"- `{entity}` needs {max_comparisons - count} more comparisons")
        non_ambiguous_pairs = accepted_pairs
    else:
        non_ambiguous_pairs = potential_pairs
    
    if len(non_ambiguous_pairs) == 0:
        logging.warning(f"No pairs generated after filtering. Skipping sampling.")
        return {}

    sampled_pairs = _sample_pairs(
        pairs=non_ambiguous_pairs,
        target_n=n
    )
    return _generate_datasets(
        sampled_pairs=sampled_pairs,
        properties=properties,
        prop_id=prop_id,
        max_comparisons=max_comparisons,
        dataset_suffix=dataset_suffix
    )
