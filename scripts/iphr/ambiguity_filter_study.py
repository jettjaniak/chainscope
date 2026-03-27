#!/usr/bin/env python3
# isort: skip_file
from collections import deque, defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import random
from functools import lru_cache
from typing import Any, Literal, cast

import click
import pandas as pd
from beartype import beartype

from chainscope.ambiguous_qs_eval import (
    FinalAmbiguityEvalResult,
    _run_ambiguity_eval_batch,
    _run_consistency_eval_batch,
)
from chainscope.api_utils.api_selector import APIPreferences
from chainscope.questions import (
    _filter_entities_by_name as filter_entities_by_name,
    _filter_entities_by_popularity as filter_entities_by_popularity,
    _generate_potential_pairs as generate_potential_pairs,
    _sample_pairs as sample_pairs,
)
from chainscope.typing import (
    DATA_DIR,
    DatasetParams,
    PotentialQuestionPair,
    Properties,
    PropRAGEval,
    RAGValue,
    SamplingParams,
    UnfaithfulnessPairsDataset,
)


STUDY_DIR = DATA_DIR / "ambiguity_filter_study"
RAG_SAMPLING = SamplingParams(temperature=0.0, top_p=0.9, max_new_tokens=1000)
DirectionLabel = Literal["clear", "ambiguous"]
FilterLabel = Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]
MIN_EXTEND_BATCH = 300
FaithfulnessLookup = dict[str, dict[str, set[str]]]


@lru_cache(maxsize=None)
def _load_faithfulness_lookup(dataset_suffix: str | None) -> FaithfulnessLookup:
    lookup: FaithfulnessLookup = {}
    faithfulness_dir = DATA_DIR / "faithfulness"
    if not faithfulness_dir.exists():
        return lookup
    suffix_token = f"_{dataset_suffix}" if dataset_suffix else None
    for model_dir in sorted(faithfulness_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for yaml_path in sorted(model_dir.glob("*.yaml")):
            stem = yaml_path.stem
            if dataset_suffix:
                assert suffix_token is not None
                if not stem.endswith(suffix_token):
                    continue
                prop_id = stem[: -len(suffix_token)]
                if not prop_id:
                    continue
            else:
                if "_" in stem:
                    continue
                prop_id = stem
            dataset = UnfaithfulnessPairsDataset.load_from_path(yaml_path)
            if dataset.dataset_suffix != dataset_suffix:
                continue
            prop_lookup = lookup.setdefault(prop_id, {})
            for qid, question in dataset.questions_by_qid.items():
                if not question.unfaithful_responses:
                    continue
                prop_lookup.setdefault(qid, set()).add(dataset.model_id)
                if question.metadata is not None and question.metadata.reversed_q_id:
                    prop_lookup.setdefault(question.metadata.reversed_q_id, set()).add(
                        dataset.model_id
                    )
    return lookup


EXCLUDED_PROPS = {"wm-nyc-place-lat", "wm-nyc-place-long"}
EXCLUDED_SUBSTRINGS = ("-popu", "-population", "dens")


def _is_supported_prop(prop_id: str) -> bool:
    if prop_id in EXCLUDED_PROPS:
        return False
    return not any(substring in prop_id for substring in EXCLUDED_SUBSTRINGS)


@dataclass
class GenerationConfig:
    new_pairs: int
    min_popularity: int | None
    max_popularity: int | None
    min_fraction_value_diff: float | None
    max_fraction_value_diff: float | None
    min_rag_values_count: int | None
    max_comparisons: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_pairs": self.new_pairs,
            "min_popularity": self.min_popularity,
            "max_popularity": self.max_popularity,
            "min_fraction_value_diff": self.min_fraction_value_diff,
            "max_fraction_value_diff": self.max_fraction_value_diff,
            "min_rag_values_count": self.min_rag_values_count,
            "max_comparisons": self.max_comparisons,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "GenerationConfig":
        return GenerationConfig(
            new_pairs=int(payload["new_pairs"]),
            min_popularity=payload.get("min_popularity"),
            max_popularity=payload.get("max_popularity"),
            min_fraction_value_diff=payload.get("min_fraction_value_diff"),
            max_fraction_value_diff=payload.get("max_fraction_value_diff"),
            min_rag_values_count=payload.get("min_rag_values_count"),
            max_comparisons=int(payload["max_comparisons"]),
        )


@dataclass
class DirectionRecord:
    qid: str
    question: str
    x_name: str
    y_name: str
    x_value: int | float
    y_value: int | float
    rag_values: dict[str, list[str]]
    filter_label: FilterLabel | None = None
    filter_analyses: list[str | None] | None = None
    human_label: DirectionLabel | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "x_name": self.x_name,
            "y_name": self.y_name,
            "x_value": self.x_value,
            "y_value": self.y_value,
            "rag_values": self.rag_values,
            "filter_label": self.filter_label,
            "filter_analyses": self.filter_analyses,
            "human_label": self.human_label,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "DirectionRecord":
        return DirectionRecord(
            qid=str(payload["qid"]),
            question=str(payload["question"]),
            x_name=str(payload["x_name"]),
            y_name=str(payload["y_name"]),
            x_value=payload["x_value"],
            y_value=payload["y_value"],
            rag_values={k: list(v) for k, v in payload.get("rag_values", {}).items()},
            filter_label=payload.get("filter_label"),
            filter_analyses=payload.get("filter_analyses"),
            human_label=payload.get("human_label"),
        )


@dataclass
class PairRecord:
    pair_id: str
    prop_id: str
    forward: DirectionRecord
    reverse: DirectionRecord
    filter_pair_label: FilterLabel | None = None
    filter_pair_analyses: list[str | None] | None = None
    human_pair_label: DirectionLabel | None = None
    source: Literal["study", "existing"] = "study"
    skip_human: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "prop_id": self.prop_id,
            "forward": self.forward.to_dict(),
            "reverse": self.reverse.to_dict(),
            "filter_pair_label": self.filter_pair_label,
            "filter_pair_analyses": self.filter_pair_analyses,
            "human_pair_label": self.human_pair_label,
            "source": self.source,
            "skip_human": self.skip_human,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "PairRecord":
        return PairRecord(
            pair_id=str(payload["pair_id"]),
            prop_id=str(payload["prop_id"]),
            forward=DirectionRecord.from_dict(payload["forward"]),
            reverse=DirectionRecord.from_dict(payload["reverse"]),
            filter_pair_label=payload.get("filter_pair_label"),
            filter_pair_analyses=payload.get("filter_pair_analyses"),
            human_pair_label=payload.get("human_pair_label"),
            source=payload.get("source", "study"),
            skip_human=payload.get("skip_human", False),
        )


@beartype
def _load_filtered_properties(
    prop_id: str, config: GenerationConfig, rag_map: dict[str, list[RAGValue]]
) -> Properties:
    properties = Properties.load(prop_id)
    logging.info(
        "Loaded %s entities for %s before filtering",
        len(properties.value_by_name),
        prop_id,
    )
    if config.min_rag_values_count is not None:
        eligible_names = {
            name
            for name, values in rag_map.items()
            if len(values) >= config.min_rag_values_count
        }
        properties.value_by_name = {
            name: value
            for name, value in properties.value_by_name.items()
            if name in eligible_names
        }
        assert properties.value_by_name, (
            f"No entities remain for {prop_id} after enforcing min_rag_values_count="
            f"{config.min_rag_values_count}"
        )
    properties = filter_entities_by_popularity(
        properties=properties,
        prop_id=prop_id,
        min_popularity=config.min_popularity,
        max_popularity=config.max_popularity,
    )
    properties = filter_entities_by_name(properties=properties, prop_id=prop_id)
    return properties


@beartype
def _convert_potential_pair(
    pair: PotentialQuestionPair, prop_id: str, rag_map: dict[str, list[RAGValue]]
) -> PairRecord:
    rag_values = {
        pair.small_name: ensure_entity_has_rag(rag_map, pair.small_name),
        pair.large_name: ensure_entity_has_rag(rag_map, pair.large_name),
    }
    forward = DirectionRecord(
        qid=pair.qid,
        question=pair.q_str,
        x_name=pair.small_name,
        y_name=pair.large_name,
        x_value=pair.small_value,
        y_value=pair.large_value,
        rag_values=rag_values_to_strings(rag_values),
    )
    reverse_qid = hashlib.sha256(pair.reversed_q_str.encode()).hexdigest()
    reverse = DirectionRecord(
        qid=reverse_qid,
        question=pair.reversed_q_str,
        x_name=pair.large_name,
        y_name=pair.small_name,
        x_value=pair.large_value,
        y_value=pair.small_value,
        rag_values=rag_values_to_strings(rag_values),
    )
    return PairRecord(
        pair_id=pair.qid,
        prop_id=prop_id,
        forward=forward,
        reverse=reverse,
        source="study",
    )


@dataclass
class StudyState:
    dataset_suffix: str
    prop_id: str
    generation: GenerationConfig
    pairs: dict[str, PairRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_suffix": self.dataset_suffix,
            "prop_id": self.prop_id,
            "generation": self.generation.to_dict(),
            "pairs": {pair_id: pair.to_dict() for pair_id, pair in self.pairs.items()},
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "StudyState":
        generation = GenerationConfig.from_dict(payload["generation"])
        pairs = {
            pair_id: PairRecord.from_dict(pair_payload)
            for pair_id, pair_payload in payload["pairs"].items()
        }
        return StudyState(
            dataset_suffix=str(payload["dataset_suffix"]),
            prop_id=str(payload["prop_id"]),
            generation=generation,
            pairs=pairs,
        )


@lru_cache(maxsize=None)
def _resolve_df_info(dataset_suffix: str | None) -> tuple[Path, str | None]:
    suffix_key = None if dataset_suffix is None else dataset_suffix.replace("-", "_")
    suffix_map: dict[str | None, tuple[str, str | None]] = {
        None: ("df-wm.pkl.gz", None),
        "non_ambiguous": ("df-wm-non-ambiguous.pkl.gz", "non-ambiguous"),
        "non_ambiguous_hard": ("df-wm-non-ambiguous-hard.pkl.gz", "non-ambiguous-hard"),
        "non_ambiguous_hard_2": (
            "df-wm-non-ambiguous-hard-2.pkl.gz",
            "non-ambiguous-hard-2",
        ),
        "non_ambiguous_hard_3": (
            "df-wm-non-ambiguous-hard-3.pkl.gz",
            "non-ambiguous-hard-3",
        ),
    }
    if suffix_key not in suffix_map:
        raise ValueError(f"Unsupported dataset suffix: {dataset_suffix}")
    filename, canonical = suffix_map[suffix_key]
    return DATA_DIR / filename, canonical


@lru_cache(maxsize=None)
def _load_dataset_dataframe(
    dataset_suffix: str | None,
) -> tuple[pd.DataFrame, str | None]:
    df_path, canonical = _resolve_df_info(dataset_suffix)
    df_loaded = pd.read_pickle(df_path)
    if not isinstance(df_loaded, pd.DataFrame):
        raise RuntimeError(f"Expected DataFrame in {df_path}, found {type(df_loaded)}")
    df: pd.DataFrame = df_loaded.copy()
    if "mode" in df.columns:
        df = df.loc[df["mode"] == "cot"].copy()
    return df, canonical


def _filter_df_by_suffix(
    df: pd.DataFrame, canonical_suffix: str | None
) -> pd.DataFrame:
    if canonical_suffix is None:
        if "dataset_suffix" in df.columns:
            return df.loc[df["dataset_suffix"].isna()].copy()
        return df.copy()
    if "dataset_suffix" not in df.columns:
        raise RuntimeError("Dataset dataframe missing dataset_suffix column.")
    return df.loc[df["dataset_suffix"] == canonical_suffix].copy()


def _collect_existing_pair_candidates(
    df: pd.DataFrame,
    canonical_suffix: str | None,
    prop_id: str,
    rag_map: dict[str, list[RAGValue]],
) -> list[PairRecord]:
    subset = df.loc[df["prop_id"] == prop_id].copy()
    subset = _filter_df_by_suffix(subset, canonical_suffix)
    if subset.empty:
        return []
    pairs: dict[tuple[str, frozenset[str]], list[dict[str, Any]]] = {}
    for row in subset.to_dict("records"):
        x_name = str(row["x_name"])
        y_name = str(row["y_name"])
        key = (str(row["comparison"]), frozenset({x_name, y_name}))
        pairs.setdefault(key, []).append(row)
    records: list[PairRecord] = []
    for rows in pairs.values():
        if len(rows) < 2:
            continue
        yes_row = next((r for r in rows if str(r["answer"]) == "YES"), None)
        no_row = next((r for r in rows if str(r["answer"]) == "NO"), None)
        if yes_row is None or no_row is None:
            continue
        pair_id = str(yes_row["qid"])
        yes_x = str(yes_row["x_name"])
        yes_y = str(yes_row["y_name"])
        no_x = str(no_row["x_name"])
        no_y = str(no_row["y_name"])
        rag_values = {
            yes_x: ensure_entity_has_rag(rag_map, yes_x),
            yes_y: ensure_entity_has_rag(rag_map, yes_y),
        }
        reverse_rag_values = {
            no_x: ensure_entity_has_rag(rag_map, no_x),
            no_y: ensure_entity_has_rag(rag_map, no_y),
        }
        forward = DirectionRecord(
            qid=pair_id,
            question=str(yes_row["q_str"]),
            x_name=yes_x,
            y_name=yes_y,
            x_value=yes_row["x_value"],
            y_value=yes_row["y_value"],
            rag_values=rag_values_to_strings(rag_values),
            filter_label="CLEAR",
        )
        reverse = DirectionRecord(
            qid=str(no_row["qid"]),
            question=str(no_row["q_str"]),
            x_name=no_x,
            y_name=no_y,
            x_value=no_row["x_value"],
            y_value=no_row["y_value"],
            rag_values=rag_values_to_strings(reverse_rag_values),
            filter_label="CLEAR",
        )
        records.append(
            PairRecord(
                pair_id=pair_id,
                prop_id=prop_id,
                forward=forward,
                reverse=reverse,
                filter_pair_label="CLEAR",
                source="existing",
            )
        )
    return records


def ensure_existing_pairs(
    state: StudyState,
    df: pd.DataFrame,
    canonical_suffix: str | None,
    rag_map: dict[str, list[RAGValue]],
    target: int,
    rng: random.Random,
) -> int:
    if target <= 0:
        return 0
    current = sum(1 for pair in state.pairs.values() if pair.source == "existing")
    if current >= target:
        return 0
    candidates = _collect_existing_pair_candidates(
        df=df,
        canonical_suffix=canonical_suffix,
        prop_id=state.prop_id,
        rag_map=rag_map,
    )
    available = [pair for pair in candidates if pair.pair_id not in state.pairs]
    if not available:
        logging.warning(
            "No existing dataset pairs available for %s with suffix %s",
            state.prop_id,
            canonical_suffix,
        )
        return 0
    rng.shuffle(available)
    needed = target - current
    added = 0
    for pair in available:
        state.pairs[pair.pair_id] = pair
        added += 1
        if added >= needed:
            break
    if added > 0:
        save_state(state)
    return added


@beartype
def state_path(dataset_suffix: str, prop_id: str) -> Path:
    return STUDY_DIR / dataset_suffix / prop_id / "state.json"


@beartype
def load_state(dataset_suffix: str, prop_id: str) -> StudyState | None:
    path = state_path(dataset_suffix, prop_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return StudyState.from_dict(payload)


@beartype
def save_state(state: StudyState) -> None:
    path = state_path(state.dataset_suffix, state.prop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _count_study_pairs(state: StudyState | None) -> int:
    if state is None:
        return 0
    return sum(1 for pair in state.pairs.values() if pair.source == "study")


def _count_existing_pairs(state: StudyState | None) -> int:
    if state is None:
        return 0
    return sum(1 for pair in state.pairs.values() if pair.source == "existing")


@beartype
def discover_prop_ids(dataset_suffix: str) -> list[str]:
    question_dir = DATA_DIR / "questions"
    assert question_dir.exists(), f"Questions directory missing at {question_dir}"
    prop_ids: set[str] = set()
    suffix_marker = f"_{dataset_suffix}"
    for dataset_file in question_dir.rglob("*.yaml"):
        dataset_id = dataset_file.stem
        if not dataset_id.endswith(suffix_marker):
            continue
        params = DatasetParams.from_id(dataset_id)
        if params.suffix != dataset_suffix:
            continue
        if _is_supported_prop(params.prop_id):
            prop_ids.add(params.prop_id)
    assert prop_ids, f"No properties found for dataset suffix {dataset_suffix}"
    return sorted(prop_ids)


@beartype
def load_rag_values(prop_id: str) -> dict[str, list[RAGValue]]:
    rag_dir = DATA_DIR / "prop_rag_eval" / RAG_SAMPLING.id
    rag_path = rag_dir / f"{prop_id}.yaml"
    assert rag_path.exists(), f"Missing RAG values at {rag_path}"
    rag_eval = PropRAGEval.load(rag_path)
    return rag_eval.values_by_entity_name


@beartype
def ensure_entity_has_rag(
    rag_map: dict[str, list[RAGValue]],
    entity_name: str,
) -> list[RAGValue]:
    rag_values = rag_map.get(entity_name, [])
    assert rag_values, f"No RAG values recorded for {entity_name}"
    return rag_values


@beartype
def rag_values_to_strings(
    rag_values: dict[str, list[RAGValue]],
) -> dict[str, list[str]]:
    return {name: [rag.value for rag in values] for name, values in rag_values.items()}


@beartype
def generate_pairs(
    prop_id: str,
    dataset_suffix: str,
    config: GenerationConfig,
    rag_map: dict[str, list[RAGValue]],
) -> StudyState:
    properties = _load_filtered_properties(prop_id, config, rag_map)
    potential_pairs = generate_potential_pairs(
        properties=properties,
        prop_id=prop_id,
        min_fraction_value_diff=config.min_fraction_value_diff,
        max_fraction_value_diff=config.max_fraction_value_diff,
        rag_values_map=rag_map,
    )
    assert potential_pairs, f"No potential pairs available for {prop_id}"
    sampled_pairs = sample_pairs(potential_pairs, config.new_pairs)
    assert sampled_pairs, f"Sampling produced no pairs for {prop_id}"
    pairs: dict[str, PairRecord] = {}
    for pair in sampled_pairs:
        record = _convert_potential_pair(pair, prop_id, rag_map)
        pairs[record.pair_id] = record
    state = StudyState(
        dataset_suffix=dataset_suffix,
        prop_id=prop_id,
        generation=config,
        pairs=pairs,
    )
    save_state(state)
    logging.info(
        "Generated %s question pairs for %s (suffix=%s)",
        len(pairs),
        prop_id,
        dataset_suffix,
    )
    return state


@beartype
def build_eval_pair(
    direction: DirectionRecord,
    counterpart_question: str,
    rag_map: dict[str, list[RAGValue]],
) -> PotentialQuestionPair:
    rag_values = {
        direction.x_name: ensure_entity_has_rag(rag_map, direction.x_name),
        direction.y_name: ensure_entity_has_rag(rag_map, direction.y_name),
    }
    return PotentialQuestionPair(
        qid=direction.qid,
        q_str=direction.question,
        reversed_q_str=counterpart_question,
        small_name=direction.x_name,
        small_value=direction.x_value,
        large_name=direction.y_name,
        large_value=direction.y_value,
        rag_values_for_q=rag_values,
    )


@beartype
def record_direction_results(
    state: StudyState,
    results: dict[str, FinalAmbiguityEvalResult],
) -> None:
    for pair in state.pairs.values():
        for direction in (pair.forward, pair.reverse):
            result = results.get(direction.qid)
            if result is None:
                continue
            direction.filter_label = result.final_classification
            direction.filter_analyses = result.ambiguity_analyses


@beartype
def record_pair_results(
    state: StudyState,
    results: dict[str, FinalAmbiguityEvalResult],
) -> None:
    for pair in state.pairs.values():
        result = results.get(pair.pair_id)
        if result is None:
            continue
        pair.filter_pair_label = result.final_classification
        pair.filter_pair_analyses = result.consistency_analyses


@beartype
def ensure_filter_labels(
    state: StudyState,
    rag_map: dict[str, list[RAGValue]],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_evals: int,
    max_retries: int,
) -> None:
    missing_directions: list[PotentialQuestionPair] = []
    for pair in state.pairs.values():
        if pair.forward.filter_label is None:
            missing_directions.append(
                build_eval_pair(pair.forward, pair.reverse.question, rag_map)
            )
        if pair.reverse.filter_label is None:
            missing_directions.append(
                build_eval_pair(pair.reverse, pair.forward.question, rag_map)
            )
    if missing_directions:
        logging.warning(
            "Evaluating ambiguity for %s question directions in %s",
            len(missing_directions),
            state.prop_id,
        )
        results = asyncio_run_ambiguity(
            questions=missing_directions,
            evaluator_model_id=evaluator_model_id,
            sampling_params=sampling_params,
            api_preferences=api_preferences,
            num_evals=num_evals,
            max_retries=max_retries,
        )
        record_direction_results(state, results)
        save_state(state)
    pending_pairs = [
        pair
        for pair in state.pairs.values()
        if pair.filter_pair_label is None
        and pair.forward.filter_label == "CLEAR"
        and pair.reverse.filter_label == "CLEAR"
    ]
    if pending_pairs:
        logging.warning(
            "Evaluating consistency for %s pairs in %s",
            len(pending_pairs),
            state.prop_id,
        )
        eval_pairs = [
            PotentialQuestionPair(
                qid=pair.pair_id,
                q_str=pair.forward.question,
                reversed_q_str=pair.reverse.question,
                small_name=pair.forward.x_name,
                small_value=pair.forward.x_value,
                large_name=pair.forward.y_name,
                large_value=pair.forward.y_value,
                rag_values_for_q={
                    pair.forward.x_name: ensure_entity_has_rag(
                        rag_map, pair.forward.x_name
                    ),
                    pair.forward.y_name: ensure_entity_has_rag(
                        rag_map, pair.forward.y_name
                    ),
                },
            )
            for pair in pending_pairs
        ]
        pair_results = asyncio_run_consistency(
            questions=eval_pairs,
            evaluator_model_id=evaluator_model_id,
            sampling_params=sampling_params,
            api_preferences=api_preferences,
            num_evals=num_evals,
            max_retries=max_retries,
        )
        record_pair_results(state, pair_results)
        save_state(state)


@beartype
def extend_state_with_pairs(
    state: StudyState,
    config: GenerationConfig,
    rag_map: dict[str, list[RAGValue]],
    additional_count: int,
) -> None:
    properties = _load_filtered_properties(state.prop_id, config, rag_map)
    potential_pairs = generate_potential_pairs(
        properties=properties,
        prop_id=state.prop_id,
        min_fraction_value_diff=config.min_fraction_value_diff,
        max_fraction_value_diff=config.max_fraction_value_diff,
        rag_values_map=rag_map,
    )
    used_ids = set(state.pairs.keys())
    added = 0
    for pair in potential_pairs:
        if pair.qid in used_ids:
            continue
        record = _convert_potential_pair(pair, state.prop_id, rag_map)
        state.pairs[record.pair_id] = record
        used_ids.add(record.pair_id)
        added += 1
        if added >= additional_count:
            break
    if added == 0:
        raise RuntimeError(
            f"Unable to find additional unique pairs for {state.prop_id}."
        )
    logging.info("Extended %s with %s new pairs", state.prop_id, added)
    save_state(state)


@beartype
def pair_llm_label(pair: PairRecord) -> FilterLabel | None:
    if pair.source != "study":
        return None
    if pair.forward.filter_label is None or pair.reverse.filter_label is None:
        return None
    if pair.forward.filter_label != "CLEAR" or pair.reverse.filter_label != "CLEAR":
        return "AMBIGUOUS"
    if pair.filter_pair_label is None:
        return None
    if pair.filter_pair_label == "CLEAR":
        return "CLEAR"
    return "AMBIGUOUS"


@beartype
def count_pairs_by_label(state: StudyState) -> dict[str, int]:
    counts = {"CLEAR": 0, "AMBIGUOUS": 0}
    for pair in state.pairs.values():
        if pair.source != "study":
            continue
        label = pair_llm_label(pair)
        if label is None:
            continue
        if label not in counts:
            counts["AMBIGUOUS"] += 1
        else:
            counts[label] += 1
    return counts


def aggregate_label_counts(states: dict[str, StudyState]) -> dict[str, int]:
    totals = {"CLEAR": 0, "AMBIGUOUS": 0}
    for state in states.values():
        local = count_pairs_by_label(state)
        totals["CLEAR"] += local["CLEAR"]
        totals["AMBIGUOUS"] += local["AMBIGUOUS"]
    return totals


def summarize_states(states: dict[str, StudyState]) -> None:
    if not states:
        click.echo("\nNo properties have been processed yet.")
        return
    click.echo("\nState before labeling:")
    totals = {
        "study": {
            "CLEAR": {"total": 0, "missing": 0},
            "AMBIGUOUS": {"total": 0, "missing": 0},
        },
        "existing": {"total": 0, "missing": 0},
    }
    for prop_id in sorted(states.keys()):
        state = states[prop_id]
        study_summary = {
            "CLEAR": {"total": 0, "missing": 0},
            "AMBIGUOUS": {"total": 0, "missing": 0},
        }
        existing_total = 0
        existing_missing = 0
        for pair in state.pairs.values():
            if pair.skip_human:
                continue
            if pair.source == "study":
                label = pair_llm_label(pair)
                if label is None:
                    continue
                study_summary[label]["total"] += 1
                if pair.human_pair_label is None:
                    study_summary[label]["missing"] += 1
            else:
                existing_total += 1
                if pair.human_pair_label is None:
                    existing_missing += 1
        clear_info = study_summary["CLEAR"]
        amb_info = study_summary["AMBIGUOUS"]
        totals["study"]["CLEAR"]["total"] += clear_info["total"]
        totals["study"]["CLEAR"]["missing"] += clear_info["missing"]
        totals["study"]["AMBIGUOUS"]["total"] += amb_info["total"]
        totals["study"]["AMBIGUOUS"]["missing"] += amb_info["missing"]
        totals["existing"]["total"] += existing_total
        totals["existing"]["missing"] += existing_missing
        click.echo(
            f"- {prop_id}: study CLEAR={clear_info['total']} (missing {clear_info['missing']}), "
            f"AMBIG={amb_info['total']} (missing {amb_info['missing']}); "
            f"existing={existing_total} (missing {existing_missing})"
        )
    clear_tot = totals["study"]["CLEAR"]
    amb_tot = totals["study"]["AMBIGUOUS"]
    existing_tot = totals["existing"]
    click.echo(
        f"TOTALS: study CLEAR={clear_tot['total']} (missing {clear_tot['missing']}), "
        f"AMBIG={amb_tot['total']} (missing {amb_tot['missing']}); "
        f"existing={existing_tot['total']} (missing {existing_tot['missing']})"
    )


def enforce_labeling_quota(
    states: dict[str, StudyState],
    target_study_clear: int,
    target_study_ambiguous: int,
    target_existing: int,
    study_per_prop_cap: int | None,
    existing_per_prop_cap: int | None,
) -> None:
    per_label_limit = (
        study_per_prop_cap // 2
        if study_per_prop_cap and study_per_prop_cap > 0
        else None
    )

    def gather_candidates() -> (
        tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]
    ):
        study_clear: dict[str, list[str]] = defaultdict(list)
        study_amb: dict[str, list[str]] = defaultdict(list)
        existing: dict[str, list[str]] = defaultdict(list)
        for prop_id in sorted(states.keys()):
            state = states[prop_id]
            for pair_id in sorted(state.pairs.keys()):
                pair = state.pairs[pair_id]
                if pair.human_pair_label is not None:
                    continue
                if pair.source == "study":
                    label = pair_llm_label(pair)
                    if label == "CLEAR":
                        study_clear[state.prop_id].append(pair_id)
                    elif label == "AMBIGUOUS":
                        study_amb[state.prop_id].append(pair_id)
                elif pair.source == "existing":
                    existing[state.prop_id].append(pair_id)
        return study_clear, study_amb, existing

    def allocate(
        by_prop: dict[str, list[str]], base_limit: int | None, target: int
    ) -> set[str]:
        if target <= 0:
            return set()
        contributions: dict[str, int] = {}
        total = 0
        for prop in sorted(by_prop.keys()):
            ids = by_prop[prop]
            prop_cap = len(ids) if base_limit is None else min(base_limit, len(ids))
            contributions[prop] = prop_cap
            total += prop_cap
        # Reduce if we exceeded target
        if total > target:
            props_list = [prop for prop, count in contributions.items() if count > 0]
            while total > target and props_list:
                for prop in list(props_list):
                    if total <= target:
                        break
                    if contributions[prop] > 0:
                        contributions[prop] -= 1
                        total -= 1
                        if contributions[prop] == 0:
                            props_list.remove(prop)
                    else:
                        props_list.remove(prop)
        # Increase if below target and capacity available
        if total < target:
            capacity = {
                prop: len(by_prop[prop]) - contributions.get(prop, 0)
                for prop in by_prop
            }
            props_list = [prop for prop, cap in capacity.items() if cap > 0]
            while total < target and props_list:
                progress = False
                for prop in list(props_list):
                    if total >= target:
                        break
                    available = capacity.get(prop, 0)
                    if available <= 0:
                        props_list.remove(prop)
                        continue
                    contributions[prop] += 1
                    capacity[prop] = available - 1
                    total += 1
                    progress = True
                    if capacity[prop] <= 0:
                        props_list.remove(prop)
                    if total >= target:
                        break
                if not progress:
                    break

        selected: set[str] = set()
        for prop in sorted(by_prop.keys()):
            ids = sorted(by_prop[prop])
            need = contributions.get(prop, 0)
            if need <= 0:
                continue
            selected.update(ids[:need])
        return selected

    study_clear_by_prop, study_amb_by_prop, existing_by_prop = gather_candidates()
    keep_clear = allocate(study_clear_by_prop, per_label_limit, target_study_clear)
    keep_amb = allocate(study_amb_by_prop, per_label_limit, target_study_ambiguous)
    keep_existing = allocate(
        existing_by_prop,
        existing_per_prop_cap
        if existing_per_prop_cap and existing_per_prop_cap > 0
        else None,
        target_existing,
    )

    for state in states.values():
        changed = False
        for pair_id, pair in state.pairs.items():
            if pair.human_pair_label is not None:
                continue
            original = pair.skip_human
            if pair.source == "study":
                label = pair_llm_label(pair)
                if label == "CLEAR":
                    pair.skip_human = pair_id not in keep_clear
                elif label == "AMBIGUOUS":
                    pair.skip_human = pair_id not in keep_amb
                else:
                    pair.skip_human = True
            elif pair.source == "existing":
                pair.skip_human = pair_id not in keep_existing
            if pair.skip_human != original:
                changed = True
        if changed:
            save_state(state)


@beartype
def prune_to_balanced_pairs(state: StudyState, target: int) -> None:
    assert target >= 0, "target must be non-negative"
    clear_ids: list[str] = []
    ambiguous_ids: list[str] = []
    for pair_id, pair in state.pairs.items():
        if pair.source != "study":
            continue
        label = pair_llm_label(pair)
        if label == "CLEAR":
            clear_ids.append(pair_id)
        elif label == "AMBIGUOUS":
            ambiguous_ids.append(pair_id)
    target = min(target, len(clear_ids), len(ambiguous_ids))
    if target == 0:
        state.pairs = {
            pair_id: pair
            for pair_id, pair in state.pairs.items()
            if pair.source == "existing"
        }
        save_state(state)
        return
    selected_clear = set(clear_ids[:target])
    selected_amb = set(ambiguous_ids[:target])
    keep_ids = selected_clear | selected_amb
    existing_pairs = {
        pair_id: pair
        for pair_id, pair in state.pairs.items()
        if pair.source == "existing"
    }
    study_pairs = {
        pair_id: pair
        for pair_id, pair in state.pairs.items()
        if pair.source == "study" and pair_id in keep_ids
    }
    existing_pairs.update(study_pairs)
    state.pairs = existing_pairs
    save_state(state)


@beartype
def _remove_failed_pairs(state: StudyState) -> int:
    to_remove: list[str] = []
    for pair_id, pair in state.pairs.items():
        if pair.source != "study":
            continue
        direction_failed = any(
            direction.filter_label == "FAILED_EVAL"
            for direction in (pair.forward, pair.reverse)
        )
        pair_failed = pair.filter_pair_label == "FAILED_EVAL"
        if direction_failed or pair_failed:
            to_remove.append(pair_id)
    for pair_id in to_remove:
        del state.pairs[pair_id]
    if to_remove:
        logging.warning(
            "Removed %s FAILED_EVAL pairs for %s", len(to_remove), state.prop_id
        )
        save_state(state)
    return len(to_remove)


@beartype
def grow_state_with_budget(
    state: StudyState,
    config: GenerationConfig,
    rag_map: dict[str, list[RAGValue]],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_evals: int,
    max_retries: int,
    max_pairs: int | None = None,
) -> int:
    """Generate up to config.new_pairs additional study pairs, respecting caps."""
    budget = max(0, config.new_pairs)
    state.generation = config

    ensure_filter_labels(
        state=state,
        rag_map=rag_map,
        evaluator_model_id=evaluator_model_id,
        sampling_params=sampling_params,
        api_preferences=api_preferences,
        num_evals=num_evals,
        max_retries=max_retries,
    )
    _remove_failed_pairs(state)
    if budget == 0:
        return 0

    total_added = 0
    while total_added < budget:
        total_pairs = _count_study_pairs(state)
        if max_pairs is not None:
            remaining_capacity = max(0, max_pairs - total_pairs)
            if remaining_capacity <= 0:
                logging.warning(
                    "Per-property cap reached for %s (%s pairs, cap=%s); skipping generation.",
                    state.prop_id,
                    total_pairs,
                    max_pairs,
                )
                break
        remaining_budget = budget - total_added
        batch_size = min(remaining_budget, MIN_EXTEND_BATCH)
        if max_pairs is not None:
            batch_size = min(batch_size, max(0, max_pairs - total_pairs))
        if batch_size <= 0:
            break
        try:
            extend_state_with_pairs(
                state=state,
                config=config,
                rag_map=rag_map,
                additional_count=batch_size,
            )
        except RuntimeError as error:
            logging.warning(
                "Unable to extend %s with additional study pairs: %s",
                state.prop_id,
                error,
            )
            break
        new_total = _count_study_pairs(state)
        delta = max(0, new_total - total_pairs)
        if delta == 0:
            break
        total_added += delta

    ensure_filter_labels(
        state=state,
        rag_map=rag_map,
        evaluator_model_id=evaluator_model_id,
        sampling_params=sampling_params,
        api_preferences=api_preferences,
        num_evals=num_evals,
        max_retries=max_retries,
    )
    _remove_failed_pairs(state)
    return total_added


@beartype
def asyncio_run_ambiguity(
    questions: list[PotentialQuestionPair],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_evals: int,
    max_retries: int,
) -> dict[str, FinalAmbiguityEvalResult]:
    import asyncio

    return asyncio.run(
        _run_ambiguity_eval_batch(
            questions_to_evaluate=questions,
            evaluator_model_id=evaluator_model_id,
            sampling_params=sampling_params,
            api_preferences=api_preferences,
            num_evals=num_evals,
            max_retries=max_retries,
        )
    )


@beartype
def asyncio_run_consistency(
    questions: list[PotentialQuestionPair],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
    api_preferences: APIPreferences,
    num_evals: int,
    max_retries: int,
) -> dict[str, FinalAmbiguityEvalResult]:
    import asyncio

    return asyncio.run(
        _run_consistency_eval_batch(
            questions_to_evaluate=questions,
            evaluator_model_id=evaluator_model_id,
            sampling_params=sampling_params,
            api_preferences=api_preferences,
            num_evals=num_evals,
            max_retries=max_retries,
        )
    )


SourceFilter = Literal["all", "existing", "study"]


@beartype
def build_question_tasks(
    states: dict[str, StudyState],
    source_filter: SourceFilter = "all",
) -> list[
    tuple[StudyState, PairRecord, DirectionRecord, Literal["forward", "reverse"]]
]:
    tasks: list[
        tuple[StudyState, PairRecord, DirectionRecord, Literal["forward", "reverse"]]
    ] = []
    for state in states.values():
        for pair in state.pairs.values():
            if pair.skip_human:
                continue
            if source_filter != "all" and pair.source != source_filter:
                continue
            if pair.forward.human_label is None:
                tasks.append((state, pair, pair.forward, "forward"))
            if pair.reverse.human_label is None:
                tasks.append((state, pair, pair.reverse, "reverse"))
    return tasks


@beartype
def build_pair_tasks(
    states: dict[str, StudyState],
    source_filter: SourceFilter = "all",
) -> list[tuple[StudyState, PairRecord]]:
    tasks: list[tuple[StudyState, PairRecord]] = []
    for state in states.values():
        for pair in state.pairs.values():
            if pair.skip_human:
                continue
            if source_filter != "all" and pair.source != source_filter:
                continue
            if pair.human_pair_label is not None:
                continue
            if (
                pair.forward.human_label == "clear"
                and pair.reverse.human_label == "clear"
            ):
                tasks.append((state, pair))
    return tasks


@beartype
def ask_direction_label(
    _state: StudyState,
    direction: DirectionRecord,
    _direction_name: str,
    question_index: int,
    total_questions: int,
) -> DirectionLabel | None:
    click.echo("\n" + "=" * 80)
    click.echo(f"Question {question_index}/{total_questions}")
    click.echo(f"Question:\n{direction.question.strip()}")
    click.echo("\nRAG values:")
    for entity in (direction.x_name, direction.y_name):
        click.echo(f"- {entity}:")
        for value in direction.rag_values.get(entity, []):
            click.echo(f"    • {value}")
    click.echo("\nLabel options: [c] clear, [a] ambiguous, [s] skip, [q] quit")
    while True:
        choice = click.getchar().strip().lower()
        if choice:
            click.echo(choice)
        if choice == "q":
            return None
        if choice == "s":
            return direction.human_label
        if choice in {"c", "a"}:
            return cast(DirectionLabel, {"c": "clear", "a": "ambiguous"}[choice])
        click.echo("Invalid choice, please use c/a/s/q.")


@beartype
def ask_pair_label(
    _state: StudyState,
    pair: PairRecord,
    pair_index: int,
    total_pairs: int,
) -> DirectionLabel | None:
    click.echo("\n" + "-" * 80)
    click.echo(f"Pair {pair_index}/{total_pairs}")
    click.echo("\nQuestion A:")
    click.echo(pair.forward.question.strip())
    click.echo("\nQuestion B:")
    click.echo(pair.reverse.question.strip())
    click.echo("\nRAG values:")
    entities = {
        pair.forward.x_name: pair.forward.rag_values.get(pair.forward.x_name, []),
        pair.forward.y_name: pair.forward.rag_values.get(pair.forward.y_name, []),
    }
    for entity, values in entities.items():
        click.echo(f"- {entity}:")
        for value in values:
            click.echo(f"    • {value}")
    click.echo("\nPair label options: [c] clear, [a] ambiguous, [s] skip, [q] quit")
    while True:
        choice = click.getchar().strip().lower()
        if choice:
            click.echo(choice)
        if choice == "q":
            return None
        if choice == "s":
            return pair.human_pair_label
        if choice in {"c", "a"}:
            return cast(DirectionLabel, {"c": "clear", "a": "ambiguous"}[choice])
        click.echo("Invalid choice, please use c/a/s/q.")


@beartype
def run_labeling_loops(
    states: dict[str, StudyState],
    rng: random.Random,
    source_filter: SourceFilter = "all",
) -> None:
    question_tasks = build_question_tasks(states, source_filter=source_filter)
    rng.shuffle(question_tasks)
    total_questions = len(question_tasks)
    task_queue: deque[
        tuple[StudyState, PairRecord, DirectionRecord, Literal["forward", "reverse"]]
    ] = deque(question_tasks)
    if total_questions == 0:
        click.echo("\nNo pending question-level labels.")
    while task_queue:
        state, pair, direction, direction_name = task_queue.popleft()
        question_number = total_questions - len(task_queue)
        label = ask_direction_label(
            state, direction, direction_name, question_number, total_questions
        )
        if label is None:
            click.echo("Exiting labeling loop.")
            return
        if label == direction.human_label:
            continue
        direction.human_label = label
        save_state(state)
    pair_tasks = build_pair_tasks(states, source_filter=source_filter)
    rng.shuffle(pair_tasks)
    pair_queue: deque[tuple[StudyState, PairRecord]] = deque(pair_tasks)
    total_pairs = len(pair_tasks)
    while pair_queue:
        pair_index = total_pairs - len(pair_queue)
        state, pair = pair_queue.popleft()
        label = ask_pair_label(state, pair, pair_index, total_pairs)
        if label is None:
            click.echo("Exiting pair loop.")
            return
        if label == pair.human_pair_label:
            continue
        pair.human_pair_label = label
        save_state(state)


@beartype
def is_human_positive(label: DirectionLabel | None) -> bool:
    return label == "ambiguous"


@beartype
def is_filter_positive(label: FilterLabel | None) -> bool:
    return label == "AMBIGUOUS"


@beartype
def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    assert 0 <= successes <= total
    assert total > 0
    z = 1.959963984540054
    phat = successes / total
    denominator = 1.0 + (z**2) / total
    centre = phat + (z**2) / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * total)) / total)
    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)
    return lower, upper


@beartype
def compute_metrics(
    states: dict[str, StudyState],
    residual_examples: bool,
    faithfulness_lookup: FaithfulnessLookup | None = None,
) -> None:
    study_question = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    study_pair = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    residual_stats = {
        "study": {"total": 0, "ambiguous": 0},
        "existing": {"total": 0, "ambiguous": 0},
        "combined": {"total": 0, "ambiguous": 0},
    }
    ambiguous_examples: list[tuple[str, PairRecord]] = []

    def _faithfulness_models(pair: PairRecord) -> set[str]:
        if not faithfulness_lookup:
            return set()
        prop_lookup = faithfulness_lookup.get(pair.prop_id)
        if not prop_lookup:
            return set()
        models: set[str] = set()
        for qid in (pair.forward.qid, pair.reverse.qid):
            model_ids = prop_lookup.get(qid)
            if model_ids:
                models.update(model_ids)
        return models

    def inferred_pair_human_label(pair: PairRecord) -> DirectionLabel | None:
        if pair.human_pair_label is not None:
            return pair.human_pair_label
        if pair.forward.human_label is None or pair.reverse.human_label is None:
            return None
        if (
            pair.forward.human_label == "ambiguous"
            or pair.reverse.human_label == "ambiguous"
        ):
            return "ambiguous"
        if pair.forward.human_label == "clear" and pair.reverse.human_label == "clear":
            return "clear"
        return None

    for state in states.values():
        for pair in state.pairs.values():
            if pair.skip_human:
                continue
            for direction in (pair.forward, pair.reverse):
                if direction.filter_label is None or direction.human_label is None:
                    continue
                filter_pos = is_filter_positive(direction.filter_label)
                human_pos = is_human_positive(direction.human_label)
                if pair.source == "study":
                    if filter_pos and human_pos:
                        study_question["tp"] += 1
                    elif filter_pos and not human_pos:
                        study_question["fp"] += 1
                    elif not filter_pos and human_pos:
                        study_question["fn"] += 1
                    else:
                        study_question["tn"] += 1
            if pair.filter_pair_label is None or pair.human_pair_label is None:
                pair_human_label = inferred_pair_human_label(pair)
                if pair_human_label is None:
                    continue
            else:
                pair_human_label = pair.human_pair_label
            filter_pos = is_filter_positive(pair.filter_pair_label)
            human_pair_pos = is_human_positive(pair_human_label)
            direction_ambiguous = (
                pair.forward.human_label == "ambiguous"
                or pair.reverse.human_label == "ambiguous"
            )
            adjusted_human_pos = human_pair_pos or direction_ambiguous
            if pair.filter_pair_label == "CLEAR":
                residual_stats[pair.source]["total"] += 1
                if adjusted_human_pos:
                    residual_stats[pair.source]["ambiguous"] += 1
                    if residual_examples and pair.source == "existing":
                        ambiguous_examples.append((state.prop_id, pair))
                residual_stats["combined"]["total"] += 1
                if adjusted_human_pos:
                    residual_stats["combined"]["ambiguous"] += 1
            if pair.source != "study":
                continue
            if filter_pos and adjusted_human_pos:
                study_pair["tp"] += 1
            elif filter_pos and not adjusted_human_pos:
                study_pair["fp"] += 1
            elif not filter_pos and adjusted_human_pos:
                study_pair["fn"] += 1
            else:
                study_pair["tn"] += 1

    def _print_binary_metrics(title: str, counts: dict[str, int]) -> None:
        total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
        click.echo(f"\n{title}")
        if total == 0:
            click.echo("No annotated items yet.")
            return
        precision = (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if (counts["tp"] + counts["fp"]) > 0
            else 0.0
        )
        recall = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if (counts["tp"] + counts["fn"]) > 0
            else 0.0
        )
        click.echo(
            f"Precision={precision:.3f} Recall={recall:.3f} "
            f"(TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}, TN={counts['tn']})"
        )

    _print_binary_metrics("Study question-level metrics", study_question)
    _print_binary_metrics("Study pair-level metrics", study_pair)

    def _print_residual(label: str, stats: dict[str, int]) -> None:
        total = stats["total"]
        click.echo(f"\nResidual ambiguity ({label} source)")
        if total == 0:
            click.echo("No clear pairs annotated yet.")
            return
        rate = stats["ambiguous"] / total
        low, high = wilson_interval(stats["ambiguous"], total)
        click.echo(f"{rate:.3f} (95% CI {low:.3f}-{high:.3f}, n={total})")

    _print_residual("existing", residual_stats["existing"])
    _print_residual("study", residual_stats["study"])
    _print_residual("combined", residual_stats["combined"])
    if residual_examples and ambiguous_examples:

        def _print_rag_values(direction: DirectionRecord) -> None:
            click.echo("RAG values:")
            for entity_name in sorted(direction.rag_values.keys()):
                click.echo(f"- {entity_name}:")
                for value in direction.rag_values[entity_name]:
                    click.echo(f"    • {value}")

        click.echo("\nResidual ambiguity examples (IPHR dataset):")
        for prop_id, pair in ambiguous_examples:
            click.echo("\n" + "-" * 80)
            click.echo(f"Property: {prop_id}")
            click.echo("Question A:")
            click.echo(pair.forward.question.strip())
            _print_rag_values(pair.forward)
            click.echo("\nQuestion B:")
            click.echo(pair.reverse.question.strip())
            _print_rag_values(pair.reverse)
            models = _faithfulness_models(pair)
            if models:
                click.echo(
                    "\nModels with unfaithful responses: " + ", ".join(sorted(models))
                )
            else:
                click.echo("\nModels with unfaithful responses: none recorded.")


@click.command()
@click.option(
    "--dataset-suffix", type=str, default="non-ambiguous-hard-2", show_default=True
)
@click.option(
    "--prop-id", "prop_ids", multiple=True, help="Restrict to specific property IDs."
)
@click.option(
    "--new-pairs",
    type=int,
    default=200,
    show_default=True,
    help="Number of freshly generated study pairs per property.",
)
@click.option("--min-popularity", type=int, default=None)
@click.option("--max-popularity", type=int, default=5, show_default=True)
@click.option("--min-fraction-value-diff", type=float, default=0.05, show_default=True)
@click.option("--max-fraction-value-diff", type=float, default=0.25, show_default=True)
@click.option("--min-rag-values-count", type=int, default=2, show_default=True)
@click.option("--max-comparisons", type=int, default=1, show_default=True)
@click.option(
    "--evaluator-model-id", type=str, default="chatgpt-4o-latest", show_default=True
)
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--top-p", type=float, default=0.9, show_default=True)
@click.option("--max-new-tokens", type=int, default=1500, show_default=True)
@click.option("--num-evals", type=int, default=3, show_default=True)
@click.option("--max-retries", type=int, default=2, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--stats-only", is_flag=True, help="Skip interactive labeling.")
@click.option(
    "--label-only",
    is_flag=True,
    help="Skip generation/evaluation; load existing state files and go straight to labeling.",
)
@click.option(
    "--residual-ambiguity-examples",
    is_flag=True,
    help="Print ambiguous residual examples from existing pairs.",
)
@click.option(
    "--existing-pairs",
    type=int,
    default=200,
    show_default=True,
    help="Total number of existing dataset pairs to mix into the study.",
)
@click.option("--use-openai/--no-openai", default=True, show_default=True)
@click.option("--use-open-router/--no-open-router", default=False, show_default=True)
@click.option("--use-anthropic/--no-anthropic", default=False, show_default=True)
@click.option("--use-deepseek/--no-deepseek", default=False, show_default=True)
@click.option(
    "--source",
    type=click.Choice(["all", "existing", "study"]),
    default="all",
    show_default=True,
    help="Only label pairs from this source.",
)
@click.option("--verbose", is_flag=True)
def main(
    dataset_suffix: str,
    prop_ids: tuple[str, ...],
    new_pairs: int,
    min_popularity: int | None,
    max_popularity: int | None,
    min_fraction_value_diff: float | None,
    max_fraction_value_diff: float | None,
    min_rag_values_count: int | None,
    max_comparisons: int,
    evaluator_model_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_evals: int,
    max_retries: int,
    seed: int | None,
    stats_only: bool,
    label_only: bool,
    residual_ambiguity_examples: bool,
    existing_pairs: int,
    use_openai: bool,
    use_open_router: bool,
    use_anthropic: bool,
    use_deepseek: bool,
    source: SourceFilter,
    verbose: bool,
) -> None:
    """Run the ambiguity filter ablation study pipeline."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    rng = random.Random(seed)

    if label_only:
        # Skip generation/evaluation/dataframe loading; just load existing
        # state files and go straight to labeling + metrics.
        study_suffix_dir = STUDY_DIR / dataset_suffix
        assert (
            study_suffix_dir.exists()
        ), f"No state directory at {study_suffix_dir}"
        found_props = sorted(
            d.name
            for d in study_suffix_dir.iterdir()
            if d.is_dir() and (d / "state.json").exists()
        )
        if prop_ids:
            found_props = [p for p in found_props if p in prop_ids]
        assert found_props, f"No state files found under {study_suffix_dir}"
        states: dict[str, StudyState] = {}
        for prop_id in found_props:
            loaded = load_state(dataset_suffix, prop_id)
            if loaded is not None:
                states[prop_id] = loaded
        faithfulness_lookup = (
            _load_faithfulness_lookup(dataset_suffix)
            if residual_ambiguity_examples
            else None
        )
        read_only_mode = stats_only or residual_ambiguity_examples
        if not read_only_mode:
            summarize_states(states)
            run_labeling_loops(states, rng, source_filter=source)
        else:
            reason = "--stats-only" if stats_only else "--residual-ambiguity-examples"
            click.echo(f"\nSkipping labeling because {reason} is set.")
        compute_metrics(
            states,
            residual_examples=residual_ambiguity_examples,
            faithfulness_lookup=faithfulness_lookup,
        )
        return

    assert (
        new_pairs % 2 == 0
    ), f"--new-pairs must be even to keep CLEAR/AMBIGUOUS balanced, got {new_pairs}"
    available_props = discover_prop_ids(dataset_suffix)
    if prop_ids:
        filtered = [prop for prop in prop_ids if _is_supported_prop(prop)]
        missing = sorted(set(filtered) - set(available_props))
        assert not missing, f"Unknown properties for suffix {dataset_suffix}: {missing}"
        target_props = [prop for prop in available_props if prop in filtered]
    else:
        target_props = available_props
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    api_preferences = APIPreferences(
        open_router=use_open_router,
        open_ai=use_openai,
        anthropic=use_anthropic,
        deepseek=use_deepseek,
    )
    assert (
        api_preferences.selects_at_least_one_api()
    ), "At least one API provider must be enabled."
    df, canonical_suffix = _load_dataset_dataframe(dataset_suffix)
    remaining_new_pairs = new_pairs
    remaining_existing = existing_pairs
    states: dict[str, StudyState] = {}
    rag_maps: dict[str, dict[str, list[RAGValue]]] = {}
    props_order = list(target_props)
    rng.shuffle(props_order)
    per_prop_cap = max(0, new_pairs // max(1, len(props_order)))
    per_prop_cap = per_prop_cap if per_prop_cap % 2 == 0 else per_prop_cap + 1
    per_prop_existing_cap = (
        math.ceil(existing_pairs / max(1, len(props_order)))
        if existing_pairs > 0
        else 0
    )

    def make_generation_config(pair_target: int) -> GenerationConfig:
        return GenerationConfig(
            new_pairs=pair_target,
            min_popularity=min_popularity,
            max_popularity=max_popularity,
            min_fraction_value_diff=min_fraction_value_diff,
            max_fraction_value_diff=max_fraction_value_diff,
            min_rag_values_count=min_rag_values_count,
            max_comparisons=max_comparisons,
        )

    def ensure_state(prop_id: str) -> StudyState:
        state = states.get(prop_id)
        if state is None:
            loaded = load_state(dataset_suffix, prop_id)
            if loaded is None:
                loaded = StudyState(
                    dataset_suffix=dataset_suffix,
                    prop_id=prop_id,
                    generation=make_generation_config(0),
                    pairs={},
                )
                save_state(loaded)
            states[prop_id] = loaded
            state = loaded
        return state

    def ensure_rag_map(prop_id: str) -> dict[str, list[RAGValue]]:
        rag_map = rag_maps.get(prop_id)
        if rag_map is None:
            rag_map = load_rag_values(prop_id)
            rag_maps[prop_id] = rag_map
        return rag_map

    # Label existing saved states so global counts are accurate.
    for prop_id in props_order:
        loaded = load_state(dataset_suffix, prop_id)
        if loaded is None:
            continue
        states[prop_id] = loaded
        rag_map = ensure_rag_map(prop_id)
        grow_state_with_budget(
            state=loaded,
            config=make_generation_config(0),
            rag_map=rag_map,
            evaluator_model_id=evaluator_model_id,
            sampling_params=sampling_params,
            api_preferences=api_preferences,
            num_evals=num_evals,
            max_retries=max_retries,
            max_pairs=per_prop_cap if per_prop_cap > 0 else 0,
        )

    for prop_id in props_order:
        if remaining_existing <= 0:
            break
        state = ensure_state(prop_id)
        rag_map = ensure_rag_map(prop_id)
        current_existing = _count_existing_pairs(state)
        target_existing = min(
            current_existing + remaining_existing, per_prop_existing_cap
        )
        if target_existing <= current_existing:
            continue
        added_existing = ensure_existing_pairs(
            state=state,
            df=df,
            canonical_suffix=canonical_suffix,
            rag_map=rag_map,
            target=target_existing,
            rng=rng,
        )
        remaining_existing = max(0, remaining_existing - added_existing)
        if added_existing > 0:
            grow_state_with_budget(
                state=state,
                config=make_generation_config(0),
                rag_map=rag_map,
                evaluator_model_id=evaluator_model_id,
                sampling_params=sampling_params,
                api_preferences=api_preferences,
                num_evals=num_evals,
                max_retries=max_retries,
                max_pairs=per_prop_cap if per_prop_cap > 0 else 0,
            )

    global_target = new_pairs // 2
    global_counts = aggregate_label_counts(states)

    def needs_balance() -> bool:
        return (
            global_counts["CLEAR"] < global_target
            or global_counts["AMBIGUOUS"] < global_target
        )

    while remaining_new_pairs > 0 and needs_balance():
        made_progress = False
        for prop_id in props_order:
            if remaining_new_pairs <= 0 or not needs_balance():
                break
            state = ensure_state(prop_id)
            rag_map = ensure_rag_map(prop_id)
            cap_limit = per_prop_cap if per_prop_cap > 0 else 0
            cap_arg = cap_limit if cap_limit > 0 else None
            current_pairs = _count_study_pairs(state)
            remaining_capacity = (
                None if cap_arg is None else max(0, cap_limit - current_pairs)
            )
            total_labeled = global_counts["CLEAR"] + global_counts["AMBIGUOUS"]
            missing_pairs = max(0, 2 * global_target - total_labeled)
            if missing_pairs == 0:
                break
            budget = min(missing_pairs, remaining_new_pairs)
            if remaining_capacity is not None:
                budget = min(budget, remaining_capacity)
            if budget <= 0:
                continue
            if budget % 2 != 0:
                budget -= 1
            if budget <= 0:
                continue
            config = make_generation_config(budget)
            added = grow_state_with_budget(
                state=state,
                config=config,
                rag_map=rag_map,
                evaluator_model_id=evaluator_model_id,
                sampling_params=sampling_params,
                api_preferences=api_preferences,
                num_evals=num_evals,
                max_retries=max_retries,
                max_pairs=cap_arg,
            )
            if added > 0:
                remaining_new_pairs = max(
                    0, remaining_new_pairs - min(added, remaining_new_pairs)
                )
                global_counts = aggregate_label_counts(states)
                made_progress = True
                if not needs_balance():
                    break
        if not made_progress:
            logging.warning(
                "Unable to achieve global 50/50 distribution; exhausted property capacity."
            )
            break

    if remaining_new_pairs > 0 and needs_balance():
        logging.warning(
            "Stopped with %s CLEAR and %s AMBIGUOUS study pairs (target per label=%s).",
            global_counts["CLEAR"],
            global_counts["AMBIGUOUS"],
            global_target,
        )
    total_existing = sum(_count_existing_pairs(state) for state in states.values())
    if total_existing < existing_pairs:
        missing_existing = existing_pairs - total_existing
        logging.warning(
            "Unable to mix in %s existing pairs; %s short after exhausting properties.",
            existing_pairs,
            missing_existing,
        )
    study_target = min(new_pairs // 2, 200)
    existing_target = min(existing_pairs, 200)
    faithfulness_lookup = (
        _load_faithfulness_lookup(dataset_suffix)
        if residual_ambiguity_examples
        else None
    )
    read_only_mode = stats_only or residual_ambiguity_examples
    if not read_only_mode:
        enforce_labeling_quota(
            states=states,
            target_study_clear=study_target,
            target_study_ambiguous=study_target,
            target_existing=existing_target,
            study_per_prop_cap=per_prop_cap if per_prop_cap > 0 else None,
            existing_per_prop_cap=per_prop_existing_cap
            if per_prop_existing_cap > 0
            else None,
        )
        summarize_states(states)
        run_labeling_loops(states, rng, source_filter=source)
    else:
        reason = "--stats-only" if stats_only else "--residual-ambiguity-examples"
        click.echo(f"\nSkipping quota adjustments because {reason} is set.")
    compute_metrics(
        states,
        residual_examples=residual_ambiguity_examples,
        faithfulness_lookup=faithfulness_lookup,
    )


if __name__ == "__main__":
    main()  # type: ignore[misc]  # pylint: disable=no-value-for-parameter
