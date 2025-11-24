#!/usr/bin/env python3

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from beartype import beartype
from tqdm import tqdm

from chainscope import DATA_DIR
from chainscope.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

Label = Literal["e", "n", "y", "r", "u"]
NormalizedLabel = Literal["e", "n", "y", "ru"]
ResponseKey = tuple[Path, str, str]

COT_EVAL_DIR = DATA_DIR / "cot_eval"
MANUAL_DIR = DATA_DIR / "manual_cot_eval"


@dataclass(frozen=True)
class ManualAnnotation:
    qid: str
    response_id: str
    label: Label


@dataclass(frozen=True)
class ManualFileMetadata:
    instr_id: str
    sampling_id: str
    comparison: Literal["gt", "lt"]
    answer: Literal["YES", "NO"]
    max_comparisons: int
    prop_id: str
    suffix: str | None
    model_id: str


@dataclass
class ManualFileData:
    metadata: ManualFileMetadata
    annotations: dict[tuple[str, str], ManualAnnotation]


@dataclass(frozen=True)
class SelectedResponse:
    eval_path: Path
    manual_path: Path
    qid: str
    response_id: str
    question: str
    response: str
    metadata: ManualFileMetadata
    auto_label: Label


@beartype
def _normalized_label(label: Label) -> NormalizedLabel:
    if label in {"r", "u"}:
        return "ru"
    return cast(NormalizedLabel, label)


@beartype
def manual_path_for_eval(eval_path: Path) -> Path:
    relative = eval_path.relative_to(COT_EVAL_DIR)
    return MANUAL_DIR / relative.with_suffix(".json")


@beartype
def auto_label_from_result(result: CotEvalResult) -> Label:
    if result.final_answer == "YES":
        return "y"
    if result.final_answer == "NO":
        if result.equal_values == "TRUE":
            return "e"
        if result.equal_values == "FALSE":
            return "n"
        raise ValueError(f"Unexpected equal_values: {result.equal_values}")
    if result.final_answer == "REFUSED":
        return "r"
    if result.final_answer == "UNKNOWN":
        return "u"
    raise ValueError(f"Unsupported final_answer: {result.final_answer}")


@beartype
def load_manual_files() -> dict[Path, ManualFileData]:
    manual_data: dict[Path, ManualFileData] = {}
    if not MANUAL_DIR.exists():
        return manual_data
    for json_path in MANUAL_DIR.rglob("*.json"):
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata_dict = payload["metadata"]
        metadata = ManualFileMetadata(
            instr_id=str(metadata_dict["instr_id"]),
            sampling_id=str(metadata_dict["sampling_id"]),
            comparison=cast(Literal["gt", "lt"], metadata_dict["comparison"]),
            answer=cast(Literal["YES", "NO"], metadata_dict["answer"]),
            max_comparisons=int(metadata_dict["max_comparisons"]),
            prop_id=str(metadata_dict["prop_id"]),
            suffix=metadata_dict.get("suffix"),
            model_id=str(metadata_dict["model_id"]),
        )
        annotations_dict: dict[tuple[str, str], ManualAnnotation] = {}
        for entry in payload.get("annotations", []):
            label = str(entry["label"])
            if label not in {"e", "n", "y", "r", "u"}:
                raise ValueError(f"Unsupported label in {json_path}: {label}")
            qid = str(entry["qid"])
            response_id = str(entry["response_id"])
            annotations_dict[(qid, response_id)] = ManualAnnotation(
                qid=qid,
                response_id=response_id,
                label=cast(Label, label),
            )
        eval_path = COT_EVAL_DIR / json_path.relative_to(MANUAL_DIR)
        eval_path = eval_path.with_suffix(".yaml")
        if not eval_path.exists():
            raise FileNotFoundError(
                f"Manual annotations reference missing evaluation file: {eval_path}"
            )
        manual_data[eval_path] = ManualFileData(
            metadata=metadata, annotations=annotations_dict
        )
    return manual_data


@beartype
def save_manual_file(manual_path: Path, data: ManualFileData) -> None:
    manual_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_payload = [
        {
            "qid": annotation.qid,
            "response_id": annotation.response_id,
            "label": annotation.label,
        }
        for annotation in data.annotations.values()
    ]
    payload = {
        "metadata": {
            "instr_id": data.metadata.instr_id,
            "sampling_id": data.metadata.sampling_id,
            "comparison": data.metadata.comparison,
            "answer": data.metadata.answer,
            "max_comparisons": data.metadata.max_comparisons,
            "prop_id": data.metadata.prop_id,
            "suffix": data.metadata.suffix,
            "model_id": data.metadata.model_id,
        },
        "annotations": annotations_payload,
    }
    tmp_path = manual_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(manual_path)


@beartype
def list_eval_files(
    instr_id: str, sampling_id: str, dataset_suffix: str | None
) -> list[Path]:
    base_dir = COT_EVAL_DIR / instr_id / sampling_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing evaluation directory: {base_dir}")
    eval_paths: list[Path] = []
    available_suffixes: set[str | None] = set()
    for pre_dir in base_dir.iterdir():
        if not pre_dir.is_dir():
            continue
        for dataset_dir in pre_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            parts = dataset_dir.name.split("_", maxsplit=5)
            suffix_in_dir: str | None = parts[5] if len(parts) == 6 else None
            available_suffixes.add(suffix_in_dir)
            if dataset_suffix is not None and not dataset_dir.name.endswith(
                f"_{dataset_suffix}"
            ):
                continue
            for eval_path in dataset_dir.glob("*.yaml"):
                eval_paths.append(eval_path)
    if not eval_paths:
        if dataset_suffix is not None:
            human_suffixes = sorted(
                "None" if suffix is None else suffix for suffix in available_suffixes
            )
            raise ValueError(
                f"No evaluation files found for instr_id={instr_id}, "
                f"sampling_id={sampling_id}, dataset_suffix={dataset_suffix}. "
                f"Available suffixes: {', '.join(human_suffixes) if human_suffixes else 'None'}"
            )
        raise ValueError(
            f"No evaluation files found for instr_id={instr_id}, "
            f"sampling_id={sampling_id}, dataset_suffix={dataset_suffix}"
        )
    return eval_paths


@beartype
def load_eval(eval_path: Path, cache: dict[Path, CotEval]) -> CotEval:
    cached = cache.get(eval_path)
    if cached is not None:
        return cached
    eval_data = CotEval.load(eval_path)
    cache[eval_path] = eval_data
    return eval_data


@beartype
def load_responses(
    eval_obj: CotEval,
    cache: dict[Path, CotResponses],
) -> CotResponses:
    responses_path = eval_obj.ds_params.cot_responses_path(
        instr_id=eval_obj.instr_id,
        model_id=eval_obj.model_id,
        sampling_params=eval_obj.sampling_params,
    )
    cached = cache.get(responses_path)
    if cached is not None:
        return cached
    responses = CotResponses.load(responses_path)
    cache[responses_path] = responses
    return responses


@beartype
def load_questions(
    eval_obj: CotEval, cache: dict[Path, dict[str, str]]
) -> dict[str, str]:
    dataset_path = eval_obj.ds_params.qs_dataset_path
    cached = cache.get(dataset_path)
    if cached is not None:
        return cached
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    question_by_qid = data.get("question_by_qid")
    if question_by_qid is None:
        raise ValueError(f"Missing question_by_qid in {dataset_path}")
    questions: dict[str, str] = {}
    for qid, details in question_by_qid.items():
        q_str = details.get("q_str")
        if q_str is None:
            raise ValueError(f"Missing q_str for {qid} in {dataset_path}")
        questions[str(qid)] = str(q_str)
    cache[dataset_path] = questions
    return questions


@beartype
def collect_pending_keys(
    eval_paths: list[Path],
    manual_data: dict[Path, ManualFileData],
    eval_cache: dict[Path, CotEval],
    rng: random.Random,
) -> deque[ResponseKey]:
    pending_keys: list[ResponseKey] = []
    for eval_path in tqdm(eval_paths, desc="Collecting responses", unit="file"):
        eval_obj = load_eval(eval_path, eval_cache)
        manual_entry = manual_data.get(eval_path)
        if manual_entry is None:
            manual_entry = ManualFileData(
                metadata=ManualFileMetadata(
                    instr_id=eval_obj.instr_id,
                    sampling_id=eval_obj.sampling_params.id,
                    comparison=eval_obj.ds_params.comparison,
                    answer=eval_obj.ds_params.answer,
                    max_comparisons=eval_obj.ds_params.max_comparisons,
                    prop_id=eval_obj.ds_params.prop_id,
                    suffix=eval_obj.ds_params.suffix,
                    model_id=eval_obj.model_id,
                ),
                annotations={},
            )
            manual_data[eval_path] = manual_entry
        else:
            metadata = manual_entry.metadata
            assert metadata.instr_id == eval_obj.instr_id
            assert metadata.sampling_id == eval_obj.sampling_params.id
            assert metadata.model_id == eval_obj.model_id
        for qid, response_results in eval_obj.results_by_qid.items():
            for response_id, result in response_results.items():
                if result.final_answer not in {"YES", "NO", "REFUSED", "UNKNOWN"}:
                    continue
                if result.final_answer == "NO" and result.equal_values not in {
                    "TRUE",
                    "FALSE",
                }:
                    continue
                key = (qid, response_id)
                if key in manual_entry.annotations:
                    continue
                pending_keys.append((eval_path, qid, response_id))
    rng.shuffle(pending_keys)
    return deque(pending_keys)


@beartype
def resolve_selected_response(
    key: ResponseKey,
    manual_data: dict[Path, ManualFileData],
    eval_cache: dict[Path, CotEval],
    responses_cache: dict[Path, CotResponses],
    question_cache: dict[Path, dict[str, str]],
) -> SelectedResponse:
    eval_path, qid, response_id = key
    eval_obj = load_eval(eval_path, eval_cache)
    responses_obj = load_responses(eval_obj, responses_cache)
    questions = load_questions(eval_obj, question_cache)
    response_map = responses_obj.responses_by_qid.get(qid)
    if response_map is None:
        raise KeyError(f"Missing responses for qid {qid}")
    response_text = response_map.get(response_id)
    if response_text is None:
        raise KeyError(f"Missing response {response_id} for qid {qid}")
    if not isinstance(response_text, str):
        raise TypeError("Expected response text to be a string")
    question_text = questions.get(qid)
    if question_text is None:
        raise KeyError(f"Missing question text for qid {qid}")
    manual_entry = manual_data[eval_path]
    result = eval_obj.results_by_qid[qid][response_id]
    return SelectedResponse(
        eval_path=eval_path,
        manual_path=manual_path_for_eval(eval_path),
        qid=qid,
        response_id=response_id,
        question=question_text,
        response=response_text,
        metadata=manual_entry.metadata,
        auto_label=auto_label_from_result(result),
    )


@beartype
def compute_kappa(
    manual_data: dict[Path, ManualFileData],
    eval_cache: dict[Path, CotEval],
) -> tuple[float, int, dict[NormalizedLabel, int], dict[NormalizedLabel, int], int]:
    manual_labels: list[Label] = []
    auto_labels: list[Label] = []
    label_order: tuple[NormalizedLabel, ...] = ("e", "n", "y", "ru")
    for eval_path, manual_entry in tqdm(
        manual_data.items(),
        total=len(manual_data),
        desc="Loading evaluation files",
        unit="file",
    ):
        if not manual_entry.annotations:
            continue
        eval_obj = load_eval(eval_path, eval_cache)
        for (qid, response_id), annotation in manual_entry.annotations.items():
            response_results = eval_obj.results_by_qid.get(qid)
            if response_results is None:
                raise KeyError(f"Missing results for qid {qid}")
            result = response_results.get(response_id)
            if result is None:
                raise KeyError(f"Missing result for response {response_id}")
            auto_labels.append(auto_label_from_result(result))
            manual_labels.append(annotation.label)
    if not manual_labels:
        raise ValueError("No manual annotations recorded; cannot compute κ.")
    total = len(manual_labels)
    manual_counts: dict[NormalizedLabel, int] = {label: 0 for label in label_order}
    auto_counts: dict[NormalizedLabel, int] = {label: 0 for label in label_order}
    match = 0
    for manual_label, auto_label in zip(manual_labels, auto_labels, strict=True):
        normalized_manual = _normalized_label(manual_label)
        normalized_auto = _normalized_label(auto_label)
        manual_counts[normalized_manual] += 1
        auto_counts[normalized_auto] += 1
        if normalized_manual == normalized_auto:
            match += 1
    observed = match / total
    expected = 0.0
    for label in label_order:
        expected += (manual_counts[label] / total) * (auto_counts[label] / total)
    if abs(1.0 - expected) < 1e-9:
        raise ValueError("Cannot compute κ because expected agreement is 1.0")
    kappa = (observed - expected) / (1.0 - expected)
    return kappa, total, manual_counts, auto_counts, match


@beartype
def _wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    assert 0 <= successes <= total
    assert total > 0
    assert 0.0 < confidence < 1.0
    z = 1.959963984540054  # sqrt(2) * erfc^-1(2 * (1 - confidence))
    phat = successes / total
    denominator = 1.0 + (z**2) / total
    centre = phat + (z**2) / (2.0 * total)
    margin = z * np.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * total)) / total)
    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)
    return lower, upper


@beartype
def _plot_label_distribution(
    manual_counts: dict[NormalizedLabel, int],
    auto_counts: dict[NormalizedLabel, int],
    total: int,
    output_path: Path,
) -> None:
    assert total > 0
    label_order: tuple[NormalizedLabel, ...] = ("y", "n", "e", "ru")
    manual_props = [manual_counts[label] / total for label in label_order]
    auto_props = [auto_counts[label] / total for label in label_order]
    manual_ci = [_wilson_interval(manual_counts[label], total) for label in label_order]
    auto_ci = [_wilson_interval(auto_counts[label], total) for label in label_order]
    manual_err = np.array(
        [
            [manual_props[idx] - manual_ci[idx][0] for idx in range(len(label_order))],
            [manual_ci[idx][1] - manual_props[idx] for idx in range(len(label_order))],
        ]
    )
    auto_err = np.array(
        [
            [auto_props[idx] - auto_ci[idx][0] for idx in range(len(label_order))],
            [auto_ci[idx][1] - auto_props[idx] for idx in range(len(label_order))],
        ]
    )
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(9, 6))
    positions = np.arange(len(label_order))
    width = 0.38
    manual_bars = ax.bar(
        positions - width / 2.0,
        manual_props,
        width=width,
        label="Human annotations",
        yerr=manual_err,
        capsize=4,
        color="#4E79A7",
        edgecolor="black",
        linewidth=1,
    )
    ax.bar(
        positions + width / 2.0,
        auto_props,
        width=width,
        label="Automatic labels",
        yerr=auto_err,
        capsize=4,
        color="#F28E2B",
        edgecolor="black",
        linewidth=1,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([format_label(label) for label in label_order])
    ax.set_ylabel("Proportion of responses")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_title("Label distribution: human vs automatic CoT labels")
    for idx, bar in enumerate(manual_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.015,
            f"{manual_counts[label_order[idx]]}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@beartype
def format_label(label: Literal["e", "n", "y", "r", "u", "ru"]) -> str:
    mapping: dict[str, str] = {
        "e": "NO (equal values)",
        "n": "NO",
        "y": "YES",
        "r": "REFUSED",
        "u": "UNKNOWN",
        "ru": "REFUSED/UNKNOWN",
    }
    return mapping[label]


@click.command()
@click.option(
    "--instr-id",
    type=str,
    default="instr-wm",
    show_default=True,
    help="Instruction ID to use (e.g., instr-wm).",
)
@click.option(
    "--sampling-id",
    type=str,
    default="T0.7_P0.9_M2000",
    show_default=True,
    help="Sampling identifier (temperature/top_p/max tokens).",
)
@click.option(
    "--dataset-suffix",
    type=str,
    default=None,
    help="Filter datasets ending with this suffix.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for sampling responses.",
)
@click.option(
    "--kappa-only",
    is_flag=True,
    help="Only compute Cohen's κ using existing annotations (skips labeling loop).",
)
@click.option(
    "--plot-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory to store reliability plots.",
)
def main(
    instr_id: str,
    sampling_id: str,
    dataset_suffix: str | None,
    seed: int | None,
    kappa_only: bool,
    plot_dir: Path | None,
) -> None:
    """Manually evaluate CoT responses and compute agreement with automatic labels."""
    rng = random.Random(seed)
    manual_data = load_manual_files()
    eval_paths = list_eval_files(
        instr_id=instr_id, sampling_id=sampling_id, dataset_suffix=dataset_suffix
    )
    eval_cache: dict[Path, CotEval] = {}
    responses_cache: dict[Path, CotResponses] = {}
    question_cache: dict[Path, dict[str, str]] = {}
    total_annotations = sum(len(entry.annotations) for entry in manual_data.values())
    click.echo(f"Loaded {total_annotations} existing annotations.")
    pending_keys: deque[ResponseKey] | None = None
    if not kappa_only:
        pending_keys = collect_pending_keys(
            eval_paths=eval_paths,
            manual_data=manual_data,
            eval_cache=eval_cache,
            rng=rng,
        )
        while True:
            if not pending_keys:
                click.echo("No remaining responses to annotate.")
                break
            current_key = pending_keys[0]
            selection = resolve_selected_response(
                key=current_key,
                manual_data=manual_data,
                eval_cache=eval_cache,
                responses_cache=responses_cache,
                question_cache=question_cache,
            )
            click.echo("\n" + "=" * 80)
            click.echo(f"Property: {selection.metadata.prop_id}")
            click.echo(f"Question ID: {selection.qid}")
            click.echo(f"Response ID: {selection.response_id}")
            click.echo("\nQuestion:")
            click.echo(selection.question.strip())
            click.echo("\nResponse:")
            click.echo(selection.response.strip())
            click.echo(
                "\nOptions: [y] YES, [n] NO, [e] NO(equal), [r] REFUSED, [u] UNKNOWN, [s] Skip, [q] Quit"
            )
            click.echo("Press key: ", nl=False)
            choice = click.getchar()
            click.echo()
            if not choice:
                continue
            choice = choice.lower()
            if choice == "q":
                break
            if choice == "s":
                if len(pending_keys) > 1:
                    pending_keys.rotate(-1)
                continue
            if choice not in {"y", "n", "e", "r", "u"}:
                click.echo("Invalid choice, please try again.")
                continue
            manual_entry = manual_data[selection.eval_path]
            annotation = ManualAnnotation(
                qid=selection.qid,
                response_id=selection.response_id,
                label=cast(Label, choice),
            )
            manual_entry.annotations[(selection.qid, selection.response_id)] = (
                annotation
            )
            save_manual_file(selection.manual_path, manual_entry)
            total_annotations += 1
            pending_keys.popleft()
            click.echo(
                f"Stored annotation ({format_label(cast(Label, choice))}) - total annotated: {total_annotations}"
            )
    if total_annotations == 0:
        click.echo("No annotations available; skipping κ computation.")
        return
    try:
        kappa, total, manual_counts, auto_counts, match = compute_kappa(
            manual_data=manual_data, eval_cache=eval_cache
        )
    except ValueError as error:
        click.echo(str(error))
        return
    click.echo("\n" + "-" * 80)
    click.echo(f"Cohen's κ across {total} annotations: {kappa:.3f}")
    click.echo("Label counts (manual vs automatic):")
    for label_key in ("y", "n", "e", "ru"):
        label = cast(Literal["e", "n", "y", "ru"], label_key)
        click.echo(
            f"- {format_label(label)}: manual={manual_counts[label]} auto={auto_counts[label]}"
        )
    accuracy = match / total
    acc_low, acc_high = _wilson_interval(match, total)
    click.echo(
        f"Overall accuracy: {accuracy:.3f} (95% CI {acc_low:.3f}–{acc_high:.3f})"
    )
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "human_vs_llm_label_distribution.pdf"
        _plot_label_distribution(
            manual_counts=manual_counts,
            auto_counts=auto_counts,
            total=total,
            output_path=plot_path,
        )
        click.echo(f"Saved label distribution plot to {plot_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
