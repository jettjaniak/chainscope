#!/usr/bin/env python3

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from beartype import beartype

from chainscope.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

LLM_TO_HUMAN_CATEGORY: dict[str, tuple[str, ...]] = {
    "answer-flipping": ("answer flipping",),
    "fact-manipulation": ("fact manipulation",),
    "argument-switching": ("different arguments",),
    "other": (
        "bad arguments",
        "wrong comparison",
        "missing step",
        "other",
    ),
}

MANUAL_CATEGORIES: tuple[str, ...] = (
    "answer flipping",
    "fact manipulation",
    "different arguments",
    "bad arguments",
    "wrong comparison",
    "missing step",
    "other",
)

TARGET_CATEGORIES: tuple[str, ...] = (
    "answer flipping",
    "fact manipulation",
    "different arguments",
    "other",
)

UNION_TARGET: str = "any-pair-level-pattern"

TARGET_CATEGORIES = (*TARGET_CATEGORIES, UNION_TARGET)

TARGET_TO_HUMAN_CATEGORIES: dict[str, tuple[str, ...]] = {
    "answer flipping": ("answer flipping",),
    "fact manipulation": ("fact manipulation",),
    "different arguments": ("different arguments",),
    "other": (
        "bad arguments",
        "wrong comparison",
        "missing step",
        "other",
    ),
    UNION_TARGET: ("fact manipulation", "different arguments", "other"),
}

TARGET_TO_LLM_CATEGORY: dict[str, str] = {
    "answer flipping": "answer-flipping",
    "fact manipulation": "fact-manipulation",
    "different arguments": "argument-switching",
    "other": "other",
}

ANSWER_FLIPPING_LABEL = "answer-flipping"
QUESTION_ANALYSIS_KEYS: tuple[str, ...] = ("q1_analysis", "q2_analysis")
RESPONSE_PATTERN_KEYS: tuple[str, ...] = (
    "unfaithfulness_patterns",
    "evidence_of_unfaithfulness",
)

SAMPLING_DIRNAME = "T0.0_P0.9_M8000"
MAX_MISSING_DEBUG = 300
BOOTSTRAP_SAMPLES = 2000

MODEL_ID_REMAP: dict[str, str] = {
    "gpt-4o": "gpt-4o-2024-08-06",
}


@dataclass(frozen=True)
class ManualAnnotation:
    model_id: str
    qid: str
    categories: dict[str, bool]


@dataclass(frozen=True)
class ManualSummary:
    model_id: str
    total_pairs: int
    annotated_pairs: int


@beartype
def _normalize_manual_categories(raw_categories: dict[str, Any]) -> dict[str, bool]:
    assert raw_categories is not None
    normalized: dict[str, bool] = {}
    for category in MANUAL_CATEGORIES:
        normalized[category] = bool(raw_categories.get(category, False))
    return normalized


@beartype
def load_manual_annotations(
    case_studies_dir: Path, allow_models: set[str] | None
) -> tuple[list[ManualAnnotation], list[ManualSummary]]:
    assert case_studies_dir.is_dir(), f"Missing case studies dir: {case_studies_dir}"
    annotations: list[ManualAnnotation] = []
    summary_counts: dict[str, dict[str, int]] = {}

    for model_path in sorted(case_studies_dir.glob("*.yaml")):
        model_id = model_path.stem
        if allow_models is not None and model_id not in allow_models:
            continue

        canonical_model_id = MODEL_ID_REMAP.get(model_id, model_id)

        with model_path.open("r") as handle:
            model_data = yaml.safe_load(handle)

        assert isinstance(model_data, dict)
        total_pairs = 0
        annotated_pairs = 0
        for qid, qdata in model_data.items():
            total_pairs += 1
            manual = qdata.get("manual_analysis")
            if manual is None:
                continue

            raw_categories = manual.get("categories")
            if raw_categories is None:
                continue

            categories = _normalize_manual_categories(raw_categories)
            annotated_pairs += 1
            annotations.append(
                ManualAnnotation(
                    model_id=canonical_model_id, qid=str(qid), categories=categories
                )
            )

        counts = summary_counts.setdefault(
            canonical_model_id, {"total": 0, "annotated": 0}
        )
        counts["total"] += total_pairs
        counts["annotated"] += annotated_pairs

    summaries: list[ManualSummary] = [
        ManualSummary(
            model_id=model_id,
            total_pairs=counts["total"],
            annotated_pairs=counts["annotated"],
        )
        for model_id, counts in sorted(summary_counts.items())
    ]

    return annotations, summaries


@beartype
def _collect_model_ids(
    annotations: Iterable[ManualAnnotation],
) -> set[str]:
    model_ids: set[str] = set()
    for annotation in annotations:
        model_ids.add(annotation.model_id)
    return model_ids


@beartype
def load_pattern_lookup(
    pattern_dir: Path, model_ids: set[str]
) -> tuple[dict[tuple[str, str], set[str]], dict[str, int]]:
    assert pattern_dir.is_dir(), f"Missing pattern dir: {pattern_dir}"
    lookup: dict[tuple[str, str], set[str]] = {}
    coverage: dict[str, int] = {}

    for pattern_path in sorted(pattern_dir.rglob("*.yaml")):
        model_name = pattern_path.stem
        canonical_model = MODEL_ID_REMAP.get(model_name, model_name)
        relevant_model = canonical_model in model_ids
        coverage.setdefault(canonical_model, 0)

        with pattern_path.open("r") as handle:
            pattern_data = yaml.safe_load(handle)

        assert isinstance(pattern_data, dict)
        analyses = pattern_data.get("pattern_analysis_by_qid")
        if analyses is None:
            continue

        assert isinstance(analyses, dict)
        count_with_categories = 0
        for qid, analysis in analyses.items():
            assert isinstance(analysis, dict)
            categories = analysis.get("categorization_for_pair")
            augmented = _pair_has_answer_flipping(analysis)
            if categories is None and not augmented:
                continue

            count_with_categories += 1
            if not relevant_model:
                continue

            key = (canonical_model, str(qid))
            base_categories: list[str] = categories or []
            category_set = set(base_categories)
            if augmented:
                category_set.add(ANSWER_FLIPPING_LABEL)
            existing = lookup.get(key)
            if existing is None:
                lookup[key] = category_set
                continue
            if existing != category_set:
                logging.warning(
                    "Conflicting pattern categories for %s: existing=%s new=%s. Merging.",
                    key,
                    sorted(existing),
                    sorted(category_set),
                )
                existing |= category_set

        if count_with_categories > 0:
            coverage[canonical_model] += count_with_categories

    return lookup, coverage


@beartype
def _human_label_for_category(categories: dict[str, bool], target: str) -> int:
    category_list = TARGET_TO_HUMAN_CATEGORIES[target]
    return int(any(categories[category] for category in category_list))


@beartype
def _llm_label_for_category(category_set: set[str], target: str) -> int:
    if target == UNION_TARGET:
        return int(
            any(
                label in category_set
                for label in ("fact-manipulation", "argument-switching", "other")
            )
        )
    mapped = TARGET_TO_LLM_CATEGORY[target]
    return int(mapped in category_set)


@beartype
def _pair_has_answer_flipping(analysis: dict[str, Any]) -> bool:
    for question_key in QUESTION_ANALYSIS_KEYS:
        question_analysis = analysis.get(question_key)
        if not isinstance(question_analysis, dict):
            continue
        responses = question_analysis.get("responses")
        if not isinstance(responses, dict):
            continue
        for response in responses.values():
            if not isinstance(response, dict):
                continue
            classification = response.get("answer_flipping_classification")
            if isinstance(classification, str) and classification.upper() == "YES":
                return True
            for pattern_key in RESPONSE_PATTERN_KEYS:
                patterns = response.get(pattern_key)
                if not isinstance(patterns, list):
                    continue
                for pattern in patterns:
                    if isinstance(pattern, str) and pattern == ANSWER_FLIPPING_LABEL:
                        return True
    return False


@beartype
def _compute_counts(
    human_labels: list[int], llm_labels: list[int]
) -> tuple[int, int, int, int]:
    assert len(human_labels) == len(llm_labels) > 0
    tp = tn = fp = fn = 0
    for human, llm in zip(human_labels, llm_labels, strict=True):
        if human == 1 and llm == 1:
            tp += 1
        elif human == 0 and llm == 0:
            tn += 1
        elif human == 0 and llm == 1:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


@beartype
def compute_binary_metrics(
    human_labels: list[int], llm_labels: list[int]
) -> dict[str, float | int | None]:
    tp, tn, fp, fn = _compute_counts(human_labels, llm_labels)
    total = tp + tn + fp + fn
    assert total > 0

    po = (tp + tn) / total
    human_yes = (tp + fn) / total
    llm_yes = (tp + fp) / total
    human_no = 1.0 - human_yes
    llm_no = 1.0 - llm_yes
    pe = human_yes * llm_yes + human_no * llm_no

    if abs(1.0 - pe) < 1e-9:
        raise ValueError("Cannot compute κ because expected agreement is 1.0")

    kappa = (po - pe) / (1.0 - pe)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or precision + recall == 0:
        f1: float | None = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pairs": total,
        "human_positive": tp + fn,
        "llm_positive": tp + fp,
    }


@beartype
def compute_kappa(
    annotations: list[ManualAnnotation],
    pattern_lookup: dict[tuple[str, str], set[str]],
) -> tuple[
    dict[str, dict[str, float | int | None]],
    dict[str, dict[str, list[tuple[str, str]]]],
    int,
    int,
    list[tuple[str, str]],
    dict[str, dict[str, int]],
    dict[str, list[tuple[int, int]]],
]:
    human_labels: dict[str, list[int]] = {target: [] for target in TARGET_CATEGORIES}
    llm_labels: dict[str, list[int]] = {target: [] for target in TARGET_CATEGORIES}
    label_records: dict[str, list[tuple[str, str, int, int]]] = {
        target: [] for target in TARGET_CATEGORIES
    }
    label_pairs: dict[str, list[tuple[int, int]]] = {
        target: [] for target in TARGET_CATEGORIES
    }

    missing_pairs = 0
    considered_pairs = 0
    missing_examples: list[tuple[str, str]] = []
    per_model_stats: dict[str, dict[str, int]] = {}

    for annotation in annotations:
        key = (annotation.model_id, annotation.qid)
        category_set = pattern_lookup.get(key)
        stats = per_model_stats.setdefault(
            annotation.model_id, {"total": 0, "with_pattern": 0}
        )
        stats["total"] += 1
        if category_set is None:
            missing_pairs += 1
            if len(missing_examples) < MAX_MISSING_DEBUG:
                missing_examples.append(key)
            continue

        considered_pairs += 1
        stats["with_pattern"] += 1
        for target in TARGET_CATEGORIES:
            human = _human_label_for_category(annotation.categories, target)
            llm = _llm_label_for_category(category_set, target)
            human_labels[target].append(human)
            llm_labels[target].append(llm)
            label_records[target].append(
                (annotation.model_id, annotation.qid, human, llm)
            )
            label_pairs[target].append((human, llm))

    metrics_by_category: dict[str, dict[str, float | int | None]] = {}
    examples_by_category: dict[str, dict[str, list[tuple[str, str]]]] = {
        target: {"fp": [], "fn": []} for target in TARGET_CATEGORIES
    }

    for target in TARGET_CATEGORIES:
        labels_human = human_labels[target]
        labels_llm = llm_labels[target]
        if not labels_human:
            continue

        metrics_by_category[target] = compute_binary_metrics(labels_human, labels_llm)

        # Collect TN and FN examples (model_id, qid)
        fp_examples: list[tuple[str, str]] = []
        fn_examples: list[tuple[str, str]] = []
        for model_id, qid, human, llm in label_records[target]:
            if human == 0 and llm == 1 and len(fp_examples) < 3:
                fp_examples.append((model_id, qid))
            if human == 1 and llm == 0 and len(fn_examples) < 3:
                fn_examples.append((model_id, qid))
            if len(fp_examples) >= 3 and len(fn_examples) >= 3:
                break
        examples_by_category[target]["fp"] = fp_examples
        examples_by_category[target]["fn"] = fn_examples

    return (
        metrics_by_category,
        examples_by_category,
        considered_pairs,
        missing_pairs,
        missing_examples,
        per_model_stats,
        label_pairs,
    )


@beartype
def _classification_counts(pairs: list[tuple[int, int]]) -> tuple[int, int, int]:
    assert len(pairs) > 0
    tp = fp = fn = 0
    for human, llm in pairs:
        if human == 1 and llm == 1:
            tp += 1
        elif human == 0 and llm == 1:
            fp += 1
        elif human == 1 and llm == 0:
            fn += 1
    return tp, fp, fn


@beartype
def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    assert tp >= 0 and fp >= 0 and fn >= 0
    if tp == 0:
        if fp + fn == 0:
            return 1.0
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (2.0 * precision * recall) / (precision + recall)


@beartype
def _bootstrap_f1(
    pairs: list[tuple[int, int]],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    assert len(pairs) > 0
    base_tp, base_fp, base_fn = _classification_counts(pairs)
    base_f1 = _f1_from_counts(base_tp, base_fp, base_fn)
    if len(pairs) == 1 or n_samples == 0:
        return base_f1, base_f1, base_f1
    scores = np.empty(n_samples, dtype=float)
    population_size = len(pairs)
    for idx in range(n_samples):
        sample_indices = rng.integers(0, population_size, size=population_size)
        sample_pairs = [pairs[int(index)] for index in sample_indices]
        tp, fp, fn = _classification_counts(sample_pairs)
        scores[idx] = _f1_from_counts(tp, fp, fn)
    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    return base_f1, lower, upper


@beartype
def _plot_f1_scores(
    f1_stats: dict[str, tuple[float, float, float]], output_path: Path
) -> None:
    assert f1_stats
    categories = [target for target in TARGET_CATEGORIES if target in f1_stats]
    assert categories
    values = [f1_stats[target][0] for target in categories]
    lowers = [f1_stats[target][1] for target in categories]
    uppers = [f1_stats[target][2] for target in categories]
    errors = np.array(
        [
            [values[idx] - lowers[idx] for idx in range(len(categories))],
            [uppers[idx] - values[idx] for idx in range(len(categories))],
        ]
    )
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(categories))
    bars = ax.bar(
        positions,
        values,
        yerr=errors,
        capsize=4,
        color="#4E79A7",
        edgecolor="black",
        linewidth=1,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [category.title() for category in categories], rotation=20, ha="right"
    )
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Unfaithfulness pattern autorater: F1 with 95% CI")
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{values[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option(
    "--models",
    "-m",
    multiple=True,
    help="Restrict computation to these model IDs (stem names).",
)
@click.option(
    "--case-studies-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=DATA_DIR / "case_studies" / "analyzed",
    show_default=True,
)
@click.option(
    "--pattern-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=DATA_DIR / "unfaithfulness_pattern_eval" / SAMPLING_DIRNAME,
    show_default=True,
)
@click.option(
    "--list-missing",
    is_flag=True,
    help="Print every manual pair that is still missing LLM pattern analysis.",
)
@click.option(
    "--plot-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory to write F1 plots.",
)
@click.option(
    "--bootstrap-seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed used for bootstrap confidence intervals.",
)
def main(
    models: tuple[str, ...],
    case_studies_dir: Path,
    pattern_dir: Path,
    list_missing: bool,
    plot_dir: Path | None,
    bootstrap_seed: int,
) -> None:
    """Compute Cohen's κ between manual annotations and LLM pattern labels."""
    allow_models: set[str] | None = set(models) if models else None

    manual_annotations, manual_summaries = load_manual_annotations(
        case_studies_dir, allow_models
    )
    if not manual_annotations:
        raise ValueError("No manual annotations found for the given filters.")

    click.echo("Manual annotation coverage per model:")
    for summary in manual_summaries:
        coverage = (
            f"{summary.annotated_pairs}/{summary.total_pairs}"
            if summary.total_pairs > 0
            else "0/0"
        )
        click.echo(f"- {summary.model_id}: {coverage} annotated pairs")
    click.echo()

    model_ids = (
        allow_models
        if allow_models is not None
        else _collect_model_ids(manual_annotations)
    )
    pattern_lookup, pattern_coverage = load_pattern_lookup(pattern_dir, model_ids)
    click.echo("Pattern evaluation coverage per model (LLM analyses):")
    for model_id in sorted(pattern_coverage.keys()):
        marker = "*" if model_id in model_ids else " "
        count = pattern_coverage[model_id]
        click.echo(f"{marker} {model_id}: {count} analyzed pairs")
    missing_models = sorted(model_ids - pattern_coverage.keys())
    if missing_models:
        click.echo()
        click.echo("Models with manual annotations but no LLM pattern analysis:")
        for model_id in missing_models:
            click.echo(f"- {model_id}")
    click.echo()

    (
        metrics_by_category,
        examples_by_category,
        considered_pairs,
        missing_pairs,
        missing_examples,
        per_model_overlap,
        label_pairs,
    ) = compute_kappa(manual_annotations, pattern_lookup)

    if not metrics_by_category:
        if missing_examples:
            click.echo("Sample pairs missing LLM pattern analysis (model_id :: qid):")
            for model_id, qid in missing_examples:
                click.echo(f"- {model_id} :: {qid}")
            if missing_pairs > len(missing_examples):
                click.echo(
                    f"... plus {missing_pairs - len(missing_examples)} additional missing pairs."
                )
        raise ValueError(
            f"No overlapping pairs between manual data and pattern evaluations. (models={model_ids}, pattern_dir={pattern_dir}, case_studies_dir={case_studies_dir}, allow_models={allow_models}, manual_annotations={len(manual_annotations)}, considered_pairs={considered_pairs}, missing_pairs={missing_pairs})"
        )

    click.echo(f"Total manual pairs considered: {len(manual_annotations)}")
    click.echo(f"Pairs with pattern analysis:   {considered_pairs}")
    click.echo(f"Pairs missing pattern data:    {missing_pairs}")
    click.echo()
    if per_model_overlap:
        click.echo("Overlap coverage per model:")
        for model_id in sorted(per_model_overlap.keys()):
            stats = per_model_overlap[model_id]
            click.echo(
                f"- {model_id}: {stats['with_pattern']}/{stats['total']} pairs have pattern evals"
            )
        click.echo()

    if list_missing and missing_pairs > 0:
        pattern_keys = set(pattern_lookup.keys())
        full_missing = sorted(
            (annotation.model_id, annotation.qid)
            for annotation in manual_annotations
            if (annotation.model_id, annotation.qid) not in pattern_keys
        )
        click.echo("All pairs missing pattern analysis (model_id :: qid):")
        for model_id, qid in full_missing:
            click.echo(f"- {model_id} :: {qid}")
        click.echo()

    click.echo("Results per category:")
    for target in TARGET_CATEGORIES:
        metrics = metrics_by_category.get(target)
        if not metrics:
            click.echo(f"- {target}: no overlapping annotations")
            continue

        click.echo(f"- {target}")
        click.echo(
            f"  κ = {metrics['kappa']:.3f} "
            f"(n={metrics['n_pairs']}, human+= {metrics['human_positive']}, "
            f"llm+= {metrics['llm_positive']})"
        )
        precision = (
            f"{metrics['precision']:.3f}"
            if isinstance(metrics["precision"], float)
            else "N/A"
        )
        recall = (
            f"{metrics['recall']:.3f}"
            if isinstance(metrics["recall"], float)
            else "N/A"
        )
        f1 = f"{metrics['f1']:.3f}" if isinstance(metrics["f1"], float) else "N/A"
        click.echo(
            f"  tp={metrics['tp']}, tn={metrics['tn']}, "
            f"fp={metrics['fp']}, fn={metrics['fn']}, "
            f"precision={precision}, recall={recall}, f1={f1}"
        )
        fp_examples = examples_by_category[target]["fp"]
        fn_examples = examples_by_category[target]["fn"]
        if fp_examples:
            click.echo("    FP examples (LLM only):")
            for model_id, qid in fp_examples:
                click.echo(f"      - {model_id} :: {qid}")
        if fn_examples:
            click.echo("    FN examples (human only):")
            for model_id, qid in fn_examples:
                click.echo(f"      - {model_id} :: {qid}")
    if plot_dir is not None:
        f1_stats: dict[str, tuple[float, float, float]] = {}
        for idx, target in enumerate(TARGET_CATEGORIES):
            pairs = label_pairs.get(target, [])
            if not pairs:
                continue
            rng = np.random.default_rng(bootstrap_seed + idx)
            mean, lower, upper = _bootstrap_f1(
                pairs=pairs, n_samples=BOOTSTRAP_SAMPLES, rng=rng
            )
            f1_stats[target] = (mean, lower, upper)
        if f1_stats:
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "human_vs_llm_unf_pattern_f1_scores.pdf"
            _plot_f1_scores(f1_stats=f1_stats, output_path=plot_path)
            click.echo(f"Saved F1 plot to {plot_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
