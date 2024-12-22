#!/usr/bin/env python3

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from chainscope.typing import *
from chainscope.utils import MODELS_MAP


def load_eval_pair(
    dataset_id: str, model_id: str, instr_id: str, sampling_params: SamplingParams
) -> tuple[DirectEval, CotEval]:
    """Load corresponding direct and CoT evaluations for a model/dataset pair."""
    # Load direct eval
    direct_path = (
        DATA_DIR
        / "direct_eval"
        / instr_id
        / dataset_id
        / f"{model_id.replace('/', '__')}.yaml"
    )
    direct_eval = DirectEval.from_yaml_file(direct_path)
    if isinstance(direct_eval, list):
        direct_eval = direct_eval[0]

    # Load CoT eval
    sp_id = sampling_params.get_identifier()
    cot_path = (
        DATA_DIR
        / "cot_eval"
        / instr_id
        / sp_id
        / dataset_id
        / f"{model_id.replace('/', '__')}.yaml"
    )
    cot_eval = CotEval.from_yaml_file(cot_path)
    if isinstance(cot_eval, list):
        cot_eval = cot_eval[0]

    return direct_eval, cot_eval


def compute_accuracies(
    direct_eval: DirectEval, cot_eval: CotEval, gt_answer: Literal["YES", "NO"]
):
    """Compute direct probabilities and CoT accuracies for each question."""
    direct_probs = []
    cot_accs = []

    for qid in direct_eval.probs_by_qid:
        # Get direct probability of correct answer
        probs = direct_eval.probs_by_qid[qid]
        direct_prob = probs.p_yes if gt_answer == "YES" else probs.p_no
        direct_probs.append(direct_prob)

        # Calculate CoT accuracy for this question
        cot_results = cot_eval.results_by_qid[qid]
        correct = sum(1 for r in cot_results.values() if r == gt_answer)
        total = len(cot_results)
        cot_acc = correct / total if total > 0 else 0
        cot_accs.append(cot_acc)

    return np.array(direct_probs), np.array(cot_accs)


def compute_accuracies_by_answer(
    direct_eval: DirectEval, cot_eval: CotEval
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute direct probabilities and CoT accuracies for YES and NO answers separately."""
    direct_yes_probs = []
    direct_no_probs = []
    cot_yes_accs = []
    cot_no_accs = []

    for qid in direct_eval.probs_by_qid.keys():
        probs = direct_eval.probs_by_qid[qid]
        direct_yes_probs.append(probs.p_yes)
        direct_no_probs.append(probs.p_no)

        cot_results = cot_eval.results_by_qid[qid]
        total = len(cot_results)
        if total > 0:
            yes_count = sum(1 for r in cot_results.values() if r == "YES")
            no_count = sum(1 for r in cot_results.values() if r == "NO")
            cot_yes_accs.append(yes_count / total)
            cot_no_accs.append(no_count / total)
        else:
            cot_yes_accs.append(0)
            cot_no_accs.append(0)

    return (
        np.array(direct_yes_probs),
        np.array(direct_no_probs),
        np.array(cot_yes_accs),
        np.array(cot_no_accs),
    )


def plot_model_comparison(
    direct_probs: np.ndarray, cot_accs: np.ndarray, model_id: str
):
    """Create scatter plot for a single model."""
    plt.figure(figsize=(8, 8))

    # Create scatter plot with semi-transparent circles
    plt.scatter(direct_probs, cot_accs, alpha=0.5)

    # Add diagonal line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

    # Calculate and display correlation coefficient
    corr, _ = stats.pearsonr(direct_probs, cot_accs)
    plt.title(f"Model: {model_id}\nCorrelation: {corr:.3f}")

    plt.xlabel("P(Correct) Direct Answer")
    plt.ylabel("CoT Accuracy")
    plt.tight_layout()
    return corr


def plot_model_comparison_by_answer(
    direct_yes_probs: np.ndarray,
    direct_no_probs: np.ndarray,
    cot_yes_accs: np.ndarray,
    cot_no_accs: np.ndarray,
    model_id: str,
) -> tuple[float, float]:
    """Create separate scatter plots for YES and NO answers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot YES comparisons
    ax1.scatter(direct_yes_probs, cot_yes_accs, alpha=0.5)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    corr_yes, _ = stats.pearsonr(direct_yes_probs, cot_yes_accs)
    ax1.set_title(f"YES Answers\nCorrelation: {corr_yes:.3f}")
    ax1.set_xlabel("P(YES) Direct Answer")
    ax1.set_ylabel("CoT YES Percentage")

    # Plot NO comparisons
    ax2.scatter(direct_no_probs, cot_no_accs, alpha=0.5)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    corr_no, _ = stats.pearsonr(direct_no_probs, cot_no_accs)
    ax2.set_title(f"NO Answers\nCorrelation: {corr_no:.3f}")
    ax2.set_xlabel("P(NO) Direct Answer")
    ax2.set_ylabel("CoT NO Percentage")

    plt.suptitle(f"Model: {model_id}")
    plt.tight_layout()
    return corr_yes, corr_no


@click.command()
@click.option("-d", "--dataset-id", type=str, required=True)
@click.option("-i", "--instr-id", type=str, default="instr-v0")
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=2000)
@click.option("--output-dir", type=click.Path(), default="plots")
def main(
    dataset_id: str,
    instr_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    output_dir: str,
):
    model_ids: list[str] = list(MODELS_MAP.values())

    # Load dataset to get ground truth answer
    qs_dataset = QsDataset.load(dataset_id)
    gt_answer = qs_dataset.answer

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Update data structures to store correlations
    correlations = {
        "combined": {},
        "yes": {},
        "no": {},
    }
    differences_by_model = {}

    # Process each model
    for model_id in model_ids:
        try:
            direct_eval, cot_eval = load_eval_pair(
                dataset_id, model_id, instr_id, sampling_params
            )

            # Original combined analysis
            direct_probs, cot_accs = compute_accuracies(
                direct_eval, cot_eval, gt_answer
            )
            corr = plot_model_comparison(direct_probs, cot_accs, model_id)
            plt.savefig(output_path / f"{model_id.replace('/', '_')}_comparison.png")
            plt.close()

            # New separate YES/NO analysis
            direct_yes_probs, direct_no_probs, cot_yes_accs, cot_no_accs = (
                compute_accuracies_by_answer(direct_eval, cot_eval)
            )
            corr_yes, corr_no = plot_model_comparison_by_answer(
                direct_yes_probs, direct_no_probs, cot_yes_accs, cot_no_accs, model_id
            )
            plt.savefig(
                output_path / f"{model_id.replace('/', '_')}_comparison_by_answer.png"
            )
            plt.close()

            correlations["combined"][model_id] = corr
            correlations["yes"][model_id] = corr_yes
            correlations["no"][model_id] = corr_no
            differences_by_model[model_id] = cot_accs - direct_probs

        except Exception as e:
            logging.warning(f"Failed to process model {model_id}: {e}")

    # Create separate figures for correlation and advantage plots
    # Correlation plot
    plt.figure(figsize=(10, 6))
    model_names = list(correlations["combined"].keys())
    combined_corrs = [correlations["combined"][m] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.6  # Made wider since we only have one bar now

    plt.bar(x, combined_corrs, width, label="Combined")
    plt.xticks(x, [m.split("/")[-1] for m in model_names], rotation=45)
    plt.ylabel("Correlation Coefficient")
    plt.title("Correlation between P(correct) and CoT Accuracy")
    plt.tight_layout()
    plt.savefig(output_path / "cot_acc_vs_direct_corr_all_models.png")
    plt.close()

    # New boxplot for differences
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [differences_by_model[m] for m in model_names],
        labels=[m.split("/")[-1] for m in model_names],
    )
    plt.xticks(rotation=45)
    plt.ylabel("CoT acc - P(correct)")
    plt.title("Difference between CoT Accuracy and P(correct)")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "cot_acc_diff_over_direct_all_models.png")
    plt.close()


if __name__ == "__main__":
    main()
