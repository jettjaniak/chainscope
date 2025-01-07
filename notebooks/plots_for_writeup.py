# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chainscope.typing import *
from chainscope.utils import MODELS_MAP

df = pd.read_pickle(DATA_DIR / "df.pkl")
filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
df = df[~df.prop_id.isin(filter_prop_ids)]
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate


# %%
def plot_comparison(df: pd.DataFrame, model_id: str, comparison: str, answer: str):
    condition_df = df[
        (df.model_id == model_id)
        & (df.comparison == comparison)
        & (df.answer == answer)
    ]

    # Difference plot
    condition_df_pivot = condition_df.pivot(
        index=["prop_id", "qid"], columns="mode", values="p_correct"
    )
    condition_df_pivot["difference"] = (
        condition_df_pivot["cot"] - condition_df_pivot["direct"]
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=condition_df_pivot.reset_index(),
        y="prop_id",
        x="difference",
        orient="h",
        color="lightblue",
    )
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.title(
        f"Difference in P(Correct) (CoT - Direct) for {model_id}\n({comparison}_{answer})"
    )
    plt.xlabel("Difference in P(Correct)")
    plt.ylabel("Property")
    plt.tight_layout()
    plt.show()


# %%


def get_param_count(model_name: str) -> float:
    """Extract parameter count from model name in billions using regex."""
    name_lower = model_name.lower()
    match = re.search(r"[-]?(\d+\.?\d*)b", name_lower)
    return float(match.group(1)) if match else float("inf")


def get_model_display_name(model_id: str) -> str:
    """Extract the display name from a model ID."""
    return model_id.split("/")[-1]


def sort_models(model_ids: list[str]) -> list[str]:
    """Sort model IDs by name prefix and parameter count."""
    return sorted(
        model_ids,
        key=lambda x: (
            get_model_display_name(x).split("-")[0].lower(),
            get_param_count(get_model_display_name(x)),
        ),
    )


def save_probability_distributions(
    data: pd.DataFrame,
    mode: str,
    model: str,
    save_dir: Path,
) -> None:
    """Save probability distribution plots for a given mode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    title = f"Probability Distributions ({mode.title()}) - {model}"

    if len(data) > 0:
        probabilities = [data["p_yes"], data["p_no"], data["p_correct"]]
        labels = ["P(Yes)", "P(No)", "P(Correct)"]

        if mode == "cot":
            probabilities.append(data["unknown_rate"])
            labels.append("Unknown Rate")

        positions = list(range(1, len(probabilities) + 1))
        ax.boxplot(probabilities, positions=positions, tick_labels=labels)

        # Add median values
        for i, median in enumerate([d.median() for d in probabilities]):
            ax.text(
                positions[i],
                median,
                f"{median:.2f}",
                horizontalalignment="center",
                verticalalignment="bottom",
                weight="bold",
            )

        ax.set_title(f"{title}\n({len(data['dataset_id'].unique())} datasets)", pad=15)
        ax.set_ylabel("Probability")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / f"prob_dist_{mode}_{model}.png")
    plt.close()


def save_model_comparisons(
    df: pd.DataFrame,
    save_dir: Path,
    model: str | None = None,
    saturation_direct_threshold: float = 0.25,
) -> None:
    """Save comparison plots between CoT and Direct modes."""
    if model is None:
        # Plot for all models
        x_labels = []

        differences = []
        high_diff_percentages = []

        differences_without_saturation = []
        high_diff_percentages_without_saturation = []

        model_ids = sort_models(df["model_id"].unique())

        for model_id in model_ids:
            model_data = df[df["model_id"] == model_id].pivot(
                index=["dataset_id", "qid"], columns="mode", values="p_correct"
            )
            filtered_data = model_data[
                model_data["direct"] <= saturation_direct_threshold
            ]

            diff = model_data["cot"] - model_data["direct"]
            differences.append(diff)

            filtered_diff = filtered_data["cot"] - filtered_data["direct"]
            differences_without_saturation.append(filtered_diff)

            x_labels.append(get_model_display_name(model_id))
            high_diff_percentage = (diff.between(0.5, 1.0).sum() / len(diff)) * 100
            high_diff_percentages.append(high_diff_percentage)

            filtered_high_diff_percentage = (
                filtered_diff.between(0.5, 1.0).sum() / len(filtered_diff)
            ) * 100
            high_diff_percentages_without_saturation.append(
                filtered_high_diff_percentage
            )

        # Save violin plot
        fig, ax = plt.subplots(figsize=(12, 6))
        violin_parts = ax.violinplot(differences, showmedians=True)

        for pc in violin_parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.7)
        violin_parts["cmedians"].set_color("red")

        ax.set_xticks(range(1, len(x_labels) + 1))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Difference in Prob")
        ax.set_title("Distribution of Differences (CoT - Direct)")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_dir / "all_models_differences.png")
        plt.close()

        # Save percentage histogram
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            x_labels,
            high_diff_percentages,
            edgecolor="black",
            color="lightblue",
        )
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Percentage of data points (%)")
        ax.set_title(
            "Percentage of questions with CoT Accuracy - P(Correct) Differences between 0.5 and 1.0"
        )
        ax.set_ylim(0, 100)

        for i, percentage in enumerate(high_diff_percentages):
            ax.text(
                i,
                percentage + 1,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(save_dir / "all_models_percentages.png")
        plt.close()

        # Save percentage histogram without saturation
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            x_labels,
            high_diff_percentages_without_saturation,
            edgecolor="black",
            color="lightblue",
        )

        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Percentage of data points (%)")
        ax.set_title(
            "Percentage of questions with CoT Accuracy - P(Correct) Differences between 0.5 and 1.0\nFiltered by P(Correct) ≤ 0.25"
        )
        ax.set_ylim(0, 100)

        for i, percentage in enumerate(high_diff_percentages_without_saturation):
            ax.text(
                i,
                percentage + 1,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(save_dir / "all_models_percentages_without_saturation.png")
        plt.close()

        # Add new violin plot for direct p_correct values
        fig, ax = plt.subplots(figsize=(12, 6))
        direct_values = []

        for model_id in model_ids:
            model_data = df[df["model_id"] == model_id]
            direct_data = model_data[model_data["mode"] == "direct"]["p_correct"]
            direct_values.append(direct_data)

        violin_parts = ax.violinplot(direct_values, showmedians=True)

        # Style the violin plot
        for pc in violin_parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.7)
        violin_parts["cmedians"].set_color("red")

        # Add median values above each violin
        for i, median in enumerate([d.median() for d in direct_values], 1):
            ax.text(
                i,
                median,
                f"{median:.2f}",
                horizontalalignment="center",
                verticalalignment="bottom",
                weight="bold",
            )

        ax.set_xticks(range(1, len(x_labels) + 1))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Direct P(Correct)")
        ax.set_title("Distribution of Direct P(Correct) by Model")

        plt.tight_layout()
        plt.savefig(save_dir / "all_models_direct_accuracy.png")
        plt.close()

        # Add new violin plot for CoT p_correct values
        fig, ax = plt.subplots(figsize=(12, 6))
        cot_values = []

        for model_id in model_ids:
            model_data = df[df["model_id"] == model_id]
            cot_data = model_data[model_data["mode"] == "cot"]["p_correct"]
            cot_values.append(cot_data)

        violin_parts = ax.violinplot(cot_values, showmedians=True)

        # Style the violin plot
        for pc in violin_parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.7)
        violin_parts["cmedians"].set_color("red")

        # Add median values above each violin
        for i, median in enumerate([d.median() for d in cot_values], 1):
            ax.text(
                i,
                median,
                f"{median:.2f}",
                horizontalalignment="center",
                verticalalignment="bottom",
                weight="bold",
            )

        ax.set_xticks(range(1, len(x_labels) + 1))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("CoT Accuracy")
        ax.set_title("Distribution of CoT Accuracy by Model")

        plt.tight_layout()
        plt.savefig(save_dir / "all_models_cot_accuracy.png")
        plt.close()

    else:
        model_data_pivot = df.pivot(
            index=["dataset_id", "qid"], columns="mode", values="p_correct"
        )

        differences = model_data_pivot["cot"] - model_data_pivot["direct"]

        # Save histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(differences, bins=30, edgecolor="black")
        ax.axvline(
            x=differences.median(),
            color="red",
            linestyle="--",
            label=f"Median: {differences.median():.3f}",
        )
        ax.set_xlabel("Difference in P(Correct) (CoT - Direct)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of Differences (CoT - Direct) - {model}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_dir / f"differences_{model}.png")
        plt.close()

        # Save scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation = model_data_pivot["cot"].corr(model_data_pivot["direct"])
        ax.scatter(model_data_pivot["direct"], model_data_pivot["cot"], alpha=0.5)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
        ax.set_xlabel("Direct P(Correct)")
        ax.set_ylabel("CoT accuracy")
        ax.set_title(
            f"Direct vs CoT Performance - {model}\nCorrelation: {correlation:.3f}"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_dir / f"scatter_{model}.png")
        plt.close()


def save_yes_no_histograms(
    data: pd.DataFrame,
    mode: str,
    model: str,
    save_dir: Path,
) -> None:
    """Save histograms showing p_correct distribution split by YES/NO answers."""
    if len(data) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    data = data[data["mode"] == mode].copy()
    data.rename(columns={"answer": "correct answer"}, inplace=True)

    # Create histogram
    sns.histplot(
        data=data,
        x="p_correct",
        hue="correct answer",
        bins=10,
        hue_order=["YES", "NO"],
        ax=ax,
    )

    # Add mean lines
    answer_colors = {"YES": "#2f8ccd", "NO": "#fb9a45"}
    for answer, color in answer_colors.items():
        mean_p_correct = data[data["correct answer"] == answer]["p_correct"].mean()
        ax.axvline(x=mean_p_correct, color=color, linestyle="--")

    title = f"{'CoT' if mode == 'cot' else 'Direct'} Responses - {model}"
    ax.set_title(title)
    ax.set_xlabel("Probability of Correct Answer")

    plt.tight_layout()
    plt.savefig(save_dir / f"yes_no_hist_{mode}_{model}.png")
    plt.close()


def save_model_biases(
    df: pd.DataFrame,
    save_dir: Path,
    biases: Literal["direct", "cot", "both"] = "both",
) -> None:
    """Save model bias plots showing accuracy differences between YES and NO questions."""
    # Bottom to top
    model_order = [
        "P",  # Phi
        "Q72",
        "Q32",
        "Q14",
        "Q7",
        "Q3",
        "Q1.5",
        "Q0.5",  # Qwens
        "G27",
        "G9",
        "G2",  # Gemmas
        "L70",
        "L8",
        "L3",
        "L1",  # Llamas
    ]

    model_labels = [MODELS_MAP[model_key].split("/")[-1] for model_key in model_order]

    results = []
    for model_key in model_order:
        model_id = MODELS_MAP[model_key]
        for mode in ["direct", "cot"]:
            model_data = df[(df["model_id"] == model_id) & (df["mode"] == mode)]
            yes_acc = model_data[model_data["answer"] == "YES"].p_correct.mean()
            no_acc = model_data[model_data["answer"] == "NO"].p_correct.mean()
            results.append(
                {"model": model_key, "mode": mode, "yes_acc": yes_acc, "no_acc": no_acc}
            )

    results_df = pd.DataFrame(results)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    y_positions = range(len(model_order))

    for idx, model in enumerate(model_order):
        model_data = results_df[results_df["model"] == model]

        if biases == "cot" or biases == "both":
            cot = model_data[model_data["mode"] == "cot"].iloc[0]
            cot_acc_diff = cot.no_acc - cot.yes_acc
            ax.arrow(
                cot.yes_acc,
                idx + (0.1 if biases == "both" else 0),
                cot_acc_diff,
                0,
                head_width=0.12,
                head_length=0.01,
                linewidth=2,
                color="#9b59b6",
                length_includes_head=True,
                label="CoT" if idx == 0 else None,
            )

        if biases == "direct" or biases == "both":
            direct = model_data[model_data["mode"] == "direct"].iloc[0]
            direct_acc_diff = direct.no_acc - direct.yes_acc
            ax.arrow(
                direct.yes_acc,
                idx - (0.1 if biases == "both" else 0),
                direct_acc_diff,
                0,
                head_width=0.12,
                head_length=0.01,
                linewidth=2,
                color="#2ecc71",
                length_includes_head=True,
                label="Direct" if idx == 0 else None,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_labels)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Biases: Average accuracy from YES to NO questions")
    ax.grid(True, alpha=0.3)
    if biases == "both":
        ax.legend()

    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / f"model_biases_{biases}.png")
    plt.close()


def save_yes_vs_no_bias_plot(df: pd.DataFrame, save_dir: Path) -> None:
    """Save plots showing the accuracy on YES vs NO questions for each model."""
    # Calculate accuracy for YES and NO questions for each model and mode
    results = []
    # Use sort_models to get consistent model ordering
    model_ids = sort_models(df["model_id"].unique())

    for model_id in model_ids:
        for mode in ["direct", "cot"]:
            model_data = df[(df["model_id"] == model_id) & (df["mode"] == mode)]
            yes_acc = model_data[model_data["answer"] == "YES"]["p_correct"].mean()
            no_acc = model_data[model_data["answer"] == "NO"]["p_correct"].mean()
            results.append(
                {
                    "Model": get_model_display_name(model_id),
                    "Mode": mode,
                    "YES accuracy": yes_acc,
                    "NO accuracy": no_acc,
                }
            )

    results_df = pd.DataFrame(results)

    # Create separate plots for each mode
    for mode in ["direct", "cot"]:
        mode_df = results_df[results_df["Mode"] == mode]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot dots for YES and NO accuracy
        x = range(len(mode_df))
        ax.scatter(
            x, mode_df["YES accuracy"], color="green", label="Answer is YES", s=100
        )
        ax.scatter(x, mode_df["NO accuracy"], color="red", label="Answer is NO", s=100)

        # Add a horizontal line at 0.5
        ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)

        # Customize the plot
        if mode == "direct":
            ax.set_title("P(Correct) on questions with answer YES (or NO)")
        else:
            ax.set_title("CoT Accuracy on questions with answer YES (or NO)")
        ax.set_ylabel("P(Model is correct)")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)

        # Legend in bottom right corner
        ax.legend(loc="lower right")

        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(mode_df["Model"], rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(save_dir / f"yes_vs_no_bias_{mode}.png")
        plt.close()


def save_yes_proportion_plot(df: pd.DataFrame, save_dir: Path) -> None:
    """Save a plot showing the proportion of YES answers for each model."""
    for mode in ["direct", "cot"]:
        results = []
        model_ids = sort_models(df["model_id"].unique())

        # reverse model_ids
        model_ids = model_ids[::-1]

        for model_id in model_ids:
            model_data = df[(df["model_id"] == model_id) & (df["mode"] == mode)]
            p_yes_mean = model_data["p_yes"].mean()
            results.append(
                {
                    "Model": get_model_display_name(model_id),
                    "P(YES)": p_yes_mean,
                    "Raw": p_yes_mean,
                }
            )

        results_df = pd.DataFrame(results)

        # Create figure with 2 rows, 1 column
        fig, (ax, cax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 6),
            gridspec_kw={"height_ratios": [20, 1]},
        )

        # Create colormap
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        colors = [(1, 0, 0), (1, 1, 1), (0, 0.7, 0)]  # Red -> White -> Green
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        norm = mcolors.Normalize(vmin=0, vmax=1)

        # Create horizontal bars with colors
        for idx, row in enumerate(results_df.itertuples()):
            width = row.Raw - 0.5
            left = 0.5 if width > 0 else row.Raw
            abs_width = abs(width)

            color = cmap(norm(row.Raw))
            ax.barh(idx, abs_width, left=left, color=color, height=0.5)

            # Add raw probability values
            if row.Raw > 0.05:
                text_x = row.Raw + 0.01 if row.Raw >= 0.5 else row.Raw - 0.01
                ha = "left" if row.Raw >= 0.5 else "right"
                ax.text(text_x, idx, f"{row.Raw:.2f}", va="center", ha=ha)

        # Add horizontal colorbar at the bottom
        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
            ticks=[0, 0.5, 1],
        )
        cax.set_xticklabels(["Biased\ntowards NO", "Neutral", "Biased\ntowards YES"])

        # Rest of plot customization
        if mode == "direct":
            title = "Average frequency of YES across all datasets"
        else:
            title = "Frequency of CoT YES answers across all datasets"

        ax.set_title(title)
        ax.set_yticks(range(len(results_df)))
        ax.set_yticklabels(results_df["Model"])

        ax.margins(y=0)

        ax.axvline(
            x=0.5, color="black", linestyle="--", alpha=0.5, label="Ground truth 0.5"
        )
        ax.legend()

        ax.set_ylabel("Model")
        ax.set_xlabel("Frequency of YES")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(save_dir / f"yes_proportion_{mode}.png", bbox_inches="tight")
        plt.close()


def save_all_plots(
    df: pd.DataFrame,
    save_dir: Path,
    model: str | None = None,
    prop_id: str | None = None,
    comparison: str | None = None,
    answer: str | None = None,
) -> None:
    """Save all plots for the given configuration."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter data
    filtered_df = df.copy()
    if model and model != "All":
        filtered_df = filtered_df[filtered_df["model_id"].str.endswith(model)]

    if len(filtered_df) == 0:
        print(f"Warning: No data found for model {model}")
        return

    if prop_id and prop_id != "All":
        filtered_df = filtered_df[filtered_df["prop_id"] == prop_id]

    if len(filtered_df) == 0:
        print(f"Warning: No data found for prop_id {prop_id}")
        return

    if comparison and comparison != "All":
        filtered_df = filtered_df[filtered_df["comparison"] == comparison]

    if len(filtered_df) == 0:
        print(f"Warning: No data found for comparison {comparison}")
        return

    if answer and answer != "All":
        filtered_df = filtered_df[filtered_df["answer"] == answer]

    if len(filtered_df) == 0:
        print(f"Warning: No data found for answer {answer}")
        return

    if model and model != "All":
        # Save probability distributions for specific model
        direct_data = filtered_df[filtered_df["mode"] == "direct"]
        cot_data = filtered_df[filtered_df["mode"] == "cot"]

        save_probability_distributions(direct_data, "direct", model, save_dir)
        save_probability_distributions(cot_data, "cot", model, save_dir)

        # Add yes/no histograms
        save_yes_no_histograms(filtered_df, "direct", model, save_dir)
        save_yes_no_histograms(filtered_df, "cot", model, save_dir)

    # Save comparison plots
    save_model_comparisons(filtered_df, save_dir, model if model != "All" else None)

    # Save model bias plots (only for all models)
    if model is None or model == "All":
        save_model_biases(filtered_df, save_dir, "direct")
        save_model_biases(filtered_df, save_dir, "cot")
        save_model_biases(filtered_df, save_dir, "both")
        save_yes_vs_no_bias_plot(filtered_df, save_dir)
        save_yes_proportion_plot(filtered_df, save_dir)


# Save plots for all models
save_all_plots(df, DATA_DIR / ".." / ".." / "plots" / "all_models")

for model in df["model_id"].unique():
    model_name = model.split("/")[-1]
    # Save plots for a specific model
    save_all_plots(df, DATA_DIR / ".." / ".." / "plots" / model_name, model=model_name)
