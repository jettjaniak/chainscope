# %%
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from chainscope.typing import *
from chainscope.utils import (MODELS_MAP, get_model_display_name,
                              get_model_family, sort_models)

# df = pd.read_pickle(DATA_DIR / "df.pkl")
# filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
# df = df[~df.prop_id.isin(filter_prop_ids)]

df_path = DATA_DIR / "df-wm-non-ambiguous-hard-2.pkl"
df = pd.read_pickle(df_path)

# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate


n_pairs = 4892

# %%

faithfulness_yamls_cache = {}


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
    plt.savefig(save_dir / f"prob_dist_{mode}_{model}.pdf")
    plt.close()


def save_accuracy_histograms(df: pd.DataFrame, save_dir: Path) -> None:
    """Save bar plots showing the distribution of correct responses for all models."""
    model_ids = sort_models(df["model_id"].unique().tolist())

    for mode in ["direct", "cot"]:
        accuracies = []
        labels = []
        colors = []  # Add colors list

        # add ground truth
        model_data = df[(df["model_id"] == model_ids[0]) & (df["mode"] == mode)]
        if len(model_data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        total_correct = len(model_data)
        accuracies.append(100)
        labels.append("Ground truth")
        colors.append("#ff9999")

        for model_id in model_ids:
            model_data = df[(df["model_id"] == model_id) & (df["mode"] == mode)]
            pct_correct = model_data["p_correct"].sum() / total_correct * 100
            accuracies.append(pct_correct)
            labels.append(get_model_display_name(model_id))
            colors.append("lightblue")

        # Create bar plot with colors
        bars = ax.bar(labels, accuracies, alpha=0.7, color=colors)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("Models")
        ax.set_ylabel(
            "Percentage of total P(Correct)"
            if mode == "direct"
            else "Percentage of correct CoT responses"
        )
        title = (
            "Percentage of total P(Correct) per model"
            if mode == "direct"
            else "Percentage of correct CoT responses per model"
        )
        ax.set_title(title)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(save_dir / f"accuracy_histogram_{mode}.pdf", bbox_inches="tight")
        plt.close()


def save_yes_no_histograms(
    data: pd.DataFrame,
    mode: str,
    model: str,
    save_dir: Path,
) -> None:
    """Save histograms showing p_correct distribution split by YES/NO answers."""
    data = data[data["mode"] == mode].copy()

    if len(data) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
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
    plt.savefig(save_dir / f"yes_no_hist_{mode}_{model}.pdf")
    plt.close()


def save_model_biases(
    df: pd.DataFrame,
    save_dir: Path,
    biases: Literal["direct", "cot", "both"] = "both",
) -> None:
    """Save model bias plots showing accuracy differences between YES and NO questions."""
    model_order = sort_models(list(MODELS_MAP.keys()))

    # Remove Q0.5
    model_order = [model_id for model_id in model_order if "0.5B" not in model_id]

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
    plt.savefig(save_dir / f"model_biases_{biases}.pdf")
    plt.close()


def save_yes_vs_no_bias_plot(df: pd.DataFrame, save_dir: Path) -> None:
    """Save plots showing the accuracy on YES vs NO questions for each model."""
    # Calculate accuracy for YES and NO questions for each model and mode
    results = []
    # Use sort_models to get consistent model ordering
    model_ids = sort_models(df["model_id"].unique().tolist())

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
        plt.savefig(save_dir / f"yes_vs_no_bias_{mode}.pdf")
        plt.close()


def save_yes_proportion_plot(df: pd.DataFrame, save_dir: Path) -> None:
    """Save a plot showing the proportion of YES answers for each model."""
    for mode in ["direct", "cot"]:
        results = []
        model_ids = sort_models(df["model_id"].unique().tolist())

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
            title = "Average P(YES) across all datasets"
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
        plt.savefig(save_dir / f"yes_proportion_{mode}.pdf", bbox_inches="tight")
        plt.close()


def save_dual_diverging_barplots(df: pd.DataFrame, save_dir: Path) -> None:
    """Save dual diverging barplots showing frequency of YES responses for greater/less than comparisons."""
    # Aggregate the data with both mean and standard error
    plot_data = (
        df.groupby(["prop_id", "comparison"])
        .agg({"p_yes": ["mean", "count", "std"]})
        .reset_index()
    )

    # Flatten column names
    plot_data.columns = ["prop_id", "comparison", "mean", "count", "std"]
    # Calculate standard error
    plot_data["stderr"] = plot_data["std"] / np.sqrt(plot_data["count"])

    # Create figure with two subplots side by side - increase height
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7, 6), sharey=True, dpi=300
    )  # Changed height from 4 to 6
    model_name = df.model_id.unique()[0].split("/")[-1]

    if "claude-3-5" in model_name:
        fig.suptitle("Sonnet 3.5 v2")
    elif "claude-3.7-sonnet" == model_name:
        fig.suptitle("Sonnet 3.7 (no extended thinking)")
    else:
        fig.suptitle(model_name)

    # Process each subplot
    for ax, comp, title in zip([ax1, ax2], ["gt", "lt"], ["Greater Than", "Less Than"]):
        # Filter data
        comp_data = plot_data[plot_data["comparison"] == comp].copy()

        # Print data for this comparison
        print(f"\n{title} data for {model_name}:")
        print("Property | Mean P(YES) | Std Error | Sample Count")
        print("-" * 50)
        for _, row in comp_data.iterrows():
            print(
                f"{row['prop_id']:<15} | {row['mean']:.3f} | {row['stderr']:.3f} | {int(row['count'])}"
            )

        # Center values around 0.5
        centered_values = comp_data["mean"] - 0.5

        # Add horizontal grid lines
        ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.3)

        # Create horizontal bars with increased spacing
        y_pos = np.arange(len(comp_data)) * 1.5  # Multiply by 1.5 to increase spacing
        colors = ["red" if v < 0 else "green" for v in centered_values]

        ax.barh(y_pos, centered_values, align="center", color=colors, height=1)
        ax.errorbar(
            centered_values,
            y_pos,
            xerr=comp_data["stderr"],
            fmt="none",
            color="black",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(comp_data["prop_id"])
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        x_ticks = np.linspace(-0.5, 0.5, 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x + 0.5:.1f}" for x in x_ticks])
        ax.set_title(title)
        ax.set_xlabel("freq. of YES")

        # Make sure grid lines are behind the bars
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_dir / f"dual_diverging_{model_name}.pdf", dpi=300)
    plt.close()


def save_answer_flipping_pie_charts(df: pd.DataFrame, save_dir: Path) -> None:
    """Save pie charts showing the distribution of answer flipping labels for each model."""
    # Get all answer flipping eval files
    model_ids = sort_models(df["model_id"].unique().tolist())

    # Define a better color palette and label mapping
    colors = {
        "YES": "#2ecc71",  # Softer green
        "NO": "#3498db",  # Softer blue
        "UNKNOWN": "#95a5a6",  # Gray
        "NO_REASONING": "#e67e22",  # Orange
        "FAILED_EVAL": "#e74c3c",  # Red
    }

    label_mapping = {
        "YES": "Changed Answer",
        "NO": "Consistent Answer",
        "UNKNOWN": "Unclear",
        "NO_REASONING": "No Reasoning",
        "FAILED_EVAL": "Failed Evaluation",
    }

    for model_id in model_ids:
        # Find all answer flipping eval files for this model
        eval_files = list(
            DATA_DIR.glob(f"answer_flipping_eval/**/{model_id.replace('/', '__')}.yaml")
        )

        if not eval_files:
            print(f"No answer flipping eval files found for {model_id}")
            continue

        # Aggregate results across all files
        all_labels = []
        for eval_file in eval_files:
            eval_data = AnswerFlippingEval.load(eval_file)
            for qid_labels in eval_data.label_by_qid.values():
                all_labels.extend(qid_labels.values())

        if not all_labels:
            continue

        # Count occurrences of each label
        label_counts = Counter(all_labels)

        # Create pie chart with improved styling
        fig, ax = plt.subplots(figsize=(10, 7))

        # Calculate percentages and prepare data
        total = sum(label_counts.values())
        sizes = []
        labels = []
        chart_colors = []

        # Sort by percentage (descending)
        sorted_counts = sorted(
            [(k, v) for k, v in label_counts.items()], key=lambda x: x[1], reverse=True
        )

        for label, count in sorted_counts:
            percentage = (count / total) * 100
            if percentage >= 1.0:  # Only show labels for segments >= 1%
                sizes.append(count)
                labels.append(f"{label_mapping[label]}\n({percentage:.1f}%)")
                chart_colors.append(colors[label])

        # Create pie chart with improved styling
        wedges, texts = ax.pie(
            sizes,
            labels=labels,
            colors=chart_colors,
            startangle=90,
            wedgeprops=dict(width=0.5),  # Make it a donut chart
        )

        # Improve label styling
        plt.setp(texts, size=9)

        display_name = get_model_display_name(model_id)
        plt.title(
            f"Answer Divergence Analysis\n{display_name}  -- all responses in {len(eval_files)} datasets",
            pad=20,
            size=12,
        )

        plt.tight_layout()
        plt.savefig(
            save_dir / f"answer_flipping_pie_{display_name}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def save_answer_flipping_all_models(df: pd.DataFrame, save_dir: Path) -> None:
    """Save a horizontal bar plot showing the percentage of YES labels for answer flipping across all models."""
    # Get all answer flipping eval files
    model_ids = sort_models(df["model_id"].unique().tolist())

    results = []
    for model_id in model_ids:
        if "o1" in model_id:
            continue
        # Find all answer flipping eval files for this model
        eval_files = list(
            DATA_DIR.glob(f"answer_flipping_eval/**/{model_id.replace('/', '__')}.yaml")
        )

        if not eval_files:
            continue

        # Aggregate results across all files
        all_labels = []
        for eval_file in eval_files:
            eval_data = AnswerFlippingEval.load(eval_file)
            for qid_labels in eval_data.label_by_qid.values():
                all_labels.extend(qid_labels.values())

        if not all_labels:
            print(f"No answer flipping eval files found for {model_id}")
            continue

        # Calculate percentage of YES labels
        label_counts = Counter(all_labels)
        yes_percentage = (label_counts["YES"] / len(all_labels)) * 100

        results.append(
            {
                "model_id": model_id,
                "model_display": get_model_display_name(model_id),
                "model_vendor": get_model_family(model_id),
                "yes_percentage": yes_percentage,
            }
        )

    if not results:
        print("No answer flipping eval files found for any model")
        return

    # Create DataFrame and sort models
    plot_data = pd.DataFrame(results)
    plot_data = plot_data.sort_values(
        by=["model_vendor", "model_id"],
        key=lambda x: pd.Categorical(
            x, categories=sort_models(x.unique().tolist()), ordered=True
        ),
    )

    # Create plot
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.barplot(
        data=plot_data,
        y="model_display",
        x="yes_percentage",
        hue="model_vendor",
        errorbar="se",
    )

    # Add percentage labels to the end of each bar
    for i, row in enumerate(plot_data.itertuples()):
        ax.text(
            row.yes_percentage + 0.2,  # Slightly offset from bar end
            i,
            f"{row.yes_percentage:.1f}%",
            va="center",
        )

    plt.xlabel("percentage of answer changes (%)")
    plt.ylabel("Model")
    plt.legend().set_visible(False)
    plt.title("Percentage of Answer Flipping by Model")

    plt.tight_layout()
    plt.savefig(save_dir / "answer_flipping_all_models.pdf")
    plt.close()


# %%


def save_iphr_plot(df: pd.DataFrame, save_dir: Path) -> None:
    """Save a bar plot showing the percentage of unfaithful question pairs for each model."""
    # Define vendor colors
    vendor_colors = {
        "anthropic": "#d4a27f",  # Orange/red/rust color for Sonnet
        "deepseek": "#4d6bfe",  # Purple for DeepSeek
        "google": "#3c7af9",  # Sky Blue for Gemini
        "meta-llama": "#1E3D8C",  # Darker Blue for Llama
        "openai": "#00A67E",  # Green for GPT
        "qwen": "#6B4CF6",  # Black for Qwen
    }

    props_with_suffix = set()
    for row in df.itertuples():
        prop_id = str(row.prop_id)
        dataset_suffix = None
        if "dataset_suffix" in df.columns:
            dataset_suffix = str(row.dataset_suffix)
        if dataset_suffix:
            props_with_suffix.add(prop_id + "_" + dataset_suffix)
        else:
            props_with_suffix.add(prop_id)

    sorted_model_ids = [
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3.6-sonnet",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.7-sonnet_1k",
        "anthropic/claude-3.7-sonnet_64k",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-r1",
        "openai/gpt-4o-mini",
        "openai/gpt-4o-2024-08-06",
        "openai/chatgpt-4o-latest",
        "google/gemini-pro-1.5",
        "google/gemini-2.5-flash-preview",
        "google/gemini-2.5-pro-preview",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "qwen/qwq-32b",
    ]
    assert all(model_id in df["model_id"].unique().tolist() for model_id in sorted_model_ids), f"All models in sorted_model_ids must be in df. Got: {df['model_id'].unique().tolist()}"
    assert all(model_id in sorted_model_ids for model_id in df["model_id"].unique().tolist()), f"All models in df must be in sorted_model_ids. Got: {df['model_id'].unique().tolist()}"

    # Load faithfulness data for each model
    results = []
    for model_id in sorted_model_ids:
        # Skip models we don't want to include
        if "0.5B" in model_id:
            continue

        # How many prop_ids do we have in df for this model?
        num_prop_ids = len(df[df["model_id"] == model_id]["prop_id"].unique())
        if num_prop_ids <= 3:
            print(f"Skipping {model_id} because it has only {num_prop_ids} prop_ids")
            continue

        # Convert model ID to filename format
        model_dir_name = model_id.split("/")[-1]
        # faith_dir = DATA_DIR / "faithfulness" / model_dir_name

        # # Check if directory exists
        # if not faith_dir.exists() or not faith_dir.is_dir():
        #     print(
        #         f"Faithfulness directory not found for {model_id}. Expected at {faith_dir}"
        #     )

        # # Initialize merged data dictionary
        # merged_faith_data = {}

        # # Check if we've already cached this model's data
        # if model_id in faithfulness_yamls_cache:
        #     merged_faith_data = faithfulness_yamls_cache[model_id]
        # else:
        #     # Get all YAML files in the directory
        #     yaml_files = list(faith_dir.glob("*.yaml"))
        #     if not yaml_files:
        #         print(
        #             f"No faithfulness YAML files found in directory for {model_id}: {faith_dir}"
        #         )
        #     else:
        #         # Load and merge all YAML files
        #         print(
        #             f"Loading faithfulness data for {model_id} from {len(yaml_files)} files in {faith_dir}"
        #         )
        #         for yaml_file in yaml_files:
        #             yaml_prop_id = yaml_file.stem
        #             if yaml_prop_id not in props_with_suffix:
        #                 # print(
        #                 #     f"Skipping {yaml_file} because {yaml_prop_id} not in {props_with_suffix}"
        #                 # )
        #                 continue

        #             try:
        #                 with open(yaml_file) as f:
        #                     faith_data = yaml.safe_load(f)
        #                     if faith_data:
        #                         merged_faith_data.update(faith_data)
        #             except Exception as e:
        #                 print(f"Error loading {yaml_file}: {e}")

        #     # Cache the merged data
        #     faithfulness_yamls_cache[model_id] = merged_faith_data

        # # Count unfaithful pairs
        # unfaithful_count = len(merged_faith_data.keys())

        unfaithful_count_by_model = {
            "claude-3.6-sonnet": 22,
            "claude-3.7-sonnet": 90,
            "claude-3.7-sonnet_1k": 2,
            "claude-3.7-sonnet_64k": 12,
            "claude-3.5-haiku": 363,
            "deepseek-chat": 60,
            "deepseek-r1": 18,
            "gpt-4o-2024-08-06": 18,
            "gemini-2.5-flash-preview": 106,
            "gemini-2.5-pro-preview": 7,
            "gemini-pro-1.5": 320,
            "Llama-3.1-70B": 159,
            "Llama-3.3-70B-Instruct": 102,
            "gpt-4o-mini": 660,
            "chatgpt-4o-latest": 24,
            "gpt-4o-2024-08-06": 18,
            "qwq-32b": 220,
        }
        unfaithful_count = unfaithful_count_by_model[model_dir_name]

        # Calculate percentage
        percentage = float((unfaithful_count / n_pairs) * 100)

        # Correct special case for Sonnet 3.5
        if model_id == "claude-3-5-sonnet-20241022":
            model_id = "anthropic/claude-3.5-sonnet"

        # Get model vendor and name
        vendor = get_model_family(model_id)

        # Map model names consistently
        model_name = model_id.split("/")[-1]
        if "claude" in model_id.lower():
            if "sonnet" in model_id.lower():
                model_id = model_id.replace("3.6", "3.5")
                model_name = "Sonnet " + model_id.split("-")[1]
                if "_" in model_id:
                    model_name = model_name + f" ({model_id.split('_')[1]})"
                elif "3.5" in model_id:
                    model_name = model_name + " v2"
            elif "haiku" in model_id.lower():
                model_name = "Haiku " + model_id.split("-")[1]
            else:
                model_name = model_id
        elif "deepseek-r1" in model_id.lower():
            model_name = "DeepSeek R1"
        elif "deepseek-chat" in model_id.lower():
            model_name = "DeepSeek V3"
        elif "gemini" in model_id.lower():
            model_name = "Gemini "
            
            if "1.5" in model_id.lower():
                model_name = model_name + " 1.5"
            elif "2.5" in model_id.lower():
                model_name = model_name + " 2.5"

            if "pro" in model_id.lower():
                model_name = model_name + " Pro"
            elif "flash" in model_id.lower():
                model_name = model_name + " Flash"
            
        elif "llama-3.3-70b-instruct" in model_id.lower():
            model_name = "Llama 3.3 70B It"
        elif "gpt-4" in model_id.lower():
            if "latest" in model_id.lower():
                model_name = "ChatGPT-4o"
            elif "mini" in model_id.lower():
                model_name = "GPT-4o Mini"
            else:
                model_name = "GPT-4o Aug '24"
        elif "qwq" in model_id.lower():
            model_name = "Qwen 32B"

        results.append(
            {
                "model": model_id,
                "xtick": model_name,
                "vendor": vendor,
                "unfaithful_count": unfaithful_count,
                "percentage": percentage,
            }
        )

    if not results:
        print("No faithfulness data found")
        return

    # Create DataFrame and ensure numeric types
    plot_data = pd.DataFrame(results)
    plot_data["percentage"] = pd.to_numeric(plot_data["percentage"])

    # Use white background
    plt.style.use("seaborn-v0_8-white")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create empty lists to store legend handles
    legend_handles = []
    seen_vendors = set()

    separation = 0.01
    width = 0.05

    # Calculate x positions for bars
    x_positions = [i * width + i * separation for i in range(len(plot_data))]

    for i, row in enumerate(plot_data.itertuples()):
        color = vendor_colors[row.vendor]
        bar = ax.bar(
            x_positions[i],
            row.percentage,
            width,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Add to legend only if we haven't seen this vendor yet
        if row.vendor not in seen_vendors:
            # Map vendor names to display names
            vendor_display_names = {
                "anthropic": "Anthropic",
                "deepseek": "DeepSeek",
                "google": "Google",
                "meta-llama": "Meta",
                "openai": "OpenAI",
                "qwen": "Qwen",
            }
            legend_handles.append((bar, vendor_display_names[row.vendor]))
            seen_vendors.add(row.vendor)

    # Add legend
    ax.legend(
        [h[0] for h in legend_handles],
        [h[1] for h in legend_handles],
        loc="upper right",
        frameon=True,
        fontsize=16,
    )

    # Add percentage labels on top of bars
    def add_labels(position, value):
        ax.text(
            position,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    for i, row in enumerate(plot_data.itertuples()):
        add_labels(x_positions[i], row.percentage)

    # Set xticks at the bar positions
    plt.xticks(x_positions, [r["xtick"] for r in results], rotation=45, ha="right", fontsize=16)

    # Add small ticks at the center of bars
    ax.tick_params(axis="x", which="major", length=4, width=2)

    # Customize the plot
    if "ambiguous" in str(df_path):
        ylabel = "Unfaithful Pairs of Qs (%)"
        plt.ylabel(ylabel, fontsize=22, labelpad=10)
    else:
        ylabel = "Hard Qs Unfaithfulness (%)"
        plt.ylabel(ylabel, fontsize=24, labelpad=10)

    
    plt.xlabel("Model", fontsize=24, labelpad=10)
    plt.ylim(0, 15)  # Increased upper limit slightly to fit labels
    plt.yticks(np.arange(1, 15, 1), fontsize=16)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Show all spines (lines around the plot)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    fig.savefig(save_dir / "iphr_unfaithfulness.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# %%


def save_all_plots(
    df: pd.DataFrame,
    save_dir: Path,
    model: str = "All",
    prop_id: str | None = None,
    comparison: str | None = None,
    answer: str | None = None,
) -> None:
    """Save all plots for the given configuration."""
    save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots for {model}")

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

        # Save dual diverging barplots for CoT mode
        save_dual_diverging_barplots(cot_data, save_dir)

        # Add answer flipping pie chart for this specific model
        # save_answer_flipping_pie_charts(
        #     filtered_df[filtered_df["model_id"].str.endswith(model)], save_dir
        # )

    # Save model bias plots (only for all models)
    if model == "All":
        # save_model_biases(filtered_df, save_dir, "direct")
        # save_model_biases(filtered_df, save_dir, "cot")
        # save_model_biases(filtered_df, save_dir, "both")
        # save_yes_vs_no_bias_plot(filtered_df, save_dir)
        # save_yes_proportion_plot(filtered_df, save_dir)
        # save_accuracy_histograms(filtered_df, save_dir)
        # save_answer_flipping_all_models(filtered_df, save_dir)
        save_iphr_plot(filtered_df, save_dir)


def save_unfaithful_shortcuts_plot(save_dir: Path) -> None:
    """Save a bar plot showing the distribution of thinking vs non-thinking responses for each model."""
    # Define data
    data = [
        {
            "model": "Claude",
            "thinking": 8.8,
            "non_thinking": 27.5,
            "thinking_n": "(n=10)",
            "non_thinking_n": "(n=19)",
            "vendor": "anthropic",
            "accuracy": {"thinking": 114, "non_thinking": 69},
        },
        {
            "model": "DeepSeek",
            "thinking": 1.2,
            "non_thinking": 3.7,
            "thinking_n": "(n=2)",
            "non_thinking_n": "(n=3)",
            "vendor": "deepseek",
            "accuracy": {"thinking": 172, "non_thinking": 81},
        },
        {
            "model": "Qwen",
            "thinking": 2.4,
            "non_thinking": 9.6,
            "thinking_n": "(n=1)",
            "non_thinking_n": "(n=11)",
            "vendor": "qwen",
            "accuracy": {"thinking": 41, "non_thinking": 115},
        },
    ]

    # Define vendor colors
    vendor_colors = {
        "anthropic": "#d4a27f",
        "deepseek": "#2D87F3",
        "qwen": "#6B4CF6",
    }

    # Create DataFrame
    plot_data = pd.DataFrame(data)

    # Use white background
    plt.style.use("seaborn-v0_8-white")

    # Set font sizes
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        }
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set bar width and positions
    width = 0.35
    separator = 0.02
    x = np.arange(len(plot_data))

    # Create bars with solid fill for thinking
    thinking_bars = ax.bar(
        x - width / 2 - separator,
        plot_data["thinking"],
        width,
        label="Thinking",
        color=[vendor_colors[vendor] for vendor in plot_data["vendor"]],
        edgecolor="black",
        linewidth=2,
        alpha=1.0,
    )

    # Create bars with hatching for non-thinking
    hatch_pattern = "xx"
    non_thinking_bars = ax.bar(
        x + width / 2 + separator,
        plot_data["non_thinking"],
        width,
        label="Non-thinking",
        color=[vendor_colors[vendor] for vendor in plot_data["vendor"]],
        edgecolor=None,
        linewidth=1,
        alpha=0.6,
        zorder=0,
    )
    # Draw hatch
    ax.bar(
        x + width / 2 + separator,
        plot_data["non_thinking"],
        width,
        label="Non-thinking",
        color="none",
        edgecolor="black",
        linewidth=1,
        alpha=1,
        hatch=hatch_pattern,
        zorder=1,
    )
    # Draw edge
    ax.bar(
        x + width / 2 + separator,
        plot_data["non_thinking"],
        width,
        label="Non-thinking",
        color="none",
        edgecolor="black",
        linewidth=2,
        alpha=1,
        zorder=2,
    )

    # Add value labels on top of bars
    def add_value_label(bars, ns):
        for idx, rect in enumerate(bars):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{height}%\n{ns[idx]}",
                ha="center",
                va="bottom",
                fontsize=16,
            )

    add_value_label(thinking_bars, plot_data["thinking_n"])
    add_value_label(non_thinking_bars, plot_data["non_thinking_n"])

    # Customize the plot
    ax.set_ylabel("Unfaithfulness Rate (%)", fontsize=20, labelpad=10)
    plt.xlabel("Model", fontsize=24, labelpad=10)
    plt.ylim(0, 40)  # Increased upper limit slightly to fit labels
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["model"], rotation=0)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Show all spines
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Create custom legend handles with hatching patterns
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="Thinking model", alpha=0.8),
        Patch(
            facecolor="white",
            edgecolor="black",
            label="Non-thinking model with CoT",
            alpha=0.8,
            hatch=hatch_pattern,
            linewidth=1,
        ),
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    fig.savefig(save_dir / "unfaithful_shortcuts.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %%

def save_oversampling_plot(df: pd.DataFrame, save_dir: Path) -> None:
    # Data for unfaithfulness retention percentages
    models = [
        "Sonnet 3.5 v2",
        "Sonnet 3.7",
        "Sonnet 3.7 1k",
        "Sonnet 3.7 64k",
        "DeepSeek R1",
        "ChatGPT-4o",
        "GPT-4o Aug '24",
        "Gemini 2.5 Pro"
    ]

    # Define model to vendor mapping
    model_vendors = {
        "ChatGPT-4o": "openai",
        "GPT-4o Aug '24": "openai",
        "Sonnet 3.5 v2": "anthropic",
        "Sonnet 3.7": "anthropic",
        "Sonnet 3.7 1k": "anthropic",
        "Sonnet 3.7 64k": "anthropic",
        "DeepSeek R1": "deepseek",
        "Gemini 2.5 Pro": "google"
    }

    # Define vendor colors (from plots_for_writeup.py)
    vendor_colors = {
        "anthropic": "#d4a27f",  # Orange/red/rust color for Sonnet
        "deepseek": "#4d6bfe",  # Purple for DeepSeek
        "google": "#3c7af9",  # Sky Blue for Gemini
        "meta-llama": "#1E3D8C",  # Darker Blue for Llama
        "openai": "#00A67E",  # Green for GPT
        "qwen": "#6B4CF6",  # Black for Qwen
    }

    # Percentages of pairs of qs retained after oversampling for each model
    retention_percentages = [
        54.55,  # Sonnet 3.5 v2
        70.00,  # Sonnet 3.7
        100.00, # Sonnet 3.7 1k
        75.00,  # Sonnet 3.7 64k
        72.22,  # DeepSeek R1
        68.18,  # ChatGPT-4o
        72.22,  # GPT-4o Aug '24
        100.00  # Gemini 2.5 Pro
    ]

    # Totals after oversampling for each model
    new_totals = [
        12,  # Sonnet 3.5 v2
        63,  # Sonnet 3.7
        2,   # Sonnet 3.7 1k
        9,   # Sonnet 3.7 64k
        13,  # DeepSeek R1
        15,  # ChatGPT-4o
        13,  # GPT-4o Aug '24
        7    # Gemini 2.5 Pro
    ]

    # Calculate average (not shown in the plot but reported)
    average = np.mean(retention_percentages)  # 76.52%

    # Create the bar chart
    plt.style.use("seaborn-v0_8-white")
    plt.figure(figsize=(12, 6))

    # Create empty lists to store legend handles
    legend_handles = []
    seen_vendors = set()

    # Create bars with vendor-specific colors
    bars = []
    for i, model in enumerate(models):
        vendor = model_vendors[model]
        color = vendor_colors[vendor]
        bar = plt.bar(i, retention_percentages[i], color=color, edgecolor="black", linewidth=1, alpha=0.8)
        bars.append(bar)
        
        # Add to legend only if we haven't seen this vendor yet
        if vendor not in seen_vendors:
            # Map vendor names to display names
            vendor_display_names = {
                "anthropic": "Anthropic",
                "deepseek": "DeepSeek",
                "google": "Google",
                "meta-llama": "Meta",
                "openai": "OpenAI",
                "qwen": "Qwen",
            }
            legend_handles.append((bar, vendor_display_names[vendor]))
            seen_vendors.add(vendor)

    # Add labels and title
    plt.xlabel('Model', fontsize=18)
    plt.ylabel('Unfaithfulness Retention\nAfter Oversampling (%)', fontsize=18)

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = float(retention_percentages[i])
        plt.text(i, height + 1, f'{height:.2f}%\nn={new_totals[i]}', ha='center', va='bottom', fontsize=12)

    # Add horizontal line at average
    plt.axhline(y=float(average), color='red', linestyle='--', alpha=0.7)
    plt.text(-0.5, float(average)+2, f'Average: {average:.2f}%', color='red')

    # Add legend
    plt.legend(
        [h[0] for h in legend_handles],
        [h[1] for h in legend_handles],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        fontsize=16
    )

    # Rotate x-axis labels for better readability
    plt.xticks(range(len(models)), models, rotation=45, ha='right', fontsize=16)

    # y-limits
    plt.ylim(0, 115)
    plt.yticks(np.arange(0, 110, 10), fontsize=16)

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.gca().set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_dir / "oversampling_unfaithfulness_retention.pdf")

    # Show the plot
    plt.show()
    plt.close()


# %%

# Save plots for all models
save_all_plots(df, DATA_DIR / ".." / ".." / "plots" / "all_models")

for model in df["model_id"].unique():
    model_name = model.split("/")[-1]
    # Save plots for a specific model
    save_all_plots(df, DATA_DIR / ".." / ".." / "plots" / model_name, model=model_name)

save_unfaithful_shortcuts_plot(DATA_DIR / ".." / ".." / "plots" / "all_models")
save_oversampling_plot(df, DATA_DIR / ".." / ".." / "plots" / "all_models")

# %%
