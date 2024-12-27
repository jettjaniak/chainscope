# %%
import re

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from chainscope.typing import *

df = pd.read_pickle(DATA_DIR / "df.pkl")
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate


# %%
def get_param_count(model_name: str) -> float:
    """Extract parameter count from model name in billions using regex.

    Examples:
        "gemma-2-9b-it" -> 9
        "Llama-3.2-3B-Instruct" -> 3
        "Qwen2.5-0.5B-Instruct" -> 0.5
    """
    name_lower = model_name.lower()
    # Match patterns like "0.5b", "3b", "70b", ignoring case and possible hyphen prefix
    match = re.search(r"[-]?(\d+\.?\d*)b", name_lower)

    if match:
        return float(match.group(1))
    return float("inf")  # For models where we can't determine size


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


def filter_data(
    df: pd.DataFrame,
    model: str,
    prop_id: str,
    comparison: str,
    answer: str,
) -> pd.DataFrame:
    """Filter DataFrame based on selected criteria."""
    data = df.copy()

    if model != "All":
        data = data[data["model_id"].str.endswith(model)]
    if prop_id != "All":
        data = data[data["prop_id"] == prop_id]
    if comparison != "All":
        data = data[data["comparison"] == comparison]
    if answer != "All":
        data = data[data["answer"] == answer]

    return data


def setup_boxplot(
    ax: plt.Axes,
    data: list[pd.Series],
    labels: list[str],
    title: str,
    n_datasets: int | None = None,
    y_label: str = "Probability",
) -> None:
    """Set up a boxplot with consistent formatting."""
    positions = list(range(1, len(data) + 1))
    ax.boxplot(data, positions=positions, tick_labels=labels)

    # Add median values as text
    for i, median in enumerate([d.median() for d in data]):
        ax.text(
            positions[i],
            median,
            f"{median:.2f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            weight="bold",
        )

    dataset_info = f"\n({n_datasets} datasets)" if n_datasets is not None else ""
    ax.set_title(f"{title}{dataset_info}", pad=15)
    ax.set_ylabel(y_label)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)


def clear_empty_axis(ax: plt.Axes, title: str) -> None:
    """Clear axis ticks for empty plots and set title."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "Missing Data", ha="center", va="center")
    ax.set_title(title, pad=15)


# Get unique values for dropdowns
available_models = ["All"] + sort_models(df["model_id"].unique())

available_props = sorted(["All"] + list(df["prop_id"].unique()))
available_comparisons = sorted(["All"] + list(df["comparison"].unique()))
available_answers = sorted(["All"] + list(df["answer"].unique()))

# Create widgets
model_dropdown = widgets.Dropdown(
    options=available_models,
    description="Model:",
    style={"description_width": "initial"},
)

prop_dropdown = widgets.Dropdown(
    options=available_props,
    value="All",
    description="Property:",
    style={"description_width": "initial"},
)

comparison_dropdown = widgets.Dropdown(
    options=available_comparisons,
    value="All",
    description="Comparison:",
    style={"description_width": "initial"},
)

answer_dropdown = widgets.Dropdown(
    options=available_answers,
    value="All",
    description="Answer:",
    style={"description_width": "initial"},
)

# Create a single horizontal box containing all dropdowns
all_dropdowns_row = widgets.HBox(
    [model_dropdown, prop_dropdown, comparison_dropdown, answer_dropdown]
)


def plot_probability_distributions(
    ax: plt.Axes,
    data: pd.DataFrame,
    mode: str,
    model: str,
) -> None:
    """Plot probability distributions for a given mode."""
    title = f"Probability Distributions ({mode.title()}) - {model}"
    if len(data) > 0:
        probabilities = [data["p_yes"], data["p_no"], data["p_correct"]]
        labels = ["P(Yes)", "P(No)", "P(Correct)"]

        if mode == "cot":
            probabilities.append(data["unknown_rate"])
            labels.append("Unknown Rate")

        setup_boxplot(
            ax,
            probabilities,
            labels,
            title,
            len(data["dataset_id"].unique()),
        )
    else:
        clear_empty_axis(ax, title)


def plot_model_comparison(
    ax3: plt.Axes,
    ax4: plt.Axes,
    model_data_pivot: pd.DataFrame,
    model: str,
) -> None:
    """Plot comparison between CoT and Direct modes."""
    if model == "All":
        plot_all_models_comparison(ax3, ax4, model_data_pivot)
    else:
        plot_single_model_comparison(ax3, ax4, model_data_pivot, model)


def plot_all_models_comparison(
    ax3: plt.Axes,
    ax4: plt.Axes,
    model_data_pivot: pd.DataFrame,
) -> None:
    """Plot comparison metrics for all models."""
    differences = []
    correlations = []
    x_labels = []
    model_ids = sort_models(
        model_data_pivot.index.get_level_values("model_id").unique()
    )

    for model_id in model_ids:
        model_data = model_data_pivot.loc[
            model_data_pivot.index.get_level_values("model_id") == model_id
        ]
        # Calculate differences
        differences.append(model_data["cot"] - model_data["direct"])
        x_labels.append(get_model_display_name(model_id))

        # Calculate correlation
        corr = model_data["cot"].corr(model_data["direct"])
        if not pd.isna(corr):
            correlations.append((get_model_display_name(model_id), corr))

    # Plot differences
    setup_boxplot(
        ax3,
        differences,
        x_labels,
        "Distribution of Median Differences (CoT - Direct)",
        y_label="Difference in Prob",
    )
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot correlations
    corr_values = [c[1] for c in correlations]
    corr_names = [c[0] for c in correlations]
    x_pos = range(len(corr_values))
    ax4.bar(x_pos, corr_values, edgecolor="black")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(corr_names, rotation=45, ha="right")
    ax4.set_ylabel("Correlation Coefficient")
    ax4.set_title("CoT vs Direct P(Correct) Correlations by Model")


def plot_single_model_comparison(
    ax3: plt.Axes,
    ax4: plt.Axes,
    model_data_pivot: pd.DataFrame,
    model: str,
) -> None:
    """Plot comparison metrics for a single model."""
    differences = model_data_pivot["cot"] - model_data_pivot["direct"]
    setup_boxplot(
        ax3,
        [differences],
        [model],
        "Distribution of Median Differences (CoT - Direct)",
        y_label="Difference in Prob",
    )

    correlation = model_data_pivot["cot"].corr(model_data_pivot["direct"])
    ax4.scatter(model_data_pivot["direct"], model_data_pivot["cot"], alpha=0.5)
    ax4.plot([0, 1], [0, 1], "r--", alpha=0.5)
    ax4.set_xlabel("P(Correct) - Direct")
    ax4.set_ylabel("P(Correct) - CoT")
    ax4.set_title(f"Direct vs CoT Performance\nCorrelation: {correlation:.3f}")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)


def make_interactive_plots(model, prop_id, comparison, answer):
    model_data = filter_data(df, model, prop_id, comparison, answer)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)

    if model == "All":
        ax4 = fig.add_subplot(gs[0, :])
        ax3 = fig.add_subplot(gs[1, :])
    else:
        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    direct_data = model_data[model_data["mode"] == "direct"]
    cot_data = model_data[model_data["mode"] == "cot"]
    has_data_for_comparison = (
        len(direct_data) > 0 and len(cot_data) > 0 and len(direct_data) == len(cot_data)
    )

    if model != "All":
        plot_probability_distributions(ax1, direct_data, "direct", model)
        plot_probability_distributions(ax2, cot_data, "cot", model)

    if has_data_for_comparison:
        model_data_pivot = model_data.pivot(
            index=["dataset_id", "qid", "model_id"], columns="mode", values="p_correct"
        )
        plot_model_comparison(ax3, ax4, model_data_pivot, model)
    else:
        clear_empty_axis(ax3, "Distribution of Median Differences (CoT - Direct)")
        clear_empty_axis(ax4, "Direct vs CoT Performance")

    plt.tight_layout()
    plt.show()


# Create the interactive widget with removed all_datasets parameter
interactive_plot = widgets.interactive(
    make_interactive_plots,
    model=model_dropdown,
    prop_id=prop_dropdown,
    comparison=comparison_dropdown,
    answer=answer_dropdown,
)

# Create a VBox container for all widgets
container = widgets.VBox(
    [
        all_dropdowns_row,
        interactive_plot.children[-1],  # The output widget
    ]
)

# Display the container instead of the interactive_plot
display(container)

# %%
