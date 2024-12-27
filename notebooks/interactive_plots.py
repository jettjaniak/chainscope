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


# Get unique values for dropdowns
available_models = ["All"] + sorted(
    (model_id.split("/")[-1] for model_id in df["model_id"].unique()),
    key=lambda x: (x.split("-")[0].lower(), get_param_count(x)),
)

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


def plot_model_distributions(model, prop_id, comparison, answer):
    if model == "All":
        model_data = df.copy()
    else:
        model_data = df[df["model_id"].str.endswith(model)]

    # Apply filters (removed all_datasets condition)
    if prop_id != "All":
        model_data = model_data[model_data["prop_id"] == prop_id]
    if comparison != "All":
        model_data = model_data[model_data["comparison"] == comparison]
    if answer != "All":
        model_data = model_data[model_data["answer"] == answer]

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 12))
    if model == "All":
        gs = fig.add_gridspec(2, 2)
        ax4 = fig.add_subplot(gs[0, :])
        ax3 = fig.add_subplot(gs[1, :])
    else:
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

    # Check if we have any data for direct and cot modes
    direct_data = model_data[model_data["mode"] == "direct"]
    cot_data = model_data[model_data["mode"] == "cot"]
    has_direct = len(direct_data) > 0
    has_cot = len(cot_data) > 0
    has_data_for_comparison = (
        has_direct and has_cot and len(direct_data) == len(cot_data)
    )

    if model != "All":
        # Plot for direct mode
        if has_direct:
            n_datasets_direct = len(direct_data["dataset_id"].unique())
            positions = [1, 2, 3]
            data_direct = [
                direct_data["p_yes"],
                direct_data["p_no"],
                direct_data["p_correct"],
            ]
            ax1.boxplot(
                data_direct,
                positions=positions,
                tick_labels=["P(Yes)", "P(No)", "P(Correct)"],
            )
            # Add median values as text
            for i, median in enumerate([d.median() for d in data_direct]):
                ax1.text(
                    positions[i],
                    median,
                    f"{median:.2f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    weight="bold",
                )
            ax1.set_title(
                f"Probability Distributions (Direct) - {model}\n({n_datasets_direct} datasets)",
                pad=15,
            )
            ax1.set_ylabel("Probability")
        else:
            ax1.text(0.5, 0.5, "Missing Direct Data", ha="center", va="center")
            ax1.set_title(f"Probability Distributions (Direct) - {model}", pad=15)

        # Plot for cot mode
        if has_cot:
            n_datasets_cot = len(cot_data["dataset_id"].unique())
            positions = [1, 2, 3, 4]
            data_cot = [
                cot_data["p_yes"],
                cot_data["p_no"],
                cot_data["p_correct"],
                cot_data["unknown_rate"],
            ]
            ax2.boxplot(
                data_cot,
                positions=positions,
                tick_labels=["P(Yes)", "P(No)", "P(Correct)", "Unknown Rate"],
            )
            # Add median values as text
            for i, median in enumerate([d.median() for d in data_cot]):
                ax2.text(
                    positions[i],
                    median,
                    f"{median:.2f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    weight="bold",
                )
            ax2.set_title(
                f"Probability Distributions (CoT) - {model}\n({n_datasets_cot} datasets)",
                pad=15,
            )
            ax2.set_ylabel("Probability")
        else:
            ax2.text(0.5, 0.5, "Missing CoT Data", ha="center", va="center")
            ax2.set_title(f"Probability Distributions (CoT) - {model}", pad=15)

    # Add difference plot and scatter plot only if we have both types of data
    if has_data_for_comparison:
        model_data_pivot = model_data.pivot(
            index=["dataset_id", "qid", "model_id"], columns="mode", values="p_correct"
        )

        if model == "All":
            # Group differences by model
            differences = []
            x_labels = []
            for model_id in sorted(
                model_data_pivot.index.get_level_values("model_id").unique(),
                key=lambda x: (
                    x.split("/")[-1].split("-")[0].lower(),
                    get_param_count(x.split("/")[-1]),
                ),
            ):
                model_differences = model_data_pivot.loc[
                    model_data_pivot.index.get_level_values("model_id") == model_id
                ]
                differences.append(
                    model_differences["cot"] - model_differences["direct"]
                )
                x_labels.append(model_id.split("/")[-1])

            ax3.boxplot(differences, tick_labels=x_labels)
            ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax3.set_title("Distribution of Median Differences (CoT - Direct)")
            ax3.set_ylabel("Difference in Prob")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Calculate correlations for each model
            correlations = []
            model_names = []
            for model_id in sorted(
                model_data_pivot.index.get_level_values("model_id").unique(),
                key=lambda x: (
                    x.split("/")[-1].split("-")[0].lower(),
                    get_param_count(x.split("/")[-1]),
                ),
            ):
                model_data = model_data_pivot.loc[
                    model_data_pivot.index.get_level_values("model_id") == model_id
                ]
                corr = model_data["cot"].corr(model_data["direct"])
                if not pd.isna(corr):  # Only include valid correlations
                    correlations.append(corr)
                    model_names.append(model_id.split("/")[-1])

            # Plot correlations as bar chart instead of histogram
            x_pos = range(len(correlations))
            ax4.bar(x_pos, correlations, edgecolor="black")
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(model_names, rotation=45, ha="right")
            ax4.set_ylabel("Correlation Coefficient")
            ax4.set_title("CoT vs Direct P(Correct) Correlations by Model")
        else:
            # Single boxplot for selected model
            differences = model_data_pivot["cot"] - model_data_pivot["direct"]
            ax3.boxplot(
                differences,
                positions=[1],
                tick_labels=[model],
            )

            ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax3.set_title("Distribution of Median Differences (CoT - Direct)")
            ax3.set_ylabel("Difference in Prob")

            correlation = model_data_pivot["cot"].corr(model_data_pivot["direct"])
            ax4.scatter(model_data_pivot["direct"], model_data_pivot["cot"], alpha=0.5)
            ax4.plot([0, 1], [0, 1], "r--", alpha=0.5)  # diagonal line
            ax4.set_xlabel("P(Correct) - Direct")
            ax4.set_ylabel("P(Correct) - CoT")
            ax4.set_title(f"Direct vs CoT Performance\nCorrelation: {correlation:.3f}")
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, "Missing Data for Comparison", ha="center", va="center")
        ax3.set_title("Distribution of Median Differences (CoT - Direct)")

        ax4.text(0.5, 0.5, "Missing Data for Comparison", ha="center", va="center")
        ax4.set_title("Direct vs CoT Performance")

    # Ensure axes are cleared for missing data cases
    if not has_direct:
        ax1.set_xticks([])
        ax1.set_yticks([])
    if not has_cot:
        ax2.set_xticks([])
        ax2.set_yticks([])
    if not (has_direct and has_cot):
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.tight_layout()
    plt.show()


# Create the interactive widget with removed all_datasets parameter
interactive_plot = widgets.interactive(
    plot_model_distributions,
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
