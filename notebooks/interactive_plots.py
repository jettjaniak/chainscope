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
available_models = sorted(
    (model_id.split("/")[-1] for model_id in df["model_id"].unique()),
    key=lambda x: (x.split("-")[0].lower(), get_param_count(x)),
)
print(available_models)
available_props = sorted(["All"] + list(df["prop_id"].unique()))
available_comparisons = sorted(["All"] + list(df["comparison"].unique()))
available_answers = sorted(["All"] + list(df["answer"].unique()))

# Create widgets
model_dropdown = widgets.Dropdown(
    options=available_models,
    description="Model:",
    style={"description_width": "initial"},
)

all_datasets_checkbox = widgets.Checkbox(
    value=True,
    description="All datasets",
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

# Create horizontal box for model and checkbox
model_row = widgets.HBox([model_dropdown, all_datasets_checkbox])

# Create horizontal box for filter dropdowns
filter_row = widgets.HBox([prop_dropdown, comparison_dropdown, answer_dropdown])


def plot_model_distributions(model, all_datasets, prop_id, comparison, answer):
    # Filter data for selected model
    model_data = df[df["model_id"].str.endswith(model)]

    # Apply additional filters if not using all datasets
    if not all_datasets:
        if prop_id != "All":
            model_data = model_data[model_data["prop_id"] == prop_id]
        if comparison != "All":
            model_data = model_data[model_data["comparison"] == comparison]
        if answer != "All":
            model_data = model_data[model_data["answer"] == answer]

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 12))
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
        ax2.set_title(
            f"Probability Distributions (CoT) - {model}\n({n_datasets_cot} datasets)",
            pad=15,
        )
        ax2.set_ylabel("Probability")
    else:
        ax2.text(0.5, 0.5, "Missing CoT Data", ha="center", va="center")
        ax2.set_title(f"Probability Distributions (CoT) - {model}", pad=15)

    # Add difference plot and scatter plot only if we have both types of data
    if has_direct and has_cot:
        model_data_pivot = model_data.pivot(
            index=["dataset_id", "qid"], columns="mode", values="p_correct"
        )
        differences = model_data_pivot["cot"] - model_data_pivot["direct"]
        ax3.boxplot(differences, positions=[1], tick_labels=["CoT - Direct"])
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


# Create interactive widget with conditional display
def update_visibility(change):
    filter_row.layout.display = "none" if change["new"] else "flex"


all_datasets_checkbox.observe(update_visibility, "value")

# Initialize filter row as hidden since checkbox starts as True
filter_row.layout.display = "none"

# Create the interactive widget
interactive_plot = widgets.interactive(
    plot_model_distributions,
    model=model_dropdown,
    all_datasets=all_datasets_checkbox,
    prop_id=prop_dropdown,
    comparison=comparison_dropdown,
    answer=answer_dropdown,
)

# Create a VBox container for all widgets
container = widgets.VBox(
    [
        model_row,
        filter_row,
        interactive_plot.children[-1],  # The output widget
    ]
)

# Display the container instead of the interactive_plot
display(container)
