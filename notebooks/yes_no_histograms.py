# %%
import re

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

from chainscope.typing import *

df = pd.read_pickle(DATA_DIR / "df.pkl")
filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
df = df[~df.prop_id.isin(filter_prop_ids)]
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
) -> pd.DataFrame:
    """Filter DataFrame based on selected criteria."""
    data = df.copy()
    if model != "All":
        data = data[data["model_id"].str.endswith(model)]
    if prop_id != "All":
        data = data[data["prop_id"] == prop_id]
    if comparison != "All":
        data = data[data["comparison"] == comparison]

    return data


full_model_id = sort_models(df["model_id"].unique())
short_model_ids = [m.split("/")[-1] for m in full_model_id]
# Get unique values for dropdowns
available_models = ["All"] + short_model_ids

available_props = sorted(["All"] + list(df["prop_id"].unique()))
available_comparisons = sorted(["All"] + list(df["comparison"].unique()))

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


def make_answer_p_correct_hist(df: pd.DataFrame, title: str, x_label: str):
    plt.figure(figsize=(5, 3))
    sns.histplot(
        df, x="p_correct", hue="correct answer", bins=10, hue_order=["YES", "NO"]
    )
    answer_colors = {"YES": "#2f8ccd", "NO": "#fb9a45"}
    for answer, color in answer_colors.items():
        mean_p_correct = df[df["correct answer"] == answer]["p_correct"].mean()
        plt.axvline(x=mean_p_correct, color=color, linestyle="--")
    plt.title(title)
    plt.xlabel(x_label)
    plt.show()


def make_interactive_plots(model, prop_id, comparison):
    model_data = filter_data(df, model, prop_id, comparison)
    model_data.rename(columns={"answer": "correct answer"}, inplace=True)
    dir_df = model_data[model_data["mode"] == "direct"]
    make_answer_p_correct_hist(
        dir_df, "direct response", "probability of correct answer"
    )
    cot_df = model_data[model_data["mode"] == "cot"]
    make_answer_p_correct_hist(cot_df, "CoT responses", "frequency of correct answer")


# Create the interactive widget with removed all_datasets parameter
interactive_plot = widgets.interactive(
    make_interactive_plots,
    model=model_dropdown,
    prop_id=prop_dropdown,
    comparison=comparison_dropdown,
)

# Create a single horizontal box containing all dropdowns
prop_comp_row = widgets.HBox([prop_dropdown, comparison_dropdown])
# Create a VBox container for all widgets
container = widgets.VBox(
    [
        model_dropdown,
        prop_comp_row,
        interactive_plot.children[-1],  # The output widget
    ]
)

# Display the container instead of the interactive_plot
display(container)

# %%
