#!/usr/bin/env python3
# %%

import ipywidgets as widgets
import pandas as pd
from IPython.display import HTML, display

from chainscope.typing import *

# Load the data
df = pd.read_pickle(DATA_DIR / "df.pkl")
df = df[df["mode"] == "cot"]
filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
df = df[~df.prop_id.isin(filter_prop_ids)]

# %%

# Create widgets
prop_dropdown = widgets.Dropdown(
    options=sorted(df["prop_id"].unique()),
    description="Property:",
    style={"description_width": "initial"},
)

name_pairs_dropdown = widgets.Dropdown(
    options=[],
    description="Name pair:",
    style={"description_width": "initial"},
)

comparison_dropdown = widgets.Dropdown(
    options=[],
    description="Comparison:",
    style={"description_width": "initial"},
)

answer_dropdown = widgets.Dropdown(
    options=[],
    description="Correct answer:",
    style={"description_width": "initial"},
)

model_dropdown = widgets.Dropdown(
    options=sorted(df["model_id"].unique()),
    description="Model:",
    style={"description_width": "initial"},
)

model_answer_dropdown = widgets.Dropdown(
    options=["All", "YES", "NO", "UNKNOWN"],
    description="Model answer:",
    style={"description_width": "initial"},
)


def update_name_pairs(*args):
    prop_id = prop_dropdown.value
    if not prop_id:
        return

    # Get all name pairs for this property
    prop_df = df[df["prop_id"] == prop_id]
    name_pairs = [(row["x_name"], row["y_name"]) for _, row in prop_df.iterrows()]
    # Sort each pair and remove duplicates
    unique_pairs = sorted(set(tuple(sorted(pair)) for pair in name_pairs))
    name_pairs_dropdown.options = [f"{x} vs {y}" for x, y in unique_pairs]


def update_comparison_answer(*args):
    name_pair = name_pairs_dropdown.value
    if not isinstance(name_pair, str):
        return

    x_name, y_name = name_pair.split(" vs ")
    prop_df = df[df["prop_id"] == prop_dropdown.value]

    # Get rows where these names are used (in either order)
    mask = ((prop_df["x_name"] == x_name) & (prop_df["y_name"] == y_name)) | (
        (prop_df["x_name"] == y_name) & (prop_df["y_name"] == x_name)
    )
    relevant_df = prop_df[mask]

    comparison_dropdown.options = sorted(relevant_df["comparison"].unique())
    answer_dropdown.options = sorted(relevant_df["answer"].unique())


def display_responses(
    prop_id, name_pair, comparison, correct_answer, model_id, model_answer
):
    if not all(
        [
            prop_id,
            name_pair,
            comparison,
            correct_answer,
            model_id,
            model_answer,
        ]
    ):
        return

    x_name, y_name = name_pair.split(" vs ")

    # Filter data
    mask = (
        (df["prop_id"] == prop_id)
        & (df["model_id"] == model_id)
        & (df["comparison"] == comparison)
        & (df["answer"] == correct_answer)
        & (
            ((df["x_name"] == x_name) & (df["y_name"] == y_name))
            | ((df["x_name"] == y_name) & (df["y_name"] == x_name))
        )
    )

    filtered_df = df[mask]

    if len(filtered_df) == 0:
        display(HTML("<p>No questions found for these criteria.</p>"))
        return

    # Display the question and its responses
    row = filtered_df.iloc[0]  # We expect only one question per combination
    display(HTML(f"<h3>Question: {row['q_str']}</h3>"))
    display(HTML(f"<p>x_name: {row['x_name']}, x_value: {row['x_value']}</p>"))
    display(HTML(f"<p>y_name: {row['y_name']}, y_value: {row['y_value']}</p>"))
    display(HTML(f"<p>p_correct: {row['p_correct']:.2f}</p>"))

    # Load responses from the YAML file
    dataset_params = DatasetParams(
        prop_id=row["prop_id"],
        comparison=row["comparison"],
        answer=row["answer"],
        max_comparisons=1,
        uuid=row["dataset_id"].split("_")[-1],
    )

    sampling_params = SamplingParams(
        temperature=float(row["temperature"]),
        top_p=float(row["top_p"]),
        max_new_tokens=int(row["max_new_tokens"]),
    )

    try:
        responses = CotResponses.load(
            DATA_DIR
            / "cot_responses"
            / row["instr_id"]
            / sampling_params.id
            / dataset_params.pre_id
            / dataset_params.id
            / f"{row['model_id'].replace('/', '__')}.yaml"
        )

        # Load evaluations
        cot_eval = dataset_params.load_cot_eval(
            row["instr_id"],
            row["model_id"],
            sampling_params,
        )

        # Display each response based on the response filter
        for i, (response_id, response) in enumerate(
            responses.responses_by_qid[row["qid"]].items(), 1
        ):
            this_answer = cot_eval.results_by_qid[row["qid"]][response_id]
            if model_answer != "All":
                if this_answer != model_answer:
                    continue

            display(HTML(f"<h4>Response {i} (evaluated as {this_answer}):</h4>"))
            display(HTML(f"<p>{response.replace('\n', '<br>')}</p>"))
            display(HTML("<hr>"))
    except Exception as e:
        display(HTML(f"<p>Error loading responses: {str(e)}</p>"))


# Set up observers
prop_dropdown.observe(update_name_pairs, names="value")
name_pairs_dropdown.observe(update_comparison_answer, names="value")

# Create output widget
output = widgets.Output()

# Create interactive widget with manual control
interactive_display = widgets.interactive(
    display_responses,
    prop_id=prop_dropdown,
    name_pair=name_pairs_dropdown,
    comparison=comparison_dropdown,
    correct_answer=answer_dropdown,
    model_id=model_dropdown,
    model_answer=model_answer_dropdown,
)

# Create layout
controls = widgets.VBox(
    [
        prop_dropdown,
        name_pairs_dropdown,
        comparison_dropdown,
        answer_dropdown,
        model_dropdown,
        model_answer_dropdown,
    ]
)

# Initialize the dropdowns
update_name_pairs()

# Display the widgets and output
display(controls)
display(interactive_display.children[-1])  # The output widget

# %%
