#!/usr/bin/env python3

import random
from pathlib import Path
from typing import Any, TypedDict, cast

import streamlit as st
import yaml

from chainscope.typing import *
from chainscope.utils import sort_models


class Response(TypedDict):
    metadata: dict[str, Any]
    prompt: str
    faithful_responses: dict[str, str]
    unfaithful_responses: dict[str, str]


def load_yaml(file_path: Path) -> dict[str, Response]:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_available_models() -> list[str]:
    """Get a list of available model directories."""
    faithfulness_dir = DATA_DIR / "faithfulness"
    models_to_skip = [
        "claude-3.7-sonnet_10k",
        "claude-3.7-sonnet_32k",
        "gpt-4.5-preview",
        "Llama-3.1-8B-Instruct",
    ]
    model_dirs = [
        d.name
        for d in faithfulness_dir.iterdir()
        if d.is_dir() and d.name not in models_to_skip
    ]
    return sort_models(model_dirs)


def get_available_prop_ids(model: str) -> list[str]:
    """Get a list of available property IDs for a given model."""
    model_dir = DATA_DIR / "faithfulness" / model
    if not model_dir.exists() or not model_dir.is_dir():
        return []
    prop_files = [f.stem for f in model_dir.glob("*.yaml")]
    return sorted(prop_files)


def random_sample_question(
    data: dict[str, Response],
) -> tuple[str, str, dict[str, Any]]:
    """Randomly sample a question.
    Returns: (question_str, response_id, metadata)"""
    question_hash = random.choice(list(data.keys()))
    question_data = data[question_hash]
    return (
        question_data["metadata"]["q_str"],
        next(
            iter(
                question_data["faithful_responses"]
                | question_data["unfaithful_responses"]
            )
        ),
        question_data["metadata"],
    )


def main():
    st.set_page_config(
        page_title="Chain-of-Thought Reasoning In The Wild Is Not Always Faithful",
        page_icon=":imp:",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.markdown(
        "<h1 style='text-align: center'>Chain-of-Thought Reasoning In The Wild Is Not Always Faithful</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """In our [paper](https://github.com/jettjaniak/chainscope/blob/main/paper.pdf), we find unfaithful reasoning in both thinking and non-thinking frontier models, even without prompting. Here, we present the responses for the pairs of questions that were labeled as unfaithful.<hr>""",
        unsafe_allow_html=True,
    )

    # Initialize session state variables
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_response_id" not in st.session_state:
        st.session_state.current_response_id = None
    if "is_faithful" not in st.session_state:
        st.session_state.is_faithful = True
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = None
    if "current_comparison" not in st.session_state:
        st.session_state.current_comparison = None
    if "current_prop_id" not in st.session_state:
        st.session_state.current_prop_id = None
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = None
    if "filter_answer" not in st.session_state:
        st.session_state.filter_answer = "All"
    if "filter_comparison" not in st.session_state:
        st.session_state.filter_comparison = "All"
    if "previous_filters" not in st.session_state:
        st.session_state.previous_filters = ("All", "All")
    if "current_data" not in st.session_state:
        st.session_state.current_data = None

    # Get all available models
    model_names = get_available_models()

    # Model selection
    selected_model = st.selectbox("Select Model", model_names)

    # Reset prop_id selection and filters if model changes
    if selected_model != st.session_state.previous_model:
        st.session_state.current_prop_id = None
        st.session_state.filter_comparison = "All"
        st.session_state.current_question = None
        st.session_state.previous_model = selected_model
        st.session_state.current_data = None
        st.rerun()

    if selected_model:
        # Get available property IDs for the selected model
        prop_ids = get_available_prop_ids(selected_model)

        # Property ID selection
        selected_prop_id = st.selectbox(
            "Select Property ID",
            prop_ids,
            index=0
            if st.session_state.current_prop_id is None
            else max(0, prop_ids.index(st.session_state.current_prop_id)),
        )

        # Load data if prop_id changes or not loaded yet
        if (
            st.session_state.current_prop_id != selected_prop_id
            or st.session_state.current_data is None
        ):
            st.session_state.current_prop_id = selected_prop_id
            st.session_state.filter_comparison = "All"
            st.session_state.current_question = None

            # Load the data for the selected model and prop_id
            file_path = (
                DATA_DIR / "faithfulness" / selected_model / f"{selected_prop_id}.yaml"
            )
            st.session_state.current_data = load_yaml(file_path)
            st.rerun()

        data = st.session_state.current_data
        assert data is not None, "Data must be loaded before use"

        # Random sampling button
        if st.button("Sample Random Question"):
            q, rid, metadata = random_sample_question(data)
            st.session_state.current_question = q
            st.session_state.current_response_id = rid
            st.session_state.current_answer = metadata["answer"]
            st.session_state.current_comparison = metadata["comparison"]
            st.session_state.current_prop_id = metadata["prop_id"]
            # Set filter values to match the sampled response
            st.session_state.filter_comparison = metadata["comparison"]
            st.session_state.filter_prop_id = metadata["prop_id"]
            st.rerun()

        # Get unique values for filters
        all_metadata = [v["metadata"] for v in data.values()]
        unique_comparisons = sorted(set(m["comparison"] for m in all_metadata))

        # Filters
        new_filter_comparison = st.selectbox(
            "Filter by Comparison",
            ["All"] + unique_comparisons,
            index=0
            if st.session_state.filter_comparison == "All"
            else unique_comparisons.index(st.session_state.filter_comparison) + 1,
        )
        if new_filter_comparison != st.session_state.filter_comparison:
            st.session_state.filter_comparison = new_filter_comparison
            st.session_state.current_question = None
            st.rerun()

        # Filter the data
        filtered_data = {}
        for qid, qdata in data.items():
            metadata = qdata["metadata"]
            if (
                st.session_state.filter_comparison == "All"
                or metadata["comparison"] == st.session_state.filter_comparison
            ):
                filtered_data[qid] = qdata

        # Question selection
        questions = [v["metadata"]["q_str"] for v in filtered_data.values()]
        if questions:
            # If we have a randomly sampled question that's not in the filtered set,
            # add it to the options
            current_question = cast(str | None, st.session_state.current_question)
            if current_question is not None and current_question not in questions:
                questions = [current_question] + questions

            # Calculate index for selectbox
            index = 0
            if current_question is not None and current_question in questions:
                index = questions.index(current_question)

            new_selected_question = st.selectbox(
                "Select Question",
                questions,
                index=index,
            )
            if new_selected_question != st.session_state.current_question:
                st.session_state.current_question = new_selected_question
                st.rerun()

            # Find the data for selected question
            selected_data = next(
                v
                for v in data.values()
                if v["metadata"]["q_str"] == st.session_state.current_question
            )

            # Display comparison values
            metadata = selected_data["metadata"]
            x_value = (
                int(metadata["x_value"])
                if metadata["x_value"].is_integer()
                else metadata["x_value"]
            )
            y_value = (
                int(metadata["y_value"])
                if metadata["y_value"].is_integer()
                else metadata["y_value"]
            )
            bias_direction = "YES" if metadata["group_p_yes_mean"] > 0.5 else "NO"
            accuracy_diff = abs(
                metadata["p_correct"] - metadata["reversed_q_p_correct"]
            )
            st.markdown(
                f"Ground truth values:</br> - {metadata['x_name']}: {x_value}</br> - {metadata['y_name']}: {y_value}</br>Group bias {metadata['group_p_yes_mean']:.2f} (towards {bias_direction})</br>Accuracy difference between Q1 and Q2: {accuracy_diff:.2f}",
                unsafe_allow_html=True,
            )

            # Get all responses (both faithful and unfaithful)
            responses = (
                selected_data["faithful_responses"]
                | selected_data["unfaithful_responses"]
            )

            # Response ID selection
            response_ids = list(responses.keys())
            if st.session_state.current_response_id not in response_ids:
                st.session_state.current_response_id = (
                    response_ids[0] if response_ids else None
                )

            if response_ids:
                st.subheader(
                    f"Q1 (answer = {metadata['answer']}, accuracy = {metadata['p_correct']:.2f})"
                )
                # remove everything before the first ":"
                q_str_without_about = metadata["q_str"].split(":")[1].strip()
                st.markdown(f"""**Question**: {q_str_without_about}""")
                index = 0

                if (
                    st.session_state.current_response_id is not None
                    and st.session_state.current_response_id in response_ids
                ):
                    index = response_ids.index(st.session_state.current_response_id)

                # Add labels to response IDs to indicate if they are faithful or unfaithful
                labeled_response_ids = [
                    f"{rid} (correct)"
                    if rid in selected_data["faithful_responses"]
                    else f"{rid} (incorrect)"
                    for rid in response_ids
                ]

                new_selected_response_id = st.selectbox(
                    "Select Response ID",
                    options=labeled_response_ids,
                    index=index,
                )

                # Extract the original response ID by removing the label
                original_response_id = new_selected_response_id.split(" (")[0]
                if original_response_id != st.session_state.current_response_id:
                    st.session_state.current_response_id = original_response_id
                    st.rerun()

                # Display selected response
                assert st.session_state.current_response_id is not None
                response = responses[st.session_state.current_response_id]["response"]
                assert isinstance(response, str)
                st.text(response)

                reversed_q_answer = "YES" if metadata["answer"] == "NO" else "YES"
                reversed_q_str = metadata["reversed_q_str"].split(":")[1].strip()

                st.subheader(
                    f"Q2 (answer = {reversed_q_answer}, accuracy = {metadata['reversed_q_p_correct']:.2f})"
                )
                st.markdown(f"""**Question**: {reversed_q_str}""")
                reversed_correct = metadata["reversed_q_correct_responses"]
                reversed_incorrect = metadata["reversed_q_incorrect_responses"]

                # Combine and label responses
                reversed_responses = {
                    f"{rid} (correct)": resp for rid, resp in reversed_correct.items()
                } | {
                    f"{rid} (incorrect)": resp
                    for rid, resp in reversed_incorrect.items()
                }

                reversed_response_id = st.selectbox(
                    "Select response ID",
                    options=[""] + list(reversed_responses.keys()),
                    index=1,
                )

                if reversed_response_id:  # Only show if a response is selected
                    st.text(reversed_responses[reversed_response_id]["response"])

        else:
            st.warning("No questions match the selected filters.")


if __name__ == "__main__":
    main()
