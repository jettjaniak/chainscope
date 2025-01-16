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


def random_sample_response(
    data: dict[str, Response], faithful: bool
) -> tuple[str, str, str, dict[str, Any]]:
    """Randomly sample a question and response.
    Returns: (question_str, response_id, response_text, metadata)"""
    question_hash = random.choice(list(data.keys()))
    question_data = data[question_hash]
    responses = question_data[
        "faithful_responses" if faithful else "unfaithful_responses"
    ]
    response_id = random.choice(list(responses.keys()))
    return (
        question_data["metadata"]["q_str"],
        response_id,
        responses[response_id],
        question_data["metadata"],
    )


def main():
    st.title("Faithfulness dataset viewer")

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
    if "filter_prop_id" not in st.session_state:
        st.session_state.filter_prop_id = "All"
    if "previous_filters" not in st.session_state:
        st.session_state.previous_filters = ("All", "All", "All")

    # Get all YAML files
    faithfulness_dir = DATA_DIR / "faithfulness"
    yaml_files = list(faithfulness_dir.glob("*.yaml"))
    model_names = [f.stem for f in yaml_files]
    model_names = sort_models(model_names)  # Sort by family and parameter count

    # Model selection
    selected_model = st.selectbox("Select Model", model_names)

    # Reset filters if model changes
    if selected_model != st.session_state.previous_model:
        st.session_state.filter_answer = "All"
        st.session_state.filter_comparison = "All"
        st.session_state.filter_prop_id = "All"
        st.session_state.current_question = None
        st.session_state.previous_model = selected_model
        st.rerun()

    if selected_model:
        data = load_yaml(faithfulness_dir / f"{selected_model}.yaml")

        # Random sampling buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sample Random Faithful Response"):
                q, rid, _, metadata = random_sample_response(data, True)
                st.session_state.current_question = q
                st.session_state.current_response_id = rid
                st.session_state.is_faithful = True
                st.session_state.current_answer = metadata["answer"]
                st.session_state.current_comparison = metadata["comparison"]
                st.session_state.current_prop_id = metadata["prop_id"]
                # Set filter values to match the sampled response
                st.session_state.filter_answer = metadata["answer"]
                st.session_state.filter_comparison = metadata["comparison"]
                st.session_state.filter_prop_id = metadata["prop_id"]
                st.rerun()
        with col2:
            if st.button("Sample Random Unfaithful Response"):
                q, rid, _, metadata = random_sample_response(data, False)
                st.session_state.current_question = q
                st.session_state.current_response_id = rid
                st.session_state.is_faithful = False
                st.session_state.current_answer = metadata["answer"]
                st.session_state.current_comparison = metadata["comparison"]
                st.session_state.current_prop_id = metadata["prop_id"]
                # Set filter values to match the sampled response
                st.session_state.filter_answer = metadata["answer"]
                st.session_state.filter_comparison = metadata["comparison"]
                st.session_state.filter_prop_id = metadata["prop_id"]
                st.rerun()

        # Get unique values for filters
        all_metadata = [v["metadata"] for v in data.values()]
        unique_answers = sorted(set(m["answer"] for m in all_metadata))
        unique_comparisons = sorted(set(m["comparison"] for m in all_metadata))
        unique_prop_ids = sorted(set(m["prop_id"] for m in all_metadata))

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            new_filter_answer = st.selectbox(
                "Filter by Answer",
                ["All"] + unique_answers,
                index=0
                if st.session_state.filter_answer == "All"
                else unique_answers.index(st.session_state.filter_answer) + 1,
            )
            if new_filter_answer != st.session_state.filter_answer:
                st.session_state.filter_answer = new_filter_answer
                st.session_state.current_question = None
                st.rerun()

        with col2:
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

        with col3:
            new_filter_prop_id = st.selectbox(
                "Filter by Property ID",
                ["All"] + unique_prop_ids,
                index=0
                if st.session_state.filter_prop_id == "All"
                else unique_prop_ids.index(st.session_state.filter_prop_id) + 1,
            )
            if new_filter_prop_id != st.session_state.filter_prop_id:
                st.session_state.filter_prop_id = new_filter_prop_id
                st.session_state.current_question = None
                st.rerun()

        # Filter questions
        filtered_data = {}
        for k, v in data.items():
            metadata = v["metadata"]
            if (
                (
                    st.session_state.filter_answer == "All"
                    or metadata["answer"] == st.session_state.filter_answer
                )
                and (
                    st.session_state.filter_comparison == "All"
                    or metadata["comparison"] == st.session_state.filter_comparison
                )
                and (
                    st.session_state.filter_prop_id == "All"
                    or metadata["prop_id"] == st.session_state.filter_prop_id
                )
            ):
                filtered_data[k] = v

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
            st.markdown(
                f"{metadata['answer']} ({metadata['x_name']}: {x_value}, {metadata['y_name']}: {y_value}), accuracy {metadata['p_correct']}, group bias {metadata['group_p_yes_mean']:.2f} (towards {bias_direction})"
            )

            # Response type selection
            new_response_type = st.radio(
                "Response Type",
                ["Faithful", "Unfaithful"],
                index=0 if st.session_state.is_faithful else 1,
                horizontal=True,
            )
            if (new_response_type == "Faithful") != st.session_state.is_faithful:
                st.session_state.is_faithful = new_response_type == "Faithful"
                st.rerun()

            # Get responses of selected type
            responses = selected_data[
                "faithful_responses"
                if st.session_state.is_faithful
                else "unfaithful_responses"
            ]

            # Response ID selection
            response_ids = list(responses.keys())
            if st.session_state.current_response_id not in response_ids:
                st.session_state.current_response_id = (
                    response_ids[0] if response_ids else None
                )

            if response_ids:
                index = 0

                if (
                    st.session_state.current_response_id is not None
                    and st.session_state.current_response_id in response_ids
                ):
                    index = response_ids.index(st.session_state.current_response_id)
                new_selected_response_id = st.selectbox(
                    "Select Response ID",
                    options=response_ids,
                    index=index,
                )
                if new_selected_response_id != st.session_state.current_response_id:
                    st.session_state.current_response_id = new_selected_response_id
                    st.rerun()

                # Display selected response
                st.subheader("Response")
                assert st.session_state.current_response_id is not None
                response = responses[st.session_state.current_response_id]
                assert isinstance(response, str)
                st.text(response)

                st.subheader("Reversed Question insights")
                st.markdown(
                    f"""- **Reversed question**: {metadata['reversed_q_str']}
- **Model's accuracy on reversed question**: {metadata['reversed_q_p_correct']}"""
                )
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
                    "Select response for reversed question",
                    options=[""] + list(reversed_responses.keys()),
                    index=0,  # Default to no selection
                )

                if reversed_response_id:  # Only show if a response is selected
                    st.text(reversed_responses[reversed_response_id])

        else:
            st.warning("No questions match the selected filters.")


if __name__ == "__main__":
    main()
