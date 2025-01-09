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

    # Get all YAML files
    faithfulness_dir = DATA_DIR / "faithfulness"
    yaml_files = list(faithfulness_dir.glob("*.yaml"))
    model_names = [f.stem for f in yaml_files]
    model_names = sort_models(model_names)  # Sort by family and parameter count

    # Model selection
    selected_model = st.selectbox("Select Model", model_names)

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
                st.rerun()

        # Get unique values for filters
        all_metadata = [v["metadata"] for v in data.values()]
        unique_answers = sorted(set(m["answer"] for m in all_metadata))
        unique_comparisons = sorted(set(m["comparison"] for m in all_metadata))
        unique_prop_ids = sorted(set(m["prop_id"] for m in all_metadata))

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_answer = st.selectbox(
                "Filter by Answer",
                ["All"] + unique_answers,
                index=unique_answers.index(st.session_state.current_answer) + 1
                if st.session_state.current_answer in unique_answers
                else 0,
            )
        with col2:
            filter_comparison = st.selectbox(
                "Filter by Comparison",
                ["All"] + unique_comparisons,
                index=unique_comparisons.index(st.session_state.current_comparison) + 1
                if st.session_state.current_comparison in unique_comparisons
                else 0,
            )
        with col3:
            filter_prop_id = st.selectbox(
                "Filter by Property ID",
                ["All"] + unique_prop_ids,
                index=unique_prop_ids.index(st.session_state.current_prop_id) + 1
                if st.session_state.current_prop_id in unique_prop_ids
                else 0,
            )

        # Filter questions
        filtered_data = {}
        for k, v in data.items():
            metadata = v["metadata"]
            if (
                (filter_answer == "All" or metadata["answer"] == filter_answer)
                and (
                    filter_comparison == "All"
                    or metadata["comparison"] == filter_comparison
                )
                and (filter_prop_id == "All" or metadata["prop_id"] == filter_prop_id)
            ):
                filtered_data[k] = v

        # Question selection
        questions = [v["metadata"]["q_str"] for v in filtered_data.values()]
        if questions:
            # If we have a randomly sampled question that's not in the filtered set,
            # add it to the options
            current_question = cast(str | None, st.session_state.current_question)
            if current_question and current_question not in questions:
                questions = [current_question] + questions

            selected_question = st.selectbox(
                "Select Question",
                questions,
                index=questions.index(current_question)
                if current_question in questions
                else 0,
            )
            st.session_state.current_question = selected_question

            # Find the data for selected question
            selected_data = next(
                v for v in data.values() if v["metadata"]["q_str"] == selected_question
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
            st.write(
                f"{metadata['x_name']}: {x_value}, {metadata['y_name']}: {y_value}"
            )

            # Response type selection
            response_type = st.radio(
                "Response Type",
                ["Faithful", "Unfaithful"],
                index=0 if st.session_state.is_faithful else 1,
                horizontal=True,
            )
            st.session_state.is_faithful = response_type == "Faithful"

            # Get responses of selected type
            responses = selected_data[
                "faithful_responses"
                if st.session_state.is_faithful
                else "unfaithful_responses"
            ]

            # Response ID selection
            response_ids = list(responses.keys())
            current_response_id = cast(str | None, st.session_state.current_response_id)
            if current_response_id not in response_ids:
                st.session_state.current_response_id = (
                    response_ids[0] if response_ids else None
                )
                current_response_id = st.session_state.current_response_id

            if response_ids:
                selected_response_id = st.selectbox(
                    "Select Response ID",
                    response_ids,
                    index=response_ids.index(current_response_id)
                    if current_response_id in response_ids
                    else 0,
                )
                st.session_state.current_response_id = selected_response_id

                # Display selected response
                st.subheader("Response")
                st.text(responses[selected_response_id])
        else:
            st.warning("No questions match the selected filters.")


if __name__ == "__main__":
    main()
