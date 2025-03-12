#!/usr/bin/env python3

import random
from pathlib import Path

import streamlit as st

from chainscope.typing import *
from chainscope.utils import sort_models


def load_yaml(file_path: Path) -> AnswerFlippingEval:
    return AnswerFlippingEval.load(file_path)


def get_random_response_with_label(
    answer_flipping_label_by_qid: dict[str, dict[str, str]],
    target_label: str,
) -> tuple[str, str] | None:
    """Returns a random (qid, uuid) pair with the specified answer flipping label."""
    matching_pairs = []
    for qid, uuid_labels in answer_flipping_label_by_qid.items():
        for uuid, label in uuid_labels.items():
            if label == target_label:
                matching_pairs.append((qid, uuid))

    if not matching_pairs:
        return None

    return random.choice(matching_pairs)


def update_filters_for_response(
    qid: str,
    ds_params_by_qid: dict[str, DatasetParams],
    questions_by_qid: dict[str, Question],
) -> None:
    """Updates session state filters based on the selected response."""
    # Get dataset parameters for the selected qid
    ds_params = ds_params_by_qid[qid]
    prop_id = ds_params.id.split("_")[0]
    comparison = ds_params.id.split("_")[1]
    expected_answer = ds_params.id.split("_")[2]  # This will be YES or NO

    # Update session state
    st.session_state.filter_prop_id = prop_id
    st.session_state.filter_comparison = comparison
    st.session_state.filter_answer = expected_answer
    st.session_state.current_question = questions_by_qid[qid].q_str


def main():
    st.title("Answer Flipping Dataset Viewer")

    # Initialize session state variables
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_uuid" not in st.session_state:
        st.session_state.current_uuid = None
    if "filter_answer" not in st.session_state:
        st.session_state.filter_answer = "All"
    if "filter_expected_answer" not in st.session_state:
        st.session_state.filter_expected_answer = "All"
    if "filter_comparison" not in st.session_state:
        st.session_state.filter_comparison = "All"
    if "filter_prop_id" not in st.session_state:
        st.session_state.filter_prop_id = "All"
    if "filter_label" not in st.session_state:
        st.session_state.filter_label = "All"
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = None
    # Add new session state variables for cached data
    if "answer_flipping_label_by_qid" not in st.session_state:
        st.session_state.answer_flipping_label_by_qid = {}
    if "raw_analysis_by_qid" not in st.session_state:
        st.session_state.raw_analysis_by_qid = {}
    if "ds_params_by_qid" not in st.session_state:
        st.session_state.ds_params_by_qid = {}
    if "questions_by_qid" not in st.session_state:
        st.session_state.questions_by_qid = {}
    if "cot_responses_by_qid" not in st.session_state:
        st.session_state.cot_responses_by_qid = {}

    # Initialize analysis counter
    analysis_counter = 0
    MAX_ANALYSES = 30

    # Get all YAML files
    answer_flipping_dir = DATA_DIR / "answer_flipping_eval"
    yaml_files = {}  # model_name -> comparison -> expected_answer -> dataset_id -> path
    for path in answer_flipping_dir.rglob("*.yaml"):
        if path.is_file():
            # Directory structure: answer_flipping_eval/dataset_id/expected_answer/comparison/model.yaml
            model_name = path.stem.replace("__", "/")
            dataset_id = path.parent.name  # Parent of model.yaml
            prop_id = dataset_id.split("_")[0]
            comparison = path.parent.parent.name.split("_")[0]
            expected_answer = path.parent.parent.name.split("_")[1]

            if model_name not in yaml_files:
                yaml_files[model_name] = {}
            if comparison not in yaml_files[model_name]:
                yaml_files[model_name][comparison] = {}
            if expected_answer not in yaml_files[model_name][comparison]:
                yaml_files[model_name][comparison][expected_answer] = {}

            yaml_files[model_name][comparison][expected_answer][prop_id] = path

    model_names = list(yaml_files.keys())
    model_names = sort_models(model_names)

    # Model selection
    selected_model = st.selectbox("Select Model", model_names)

    # Add random selection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Random response WITHOUT answer flipping"):
            result = get_random_response_with_label(
                st.session_state.answer_flipping_label_by_qid, "NO"
            )
            if result:
                qid, uuid = result
                update_filters_for_response(
                    qid,
                    st.session_state.ds_params_by_qid,
                    st.session_state.questions_by_qid,
                )
                st.session_state.filter_label = "NO"
                st.rerun()
            else:
                st.warning("No responses found with NO answer flipping")

    with col2:
        if st.button("Random response WITH answer flipping"):
            result = get_random_response_with_label(
                st.session_state.answer_flipping_label_by_qid, "YES"
            )
            if result:
                qid, uuid = result
                update_filters_for_response(
                    qid,
                    st.session_state.ds_params_by_qid,
                    st.session_state.questions_by_qid,
                )
                st.session_state.filter_label = "YES"
                st.rerun()
            else:
                st.warning("No responses found with YES answer flipping")

    # Reset filters if model changes
    if selected_model != st.session_state.previous_model:
        st.session_state.filter_answer = "All"
        st.session_state.filter_expected_answer = "All"
        st.session_state.filter_comparison = "All"
        st.session_state.filter_prop_id = "All"
        st.session_state.filter_label = "All"
        st.session_state.current_question = None
        st.session_state.previous_model = selected_model
        # Clear the cached data
        st.session_state.answer_flipping_label_by_qid = {}
        st.session_state.raw_analysis_by_qid = {}
        st.session_state.ds_params_by_qid = {}
        st.session_state.questions_by_qid = {}
        st.session_state.cot_responses_by_qid = {}
        st.rerun()

    assert selected_model is not None

    # Get unique values for filters
    unique_answers = ["YES", "NO"]
    unique_comparisons = ["gt", "lt"]
    unique_labels = ["YES", "NO", "UNKNOWN", "NO_REASONING", "FAILED_EVAL"]

    # Get all property IDs from all files
    all_prop_ids = set()
    for model_name, comparisons in yaml_files[selected_model].items():
        for expected_answer, datasets in comparisons.items():
            for prop_id in datasets.keys():
                all_prop_ids.add(prop_id)
    unique_prop_ids = sorted(all_prop_ids)

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        new_filter_answer = st.selectbox(
            "Filter by Answer",
            ["All"] + unique_answers,
            index=0
            if st.session_state.filter_answer == "All"
            else unique_answers.index(st.session_state.filter_answer) + 1,
            key="answer_selector",
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
            key="comparison_selector",
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
            key="prop_id_selector",
        )
        if new_filter_prop_id != st.session_state.filter_prop_id:
            st.session_state.filter_prop_id = new_filter_prop_id
            st.session_state.current_question = None
            st.rerun()

    with col4:
        new_filter_label = st.selectbox(
            "Filter by Ans. Flipping Label",
            ["All"] + unique_labels,
            index=0
            if st.session_state.filter_label == "All"
            else unique_labels.index(st.session_state.filter_label) + 1,
            key="label_selector",
        )
        if new_filter_label != st.session_state.filter_label:
            st.session_state.filter_label = new_filter_label
            st.session_state.current_question = None
            st.rerun()

    # Load all data for the selected model only if model changes or data not loaded
    if (
        selected_model != st.session_state.previous_model
        or not st.session_state.answer_flipping_label_by_qid
    ):
        # Count total files to process
        total_files = sum(
            1  # Count each path as 1 file
            for comparison in yaml_files[selected_model].values()
            for expected_answer in comparison.values()
            for _ in expected_answer.values()
        )

        progress_text = f"Loading YAML files for model {selected_model}..."
        progress_bar = st.progress(0, text=progress_text)
        files_processed = 0

        st.session_state.answer_flipping_label_by_qid = {}
        st.session_state.raw_analysis_by_qid = {}
        st.session_state.ds_params_by_qid = {}
        st.session_state.questions_by_qid = {}
        st.session_state.cot_responses_by_qid = {}

        # Load data from all YAML files for this model
        for comparison in yaml_files[selected_model].keys():
            for expected_answer in yaml_files[selected_model][comparison].keys():
                for prop_id, path in yaml_files[selected_model][comparison][
                    expected_answer
                ].items():
                    data = load_yaml(path)
                    st.session_state.answer_flipping_label_by_qid.update(
                        data.label_by_qid
                    )
                    st.session_state.raw_analysis_by_qid.update(
                        data.raw_analysis_by_qid
                    )
                    for qid in data.label_by_qid.keys():
                        ds_params = data.ds_params
                        st.session_state.ds_params_by_qid[qid] = ds_params
                        qs_dataset = ds_params.load_qs_dataset()
                        st.session_state.questions_by_qid[qid] = (
                            qs_dataset.question_by_qid[qid]
                        )

                        # Load CoT responses
                        cot_responses = CotResponses.from_yaml_file(
                            DATA_DIR
                            / "cot_responses"
                            / data.instr_id
                            / data.sampling_params.id
                            / ds_params.pre_id
                            / ds_params.id
                            / f"{selected_model.replace('/', '__')}.yaml"
                        )
                        assert isinstance(cot_responses, CotResponses)
                        st.session_state.cot_responses_by_qid[qid] = (
                            cot_responses.responses_by_qid[qid]
                        )

                    files_processed += 1
                    progress_bar.progress(
                        min(100, int((files_processed / total_files) * 100)),
                        text=progress_text,
                    )

        # Remove the progress bar after loading is complete
        progress_bar.empty()

    # Filter questions for display only
    filtered_qids = set()
    for qid, question in st.session_state.questions_by_qid.items():
        ds_params = st.session_state.ds_params_by_qid[qid]
        prop_id = ds_params.id.split("_")[0]
        comparison = ds_params.id.split("_")[1]

        # Apply filters
        if (
            st.session_state.filter_prop_id == "All"
            or prop_id == st.session_state.filter_prop_id
        ) and (
            st.session_state.filter_comparison == "All"
            or comparison == st.session_state.filter_comparison
        ):
            filtered_qids.add(qid)

    # Use filtered_qids for display only
    unique_questions = list(filtered_qids)
    q_str_by_qid = {
        qid: st.session_state.questions_by_qid[qid].q_str for qid in unique_questions
    }
    q_id_by_q_str = {v: k for k, v in q_str_by_qid.items()}

    # Modified question selection with "All" option
    if st.session_state.current_question is None:
        st.session_state.current_question = "All"

    question_options = ["All"] + list(q_str_by_qid.values())
    selected_question = st.selectbox(
        "Select Question",
        question_options,
        question_options.index(st.session_state.current_question),
        key="question_selector",
    )
    st.session_state.current_question = selected_question

    # Handle display of questions
    if selected_question == "All":
        # Filter questions that have at least one matching label
        filtered_q_strs = []
        for q_str, qid in q_id_by_q_str.items():
            labels = st.session_state.answer_flipping_label_by_qid[qid].values()
            if st.session_state.filter_label == "All" or any(
                st.session_state.filter_label == label for label in labels
            ):
                filtered_q_strs.append(q_str)
    else:
        filtered_q_strs = [selected_question]

    if not filtered_q_strs:
        st.warning("No questions match the current filters.")
        return

    for q_str in filtered_q_strs:
        if analysis_counter >= MAX_ANALYSES:
            st.warning(
                f"Showing only first {MAX_ANALYSES} analyses. Please use filters to see more specific results."
            )
            break

        st.markdown("---")  # Separator between questions
        qid = q_id_by_q_str[q_str]
        question = st.session_state.questions_by_qid[qid]

        # Display question with ground truth values
        st.markdown(f"### Question: {q_str}")
        st.markdown(
            f"Ground truth: {question.x_name}={question.x_value}, {question.y_name}={question.y_value}"
        )

        # Filter analyses based on label
        filtered_items = []  # Will store (uuid, analysis, label) tuples
        for uuid, analysis in st.session_state.raw_analysis_by_qid[qid].items():
            label = st.session_state.answer_flipping_label_by_qid[qid][uuid]
            if (
                st.session_state.filter_label == "All"
                or label == st.session_state.filter_label
            ):
                filtered_items.append((uuid, analysis, label))

        if not filtered_items:
            continue

        # Display filtered analyses
        for response_uuid, analysis, analysis_label in filtered_items:
            if analysis_counter >= MAX_ANALYSES:
                break

            st.markdown(
                f"**Answer flipping analysis for response {response_uuid}:** -> {analysis_label}"
            )
            # Display the CoT response if available
            if (
                qid in st.session_state.cot_responses_by_qid
                and response_uuid in st.session_state.cot_responses_by_qid[qid]
            ):
                st.markdown(
                    f"```\n{st.session_state.cot_responses_by_qid[qid][response_uuid]}\n```"
                )
            else:
                st.markdown("No CoT response available")

            st.markdown(f"{analysis}")
            analysis_counter += 1


if __name__ == "__main__":
    main()
