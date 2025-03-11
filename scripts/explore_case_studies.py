#!/usr/bin/env python3

import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml

# Constants
CASE_STUDIES_DIR = Path("d/case_studies/analyzed")

# Categories for manual analysis
MANUAL_ANALYSIS_CATEGORIES = [
    "answer flipping",
    "fact manipulation",
    "different arguments",
    "bad arguments",
    "wrong comparison",
    "missing step",
    "other",
]


def save_model_data(model_file: str, data: Dict[str, Any]) -> None:
    """Save YAML data for a specific model."""
    with open(CASE_STUDIES_DIR / model_file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_model_data(model_file: str) -> Dict[str, Any]:
    """Load YAML data for a specific model."""
    with open(CASE_STUDIES_DIR / model_file, "r") as f:
        return yaml.safe_load(f)


def format_response_option(idx: int, uuid: str) -> str:
    """Format response option for dropdown with index and truncated UUID."""
    return f"{idx + 1} ({uuid[:8]})"


def format_value(val: float | int) -> str:
    """Format a value for display."""
    if val == int(val):
        return str(int(val))
    return f"{val:.4f}"


def is_pair_labeled(pair_data: Dict[str, Any]) -> bool:
    """Check if a pair has been manually analyzed."""
    if "manual_analysis" not in pair_data:
        return False

    manual_analysis = pair_data["manual_analysis"]
    text = manual_analysis.get("text", "").strip()
    categories = manual_analysis.get("categories", {})

    return bool(text) or any(categories.values())


def format_question_pair(qid: str, data: Dict[str, Any]) -> str:
    """Format question pair for dropdown with prop_id, comparison and truncated qid."""
    prop_id = data["prop_id"]
    comp = data["comparison"]
    formatted = f"{prop_id} {comp} ({qid[:6]})"
    if not is_pair_labeled(data):
        formatted += " [not labeled]"
    else:
        categories = []
        for cat in MANUAL_ANALYSIS_CATEGORIES:
            if data["manual_analysis"]["categories"][cat]:
                categories.append(cat)
        formatted += f" [{', '.join(categories)}]"
    return formatted


def display_response(response_data: Dict[str, Any]):
    """Display a response with proper formatting."""
    response = response_data["response"]
    result = response_data["result"]
    answer = response_data["final_answer"]
    equal_values = response_data["equal_values"]
    st.markdown(f"result: {result}, answer: {answer}, equal values: {equal_values}")
    st.markdown(response)
    with st.expander("Explanations"):
        exp_ev = response_data["explanation_equal_values"]
        st.write(f"equal values: {exp_ev}")
        exp_fa = response_data["explanation_final_answer"]
        st.write(f"final answer: {exp_fa}")


def main():
    st.title("Case Studies Explorer")

    def clear_all_states():
        """Helper function to clear all states when transitioning between questions."""
        # Clear checkbox states
        for cat in MANUAL_ANALYSIS_CATEGORIES:
            if f"checkbox_{cat}" in st.session_state:
                del st.session_state[f"checkbox_{cat}"]
        # Clear manual analysis text state
        if "manual_analysis_text" in st.session_state:
            del st.session_state["manual_analysis_text"]

    # Model selection
    model_files = [f.name for f in CASE_STUDIES_DIR.glob("*.yaml")]
    selected_model = st.selectbox("Select Model", model_files)

    # Load model data (using st.session_state to persist between reruns)
    if (
        "model_data" not in st.session_state
        or st.session_state.get("current_model") != selected_model
    ):
        st.session_state.model_data = load_model_data(selected_model)
        st.session_state.current_model = selected_model
        clear_all_states()  # Clear states when model changes

    # Question pair selection
    question_pairs = list(st.session_state.model_data.keys())
    question_pair_options = {
        format_question_pair(qid, st.session_state.model_data[qid]): qid
        for qid in question_pairs
    }

    # Store the formatted options list for index lookup
    formatted_options = sorted(list(question_pair_options.keys()))

    # Use stored index if available, otherwise default to 0
    if "selected_pair_index" not in st.session_state:
        st.session_state.selected_pair_index = 0

    def on_pair_select():
        # Update index only when user explicitly changes the selection
        new_index = formatted_options.index(st.session_state.question_pair_selectbox)
        if new_index != st.session_state.selected_pair_index:
            st.session_state.selected_pair_index = new_index
            clear_all_states()

    # Create a row with dropdown and next button
    selected_pair_display = st.selectbox(
        "Select Question Pair",
        formatted_options,
        index=st.session_state.selected_pair_index,
        key="question_pair_selectbox",
        on_change=on_pair_select,
    )
    if st.button("Next", key="next_button", use_container_width=True):
        # Move to next question pair
        current_idx = st.session_state.selected_pair_index
        next_idx = (current_idx + 1) % len(formatted_options)
        st.session_state["selected_pair_index"] = next_idx
        clear_all_states()
        st.rerun()

    selected_pair = (
        question_pair_options[selected_pair_display] if selected_pair_display else None
    )

    if selected_pair:
        pair_data = st.session_state.model_data[selected_pair]

        # Display comparison info and group p yes mean
        group_p_yes_mean = pair_data.get("group_p_yes_mean")
        bias_direction = "YES" if group_p_yes_mean > 0.5 else "NO"
        comp = pair_data.get("comparison")
        comp_str = "Greater Than" if comp == "gt" else "Less Than"
        st.markdown(
            f"freq. of YES in group: {format_value(group_p_yes_mean)} (bias toward **{bias_direction}**), "
            f"comparison: {comp_str}"
        )
        st.markdown(
            f"**{pair_data.get('x_name')}**: {format_value(pair_data.get('x_value'))}"
        )
        st.markdown(
            f"**{pair_data.get('y_name')}**: {format_value(pair_data.get('y_value'))}"
        )

        # Split and display analysis
        analysis = pair_data["analysis"]
        parts = analysis.split("Summary:", 1)
        details = parts[0].strip()
        if len(parts) > 1:
            summary = parts[1].strip()
        else:
            summary = "[failed to extract summary]"

        st.markdown(f"**Analysis Summary**:\n{summary}")
        with st.expander("Analysis Details"):
            st.text_area(
                "",  # Empty label since it's in an expander
                details,
                height=400,
            )

        # Q1 Section
        st.markdown(f"**Q1**: {pair_data.get('q1_str')}")
        correct_answer = pair_data.get("q1_correct_answer")
        p_correct = pair_data.get("q1_p_correct")
        p_correct_str = f"{p_correct:.2f}"
        st.markdown(f"correct answer: {correct_answer}, P(correct) = {p_correct_str}")

        # Q1 incorrect responses
        q1_incorrect = pair_data.get("q1_incorrect_resp", {})
        q1_options = [
            format_response_option(i, uuid)
            for i, uuid in enumerate(q1_incorrect.keys())
        ]
        selected_q1 = st.selectbox("Q1 Incorrect Responses", [""] + q1_options)

        if selected_q1:
            idx = int(selected_q1.split()[0]) - 1
            uuid = list(q1_incorrect.keys())[idx]
            display_response(q1_incorrect[uuid])

        # Q2 Section
        st.markdown(f"**Q2**: {pair_data.get('q2_str')}")
        correct_answer = pair_data.get("q2_correct_answer")
        p_correct = pair_data.get("q2_p_correct")
        p_correct_str = f"{p_correct:.2f}"
        st.markdown(f"correct answer: {correct_answer}, P(correct) = {p_correct_str}")

        # Q2 correct responses
        q2_correct = pair_data.get("q2_correct_resp", {})
        q2_options = [
            format_response_option(i, uuid) for i, uuid in enumerate(q2_correct.keys())
        ]
        selected_q2 = st.selectbox("Q2 Correct Responses", [""] + q2_options)

        if selected_q2:
            idx = int(selected_q2.split()[0]) - 1
            uuid = list(q2_correct.keys())[idx]
            display_response(q2_correct[uuid])

        # Manual Analysis Section
        st.markdown("## Manual Analysis")

        # Initialize manual analysis in pair_data if it doesn't exist
        if "manual_analysis" not in pair_data:
            pair_data["manual_analysis"] = {
                "text": "",
                "categories": {cat: False for cat in MANUAL_ANALYSIS_CATEGORIES},
            }

        # Text area for analysis
        if "manual_analysis_text" not in st.session_state:
            st.session_state.manual_analysis_text = pair_data["manual_analysis"].get(
                "text", ""
            )

        manual_text = st.text_area(
            "Analysis Notes",
            value=st.session_state.manual_analysis_text,
            height=200,
            key="manual_analysis_text",
        )

        # Categories as checkboxes
        st.markdown("### Categories")
        categories_state = pair_data["manual_analysis"].get("categories", {})
        # Initialize any missing categories
        for cat in MANUAL_ANALYSIS_CATEGORIES:
            if cat not in categories_state:
                categories_state[cat] = False

        selected_categories = {}
        cols = st.columns(3)
        for i, category in enumerate(MANUAL_ANALYSIS_CATEGORIES):
            col_idx = i % 3
            with cols[col_idx]:
                checkbox_key = f"checkbox_{category}"
                # Initialize session state if not already set
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = categories_state.get(
                        category, False
                    )
                # Use only the session state for the checkbox value
                selected_categories[category] = st.checkbox(
                    category,
                    key=checkbox_key,
                )

        # Check if there are changes
        current_state = {"text": manual_text, "categories": selected_categories}
        saved_state = pair_data["manual_analysis"]

        has_changes = current_state["text"] != saved_state.get(
            "text", ""
        ) or current_state["categories"] != saved_state.get("categories", {})

        # Save button
        if has_changes:
            if st.button("Save Changes"):
                pair_data["manual_analysis"] = {
                    "text": manual_text,
                    "categories": selected_categories,
                }
                save_model_data(selected_model, st.session_state.model_data)

                # Move to next question pair
                current_idx = formatted_options.index(selected_pair_display)
                next_idx = (current_idx + 1) % len(formatted_options)
                st.session_state["selected_pair_index"] = next_idx
                clear_all_states()

                st.success(
                    "Changes saved successfully! Moving to next question pair..."
                )
                time.sleep(0.7)
                st.rerun()


if __name__ == "__main__":
    main()
