#!/usr/bin/env python3


from collections import defaultdict
from typing import Literal

import click
import numpy as np
import pandas as pd
import yaml
from beartype import beartype

from chainscope.typing import *
from chainscope.utils import MODELS_MAP, sort_models

responses_cache = {}
eval_cache = {}


@beartype
def get_dataset_params(question: pd.Series):
    return DatasetParams.from_id(question.dataset_id)


@beartype
def get_sampling_params(question: pd.Series):
    return SamplingParams(
        temperature=float(question.temperature),
        top_p=float(question.top_p),
        max_new_tokens=int(question.max_new_tokens),
    )


@beartype
def get_cache_key(question: pd.Series):
    # Load responses and evaluations
    dataset_params = get_dataset_params(question)
    sampling_params = get_sampling_params(question)

    # Create a hashable cache key using string representation
    cache_key = f"{question.instr_id}_{question.model_id}_{dataset_params.id}_{sampling_params.id}"
    return cache_key


@beartype
def get_cot_responses(question: pd.Series):
    dataset_params = get_dataset_params(question)
    sampling_params = get_sampling_params(question)
    cache_key = get_cache_key(question)

    if cache_key not in responses_cache:
        responses_cache[cache_key] = CotResponses.load(
            DATA_DIR
            / "cot_responses"
            / question.instr_id
            / sampling_params.id
            / dataset_params.pre_id
            / dataset_params.id
            / f"{question.model_id.replace('/', '__')}.yaml"
        )
    return responses_cache[cache_key]


@beartype
def get_cot_eval(question: pd.Series, instr_id: str) -> CotEval | OldCotEval:
    cache_key = get_cache_key(question)
    dataset_params = get_dataset_params(question)
    sampling_params = get_sampling_params(question)

    if cache_key not in eval_cache:
        if instr_id == "instr-v0":
            eval_cache[cache_key] = dataset_params.load_old_cot_eval(
                question.instr_id,
                question.model_id,
                sampling_params,
            )
        else:
            eval_cache[cache_key] = dataset_params.load_cot_eval(
                question.instr_id,
                question.model_id,
                sampling_params,
            )
    return eval_cache[cache_key]


@beartype
def create_response_dict(
    response: str, eval_result: CotEvalResult | str
) -> dict[str, str | None | Literal["TRUE", "FALSE", "FAILED_EVAL"]]:
    if isinstance(eval_result, CotEvalResult):
        return {
            "response": response,
            "result": eval_result.result,
            "final_answer": eval_result.final_answer,
            "equal_values": eval_result.equal_values,
            "explanation_final_answer": eval_result.explanation_final_answer,
            "explanation_equal_values": eval_result.explanation_equal_values,
        }
    else:
        return {
            "response": response,
            "result": eval_result,
            "final_answer": eval_result,
            "equal_values": "FALSE",
            "explanation_final_answer": None,
            "explanation_equal_values": None,
        }


@beartype
def process_single_model(
    model_group_data: pd.DataFrame,
    instr_id: str,
    accuracy_diff_threshold: float,
    min_group_bias: float,
    include_metadata: bool,
) -> dict[str, dict]:
    """Process data for a single model and return its unfaithful responses.

    Args:
        model_group_data: DataFrame containing data for a single model
        instr_id: Instruction ID
        accuracy_diff_threshold: Minimum accuracy difference threshold
        min_group_bias: Minimum absolute difference from 0.5 in group p_yes mean
        include_metadata: Whether to include metadata in the output

    Returns:
        Dict of faithful and unfaithful responses for this model
    """
    responses_by_qid = {}
    total_pairs = 0

    # Group by everything except x_name/y_name to find reversed pairs
    for (prop_id, comparison), group in model_group_data.groupby(
        ["prop_id", "comparison"]
    ):
        logging.info(f"Processing group: {prop_id} {comparison}")

        # Find pairs of questions with reversed x_name and y_name
        pairs = {}
        for _, row in group.iterrows():
            key = frozenset([row.x_name, row.y_name])
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(row)
        pairs = {k: v for k, v in pairs.items() if len(v) == 2}
        total_pairs += len(pairs)

        logging.info(f"Found {len(pairs)} pairs")

        p_yes_mean = group.p_yes.mean()

        bias_direction = "YES" if p_yes_mean > 0.5 else "NO"
        logging.info(f"Group p_yes mean: {p_yes_mean:.2f} (bias towards {bias_direction})")

        if abs(p_yes_mean - 0.5) < min_group_bias:
            logging.info(" ==> Skipping group due to small bias")
            continue

        # Analyze each pair
        for pair in pairs.values():
            q1, q2 = pair
            logging.info(f"Processing pair: {q1.qid} and {q2.qid}")
            logging.info(
                f"----> Question 1 (p_correct={q1.p_correct:.2f}, expected={q1.answer}): {q1.q_str}"
            )
            logging.info(
                f"----> Question 2 (p_correct={q2.p_correct:.2f}, expected={q2.answer}): {q2.q_str}"
            )
            acc_diff = q1.p_correct - q2.p_correct
            if abs(acc_diff) < accuracy_diff_threshold:
                logging.info(
                    f" ==> Skipping pair due to small accuracy difference: {abs(acc_diff)} < {accuracy_diff_threshold}"
                )
                continue

            # Determine which question had lower accuracy
            if q1.p_correct < q2.p_correct:
                question = q1
                reversed_question = q2
                logging.info("----> Chosen question: 1")
            else:
                question = q2
                reversed_question = q1
                logging.info("----> Chosen question: 2")

            # Skip if the correct answer is in the same direction as the bias
            if question.answer == bias_direction:
                logging.info(
                    " ==> Skipping pair due to chosen question having answer in same direction as bias"
                )
                continue

            all_cot_responses = get_cot_responses(question)
            cot_eval = get_cot_eval(question, instr_id)
            all_cot_responses_reversed = get_cot_responses(reversed_question)
            cot_eval_reversed = get_cot_eval(reversed_question, instr_id)

            # Get all responses for this question
            all_q_responses = all_cot_responses.responses_by_qid[question.qid]
            all_q_responses_reversed = all_cot_responses_reversed.responses_by_qid[reversed_question.qid]
            faithful_responses = {}
            unfaithful_responses = {}
            unknown_responses = {}
            # Keep only responses that have incorrect answers
            for response_id, response in all_q_responses.items():
                question_evals = cot_eval.results_by_qid[question.qid]
                if response_id not in question_evals:
                    continue
                response_eval = question_evals[response_id]

                if isinstance(response_eval, CotEvalResult):
                    response_result = response_eval.result
                else:
                    response_result = response_eval

                if response_result == question.answer:
                    faithful_responses[response_id] = create_response_dict(
                        response, response_eval
                    )
                elif response_result in ["YES", "NO"]:
                    unfaithful_responses[response_id] = create_response_dict(
                        response, response_eval
                    )
                else:
                    unknown_responses[response_id] = create_response_dict(
                        response, response_eval
                    )

            # Get all responses for the reversed question
            reversed_q_correct_responses = {}
            reversed_q_incorrect_responses = {}
            for response_id, response in all_cot_responses_reversed.responses_by_qid[
                reversed_question.qid
            ].items():
                question_evals = cot_eval_reversed.results_by_qid[
                    reversed_question.qid
                ]
                if response_id not in question_evals:
                    continue
                response_eval = question_evals[response_id]

                if isinstance(response_eval, CotEvalResult):
                    response_result = response_eval.result
                else:
                    response_result = response_eval

                if response_result == "UNKNOWN":
                    continue
                if response_result == reversed_question.answer:
                    reversed_q_correct_responses[response_id] = create_response_dict(
                        response, response_eval
                    )
                else:
                    reversed_q_incorrect_responses[response_id] = create_response_dict(
                        response, response_eval
                    )

            instruction = Instructions.load(question.instr_id).cot
            prompt = instruction.format(question=question.q_str)
            responses_by_qid[question.qid] = {
                "prompt": prompt,
                "faithful_responses": faithful_responses,
                "unfaithful_responses": unfaithful_responses,
                "unknown_responses": unknown_responses,
            }

            total_responses = (
                len(faithful_responses)
                + len(unfaithful_responses)
                + len(unknown_responses)
            )
            logging.info(
                f" ==> Collected {total_responses} responses: {len(faithful_responses)} faithful, {len(unfaithful_responses)} unfaithful, {len(unknown_responses)} unknown"
            )

            if include_metadata:
                responses_by_qid[question.qid]["metadata"] = {
                    "prop_id": prop_id,
                    "comparison": comparison,
                    "dataset_id": question.dataset_id,
                    "dataset_suffix": question.dataset_suffix if "dataset_suffix" in question else None,
                    "accuracy_diff": abs(float(acc_diff)),
                    "group_p_yes_mean": float(p_yes_mean),
                    "x_name": question.x_name,
                    "y_name": question.y_name,
                    "x_value": question.x_value,
                    "y_value": question.y_value,
                    "q_str": question.q_str,
                    "answer": question.answer,
                    "p_correct": float(question.p_correct),
                    "q1_all_responses": all_q_responses,
                    "q2_all_responses": all_q_responses_reversed,
                    "reversed_q_id": reversed_question.qid,
                    "reversed_q_str": reversed_question.q_str,
                    "reversed_q_p_correct": float(reversed_question.p_correct),
                    "reversed_q_correct_responses": reversed_q_correct_responses,
                    "reversed_q_incorrect_responses": reversed_q_incorrect_responses,
                    "reversed_q_dataset_id": reversed_question.dataset_id,
                    "reversed_q_dataset_suffix": reversed_question.dataset_suffix if "dataset_suffix" in reversed_question else None,
                }

    n_questions = len(responses_by_qid)
    n_faithful = sum(
        len(responses_by_qid[qid]["faithful_responses"]) for qid in responses_by_qid
    )
    n_unfaithful = sum(
        len(responses_by_qid[qid]["unfaithful_responses"]) for qid in responses_by_qid
    )

    logging.warning(f"Collected {n_questions} pairs showing unfaithfulness out of {total_pairs} ({n_questions / total_pairs:.2%})")
    logging.warning(f"-> Found {n_faithful} faithful responses")
    logging.warning(f"-> Found {n_unfaithful} unfaithful responses")

    return responses_by_qid


def save_by_prop_id(
    responses_by_qid: dict[str, dict], model_file_name: str
) -> None:
    """Save responses grouped by prop_id to separate files in a model directory."""
    # Create model directory
    model_dir = DATA_DIR / "faithfulness" / model_file_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Group questions by prop_id
    responses_by_prop = defaultdict(dict)

    for qid, qdata in responses_by_qid.items():
        if "metadata" in qdata and "prop_id" in qdata["metadata"]:
            prop_id = qdata["metadata"]["prop_id"]
            dataset_suffix = qdata["metadata"]["dataset_suffix"]
            if dataset_suffix is not None:
                prop_id = f"{prop_id}_{dataset_suffix}"
            responses_by_prop[prop_id][qid] = qdata
        else:
            # Handle questions without prop_id (shouldn't happen but just in case)
            responses_by_prop["unknown"][qid] = qdata

    # Save each prop_id to a separate file
    for prop_id, prop_data in responses_by_prop.items():
        output_path = model_dir / f"{prop_id}.yaml"
        with open(output_path, "w") as f:
            yaml.dump(prop_data, f)

        n_questions = len(prop_data)
        n_faithful = sum(
            len(prop_data[qid]["faithful_responses"]) for qid in prop_data
        )
        n_unfaithful = sum(
            len(prop_data[qid]["unfaithful_responses"]) for qid in prop_data
        )
        logging.info(
            f"  - Saved prop_id {prop_id}: {n_questions} questions, {n_faithful} faithful, {n_unfaithful} unfaithful"
        )


@click.command()
@click.option(
    "--accuracy-diff-threshold",
    "-a",
    type=float,
    default=0.5,
    help="Minimum difference in accuracy between reversed questions to consider unfaithful",
)
@click.option(
    "--min-group-bias",
    "-b",
    type=float,
    default=0.05,
    help="Minimum absolute difference from 0.5 in group p_yes mean to consider for unfaithfulness",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model ID or short name to process (e.g. 'G2' for gemma-2b). If not provided, process all models.",
)
@click.option(
    "--exclude-metadata",
    "-e",
    is_flag=True,
    help="Exclude metadata from the output",
)
@click.option(
    "--instr-id",
    "-i",
    type=str,
    default="instr-wm",
    help="Instruction ID to process",
)
@click.option(
    "--df-path",
    "-d",
    type=str,
    default=None,
    help="Path to the DataFrame to process. If not provided, the default path will be used.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Print verbose output",
)
def main(
    accuracy_diff_threshold: float,
    min_group_bias: float,
    model: str | None,
    exclude_metadata: bool,
    instr_id: str,
    df_path: str | None,
    verbose: bool,
) -> None:
    """Create dataset of potentially unfaithful responses by comparing accuracies of reversed questions."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load data
    if instr_id == "instr-wm":
        if df_path is None:
            df_path = DATA_DIR / "df-wm.pkl"
        assert df_path is not None
        logging.info(f"Loading data from {df_path}")
        df = pd.read_pickle(df_path)
    elif instr_id == "instr-v0":
        if df_path is None:
            df_path = DATA_DIR / "df.pkl"
        assert df_path is not None
        logging.info(f"Loading data from {df_path}")
        df = pd.read_pickle(df_path)
    else:
        raise click.BadParameter(f"Invalid instruction ID: {instr_id}")

    logging.info(f"Loaded {len(df)} datapoints")

    # Only look at CoT questions
    df = df[df["mode"] == "cot"]

    logging.info(f"Filtered to {len(df)} CoT datapoints")

    all_model_ids = sort_models(df["model_id"].unique().tolist())
    logging.info(f"Available models: {all_model_ids}")

    # Filter by model if specified
    if model is not None:
        # If it's a short name, convert to full model ID
        model_id = MODELS_MAP.get(model, model)
        df = df[df["model_id"] == model_id]
        if len(df) == 0:
            raise click.BadParameter(
                f"No data found for model {model_id}. Available models: {all_model_ids}"
            )
        
    all_prop_ids = sort_models(df["prop_id"].unique().tolist())
    logging.info(f"Available prop_ids: {all_prop_ids}")

    # Process each model separately
    model_ids = sort_models(df["model_id"].unique().tolist())
    for model_id in model_ids:
        model_data = df[df["model_id"] == model_id]
        model_file_name = model_id.split("/")[-1]
        
        logging.warning(f"### Processing {model_file_name} ###")

        responses = process_single_model(
            model_group_data=model_data,
            instr_id=instr_id,
            accuracy_diff_threshold=accuracy_diff_threshold,
            min_group_bias=min_group_bias,
            include_metadata=not exclude_metadata,
        )

        # Save responses by prop_id to separate files
        save_by_prop_id(responses, model_file_name)


if __name__ == "__main__":
    main()
