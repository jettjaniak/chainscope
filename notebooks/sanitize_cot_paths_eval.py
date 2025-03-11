#!/usr/bin/env python3

# %%
import logging
from copy import deepcopy

from chainscope.typing import CoTPathEval
from chainscope.utils import MODELS_MAP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%

model_aliases = ["GPT4O", "C3.5S"]
dataset_ids = ["gsm8k", "math", "mmlu"]  # Added all datasets

# %%

# Load all CoTPathEval objects
cot_path_evals: dict[tuple[str, str], CoTPathEval] = {}
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        try:
            cot_path_evals[(model_id, dataset_id)] = CoTPathEval.load(
                model_id, dataset_id
            )
            logger.info(f"Loaded evaluation for {model_alias} on {dataset_id}")
        except FileNotFoundError:
            logger.warning(
                f"No evaluation file found for {model_alias} on {dataset_id}"
            )
            continue

# %%


def sanitize_cot_path_eval(cot_path_eval: CoTPathEval) -> tuple[CoTPathEval, int]:
    """Sanitize a CoTPathEval object by cleaning dependent data that shouldn't exist."""
    sanitized = deepcopy(cot_path_eval)
    changes_made = 0

    for qid in list(sanitized.answer_correct_by_qid.keys()):
        for response_uuid in list(sanitized.answer_correct_by_qid[qid].keys()):
            # If answer is not CORRECT, remove all dependent evaluations
            if (
                sanitized.answer_correct_by_qid[qid][response_uuid].answer_correct
                != "CORRECT"
            ):
                if qid in sanitized.first_pass_eval_by_qid:
                    if response_uuid in sanitized.first_pass_eval_by_qid[qid]:
                        logger.info(
                            f"Removing first pass eval for qid={qid} response_uuid={response_uuid} due to non-CORRECT answer"
                        )
                        sanitized.first_pass_eval_by_qid[qid].pop(response_uuid, None)
                        changes_made += 1
                if qid in sanitized.second_pass_eval_by_qid:
                    if response_uuid in sanitized.second_pass_eval_by_qid[qid]:
                        logger.info(
                            f"Removing second pass eval for qid={qid} response_uuid={response_uuid} due to non-CORRECT answer"
                        )
                        sanitized.second_pass_eval_by_qid[qid].pop(response_uuid, None)
                        changes_made += 1
                if qid in sanitized.third_pass_eval_by_qid:
                    if response_uuid in sanitized.third_pass_eval_by_qid[qid]:
                        logger.info(
                            f"Removing third pass eval for qid={qid} response_uuid={response_uuid} due to non-CORRECT answer"
                        )
                        sanitized.third_pass_eval_by_qid[qid].pop(response_uuid, None)
                        changes_made += 1
                continue

            # For responses with CORRECT answers, check first pass evaluations
            if (
                qid in sanitized.first_pass_eval_by_qid
                and response_uuid in sanitized.first_pass_eval_by_qid[qid]
            ):
                first_pass_result = sanitized.first_pass_eval_by_qid[qid][response_uuid]

                # Check if there are any INCORRECT steps in first pass
                has_incorrect_steps = any(
                    step_status.node_status == "INCORRECT"
                    for step_status in first_pass_result.steps_status.values()
                )

                # If no steps are marked as INCORRECT in first pass, remove second and third pass
                if not has_incorrect_steps:
                    if qid in sanitized.second_pass_eval_by_qid:
                        if response_uuid in sanitized.second_pass_eval_by_qid[qid]:
                            logger.info(
                                f"Removing second pass eval for qid={qid} response_uuid={response_uuid} due to no INCORRECT steps in first pass"
                            )
                            sanitized.second_pass_eval_by_qid[qid].pop(
                                response_uuid, None
                            )
                            changes_made += 1
                    if qid in sanitized.third_pass_eval_by_qid:
                        if response_uuid in sanitized.third_pass_eval_by_qid[qid]:
                            logger.info(
                                f"Removing third pass eval for qid={qid} response_uuid={response_uuid} due to no INCORRECT steps in first pass"
                            )
                            sanitized.third_pass_eval_by_qid[qid].pop(
                                response_uuid, None
                            )
                            changes_made += 1

            # For steps that have second pass evaluations
            if (
                qid in sanitized.second_pass_eval_by_qid
                and response_uuid in sanitized.second_pass_eval_by_qid[qid]
            ):
                second_pass_result = sanitized.second_pass_eval_by_qid[qid][
                    response_uuid
                ]

                # Remove third pass evaluations for steps that are not UNFAITHFUL or don't have MINOR/MAJOR/CRITICAL severity
                for step_num, step_status in second_pass_result.steps_status.items():
                    if (
                        step_status.node_status != "UNFAITHFUL"
                        or step_status.node_severity
                        not in ["MINOR", "MAJOR", "CRITICAL"]
                    ):
                        if (
                            qid in sanitized.third_pass_eval_by_qid
                            and response_uuid in sanitized.third_pass_eval_by_qid[qid]
                            and step_num
                            in sanitized.third_pass_eval_by_qid[qid][
                                response_uuid
                            ].steps_status
                        ):
                            logger.info(
                                f"Removing third pass eval for qid={qid} response_uuid={response_uuid} step={step_num} due to non-UNFAITHFUL or non-significant second pass"
                            )
                            del sanitized.third_pass_eval_by_qid[qid][
                                response_uuid
                            ].steps_status[step_num]
                            changes_made += 1

                # Clean up empty dictionaries
                if (
                    qid in sanitized.third_pass_eval_by_qid
                    and response_uuid in sanitized.third_pass_eval_by_qid[qid]
                    and not sanitized.third_pass_eval_by_qid[qid][
                        response_uuid
                    ].steps_status
                ):
                    logger.info(
                        f"Removing empty third pass eval for qid={qid} response_uuid={response_uuid}"
                    )
                    del sanitized.third_pass_eval_by_qid[qid][response_uuid]
                    changes_made += 1

                if (
                    qid in sanitized.third_pass_eval_by_qid
                    and not sanitized.third_pass_eval_by_qid[qid]
                ):
                    logger.info(f"Removing empty third pass eval for qid={qid}")
                    del sanitized.third_pass_eval_by_qid[qid]
                    changes_made += 1

    return sanitized, changes_made


# %%

# Sanitize all CoTPathEval objects and save them
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        if (model_id, dataset_id) not in cot_path_evals:
            continue

        logger.info(f"\nProcessing {model_alias} on {dataset_id}...")
        original_eval = cot_path_evals[(model_id, dataset_id)]
        sanitized_eval, changes = sanitize_cot_path_eval(original_eval)

        if changes > 0:
            logger.info(
                f"Made {changes} changes to the evaluation results for {model_alias} on {dataset_id}"
            )
            # Save the sanitized results
            sanitized_eval.save()
            logger.info(f"Saved sanitized results for {model_alias} on {dataset_id}")
        else:
            logger.info(f"No changes needed for {model_alias} on {dataset_id}")

# %%
