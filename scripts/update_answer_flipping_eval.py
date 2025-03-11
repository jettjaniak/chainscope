#!/usr/bin/env python3
import logging

import click

from chainscope.answer_flipping_eval import evaluate_answer_flipping
from chainscope.api_utils.api_selector import APIPreferences
from chainscope.typing import *

already_updated_paths = []


@click.command()
@click.option(
    "--evaluator_model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Model used to evaluate answer flipping",
)
@click.option(
    "--anthropic",
    "--an",
    is_flag=True,
    default=True,
    help="Use Anthropic API for evaluation",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=2,
    help="Maximum retries for evaluating answer flipping",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(
    evaluator_model_id: str,
    anthropic: bool,
    max_retries: int,
    verbose: bool,
):
    """Update existing data in the project."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    skip_paths = [f"/home/ivan/src/chainscope/{p}" for p in already_updated_paths]

    # Get all YAML files in answer_flipping_eval directory
    answer_flipping_dir = DATA_DIR / "answer_flipping_eval"
    for path in answer_flipping_dir.rglob("*.yaml"):
        if not path.is_file():
            continue

        if str(path) in skip_paths:
            logging.warning(f"Skipping {path} because it has already been updated")
            continue

        logging.warning(f"Processing {path}")

        # Load the existing evaluation
        eval_data = AnswerFlippingEval.load(path)

        # Load the corresponding responses
        responses_path = (
            DATA_DIR
            / "cot_responses"
            / eval_data.instr_id
            / eval_data.sampling_params.id
            / eval_data.ds_params.pre_id
            / eval_data.ds_params.id
            / f"{eval_data.model_id.replace('/', '__')}.yaml"
        )
        responses = CotResponses.load(responses_path)

        # Filter responses that were previously labeled as YES
        filtered_responses = CotResponses(
            responses_by_qid={},
            model_id=responses.model_id,
            instr_id=responses.instr_id,
            ds_params=responses.ds_params,
            sampling_params=responses.sampling_params,
        )

        for qid, uuid_labels in eval_data.label_by_qid.items():
            filtered_responses.responses_by_qid[qid] = {}
            for uuid, label in uuid_labels.items():
                if label == "YES":
                    filtered_responses.responses_by_qid[qid][uuid] = (
                        responses.responses_by_qid[qid][uuid]
                    )

        # Skip if no YES labels found
        if not any(filtered_responses.responses_by_qid.values()):
            logging.info(f"No YES labels found in {path}, skipping")
            continue

        api_preferences = APIPreferences.from_args(
            open_router=False,
            open_ai=False,
            anthropic=anthropic,
        )

        # Re-run evaluation on filtered responses
        new_eval = evaluate_answer_flipping(
            responses=filtered_responses,
            evaluator_model_id=evaluator_model_id,
            max_retries=max_retries,
            api_preferences=api_preferences,
        )

        # Update original evaluation with new results
        for qid, uuid_labels in new_eval.label_by_qid.items():
            for uuid, label in uuid_labels.items():
                eval_data.label_by_qid[qid][uuid] = label
                eval_data.raw_analysis_by_qid[qid][uuid] = new_eval.raw_analysis_by_qid[
                    qid
                ][uuid]

        # Save updated evaluation
        eval_data.save()
        logging.warning(f"Updated evaluation saved to {path}")


if __name__ == "__main__":
    main()
