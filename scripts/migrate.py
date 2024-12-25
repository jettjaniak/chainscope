#!/usr/bin/env python3

import yaml

from chainscope.typing import DATA_DIR, DatasetParams


def migrate_questions() -> None:
    """Migrate question files to include DatasetParams structure."""
    questions_dir = DATA_DIR / "questions"

    for question_file in questions_dir.rglob("*.yaml"):
        # Read the current content
        with open(question_file, "r") as f:
            data = yaml.safe_load(f)

        # Extract uuid from filename
        uuid = question_file.stem.split("_")[-1]

        # Create new params structure
        params = {
            "answer": data.pop("answer"),
            "comparison": data.pop("comparison"),
            "max-comparisons": data.pop("max-comparisons"),
            "prop-id": data.pop("prop-id"),
            "uuid": uuid,
        }

        # Add params to the data
        data["params"] = params

        # Write back the updated content
        with open(question_file, "w") as f:
            yaml.dump(data, f)


def migrate_direct_eval() -> None:
    """Migrate direct_eval files to include DatasetParams structure."""
    direct_eval_dir = DATA_DIR / "direct_eval"

    for eval_file in direct_eval_dir.rglob("*.yaml"):
        # Extract info from path
        # path structure: direct_eval/{instr_id}/{pre_id}/{dataset_id}/{model_id}.yaml
        dataset_id = eval_file.parent.name

        # Read current content
        with open(eval_file, "r") as f:
            data = yaml.safe_load(f)

        # Create dataset params from dataset_id
        ds_params = DatasetParams.from_id(dataset_id)

        # Update the data structure
        data["ds-params"] = yaml.safe_load(ds_params.to_yaml())

        # Write back
        with open(eval_file, "w") as f:
            yaml.dump(data, f)


def migrate_cot_responses() -> None:
    """Migrate cot_responses files to include DatasetParams structure."""
    cot_responses_dir = DATA_DIR / "cot_responses"

    for response_file in cot_responses_dir.rglob("*.yaml"):
        # Read current content
        with open(response_file, "r") as f:
            data = yaml.safe_load(f)

        # Get dataset_id from data and remove it
        dataset_id = data.pop("dataset-id")

        # Create dataset params from dataset_id
        ds_params = DatasetParams.from_id(dataset_id)

        # Update the data structure
        data["ds-params"] = yaml.safe_load(ds_params.to_yaml())

        # Write back
        with open(response_file, "w") as f:
            yaml.dump(data, f)


if __name__ == "__main__":
    # migrate_questions()
    # migrate_direct_eval()
    migrate_cot_responses()
