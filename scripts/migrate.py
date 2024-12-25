#!/usr/bin/env python3

import yaml

from chainscope.typing import DATA_DIR


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


if __name__ == "__main__":
    migrate_questions()
