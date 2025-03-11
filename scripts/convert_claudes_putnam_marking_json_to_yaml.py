#!/usr/bin/env python3

"""Convert Putnam evaluation JSON to YAML format for split_cots.py

NOTE: cot-investigation/scripts/eval_yaml_notebook.py lists the order in which scripts need be run.

Example usage:
python3 chainscope/scripts/convert_claudes_putnam_marking_json_to_yaml.py \
    /path/to/putnam_evals.json
"""

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import click
import yaml


@dataclass
class PutnamToSplitCotConverter:
    def convert_entries(self, entries: List[Dict]) -> Dict:
        """Convert JSON entries to the expected YAML format"""
        responses_by_qid = {}

        # Generate unique IDs for each response
        for i, entry in enumerate(entries):
            response_id = str(uuid.uuid4())[:8]
            if entry["is_correct"] == "YES":
                responses_by_qid[f"q{i}"] = {
                    response_id: entry[
                        "thinking"
                    ]  # Use thinking as the content to be split
                }

        # Create the full YAML structure
        yaml_data = {
            "ds-params": {
                "answer": "YES",  # Default values
                "comparison": "gt",
                "max-comparisons": 420,
                "prop-id": "IGNORE_DS_PARAMS_THIS_IS_NOT_A_COMPARISON_DATASET",
                "uuid": str(uuid.uuid4())[:8],
            },
            "instr-id": "instr-v0",
            "model-id": "GF2T",
            "responses-by-qid": responses_by_qid,
            # DO NOT SUBMIT(arthur): What do I put here???
            "sampling-params": {
                "max-tokens": 95_000,
                "temperature": 420.0,
                "top-p": 420.0,
                "max-new-tokens": 95_000,
            },
        }

        return yaml_data

    def convert_file(self, input_path: Path) -> Dict:
        """Convert entire JSON file to expected YAML format"""
        with open(input_path) as f:
            data = json.load(f)

        return self.convert_entries(data)


@click.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-force", is_flag=True, help="Overwrite existing file")
def main(input_json: str, verbose: bool, force: bool):
    """Convert Putnam evaluation JSON to YAML format"""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    converter = PutnamToSplitCotConverter()
    converted_data = converter.convert_file(Path(input_json))

    output_yaml = Path(input_json).with_suffix(".yaml")
    if output_yaml.exists() and not force:
        raise ValueError(f"Output file {output_yaml} already exists")
    with open(output_yaml, "w") as f:
        yaml.dump(converted_data, f, sort_keys=False, allow_unicode=True)

    logging.info(
        f"Converted {len(converted_data['responses-by-qid'])} entries to {output_yaml}"
    )


if __name__ == "__main__":
    main()
