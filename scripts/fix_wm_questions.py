#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Literal

import click

from chainscope.typing import *


def fix_dataset_id(dataset_id: str) -> str | None:
    """Fix a dataset ID by swapping gt/lt and YES/NO if needed.

    Returns None if the dataset ID doesn't need fixing.
    """
    if not dataset_id.startswith("wm-us-"):
        return None
    if not any(x in dataset_id for x in ["popu", "dens"]):
        return None

    prop_id, comparison, answer, max_comparisons, uuid = dataset_id.split("_")
    if comparison == "gt":
        if answer == "YES":
            comparison, answer = "lt", "NO"
        else:  # NO
            comparison, answer = "lt", "YES"
    else:  # lt
        if answer == "YES":
            comparison, answer = "gt", "NO"
        else:  # NO
            comparison, answer = "gt", "YES"

    return f"{prop_id}_{comparison}_{answer}_{max_comparisons}_{uuid}"


def fix_file(file_path: Path, root_dir: Path) -> None:
    """Fix a single file by moving it and updating its contents."""
    # Load the file using appropriate class
    if "questions" in str(file_path):
        data = QsDataset.from_yaml_file(file_path)
        assert isinstance(data, QsDataset)
        ds_params = data.params
    else:  # responses file
        data = CotResponses.from_yaml_file(file_path)
        assert isinstance(data, CotResponses)
        ds_params = data.ds_params

    # Calculate old path components before updating ds_params
    old_pre_id = (
        f"{ds_params.comparison}_{ds_params.answer}_{ds_params.max_comparisons}"
    )
    old_dataset_id = (
        f"{ds_params.prop_id}_{ds_params.comparison}"
        f"_{ds_params.answer}_{ds_params.max_comparisons}_{ds_params.uuid}"
    )
    new_dataset_id = fix_dataset_id(old_dataset_id)
    if new_dataset_id is None:
        return

    # Update the ds_params
    prop_id, comp_str, ans_str, max_comparisons, uuid = new_dataset_id.split("_")
    # Type check the comparison and answer
    comparison: Literal["gt", "lt"] = "gt" if comp_str == "gt" else "lt"
    answer: Literal["YES", "NO"] = "YES" if ans_str == "YES" else "NO"

    # Create new ds_params
    new_ds_params = DatasetParams(
        prop_id=prop_id,
        comparison=comparison,
        answer=answer,
        max_comparisons=int(max_comparisons),
        uuid=uuid,
    )

    # Update the data object
    if isinstance(data, QsDataset):
        data.params = new_ds_params
    else:
        data.ds_params = new_ds_params

    # Calculate new path
    new_pre_id = f"{comparison}_{answer}_{max_comparisons}"

    rel_path = file_path.relative_to(root_dir)
    parts = list(rel_path.parts)
    # Replace the pre_id directory
    pre_id_idx = parts.index(old_pre_id)
    parts[pre_id_idx] = new_pre_id
    if "cot_responses" in str(file_path):
        assert old_dataset_id in parts
        dsid_idx = parts.index(old_dataset_id)
        parts[dsid_idx] = new_dataset_id
    # Replace the dataset ID in the filename
    parts[-1] = parts[-1].replace(old_dataset_id, new_dataset_id)
    new_path = root_dir / Path(*parts)

    # Create the directory if it doesn't exist
    new_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the updated file
    data.to_yaml_file(new_path)

    # Remove the old file if it's different from the new one
    if new_path != file_path:
        file_path.unlink()
        # Remove empty directories
        parent = file_path.parent
        while parent != root_dir:
            if not any(parent.iterdir()):
                parent.rmdir()
            parent = parent.parent


@click.command()
@click.option(
    "-d",
    "--directory",
    type=click.Choice(["questions", "cot_responses"]),
    required=True,
    help="Directory to fix (questions or cot_responses)",
)
@click.option(
    "-p",
    "--prop_id",
    type=str,
    required=True,
    help="Property ID to fix",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(directory: str, prop_id: str, verbose: bool):
    """Fix questions and responses files for wm-us-* properties."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    root_dir = DATA_DIR / directory

    # Find all yaml files
    file_paths = list(root_dir.rglob("*.yaml"))
    for file_path in file_paths:
        if prop_id not in str(file_path):
            continue
        if verbose:
            logging.info(f"Processing {file_path}")
        fix_file(file_path, root_dir)


if __name__ == "__main__":
    main()
