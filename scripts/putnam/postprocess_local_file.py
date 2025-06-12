#!/usr/bin/env python3

from pathlib import Path

import click
import yaml
from beartype import beartype


@beartype
def process_yaml_file(file_path: Path) -> None:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Process each response
    for problem_id, responses in data.get("responses_by_qid", {}).items():
        for resp_id, resp_data in responses.items():
            # In the new format, resp_data is the string directly
            if isinstance(resp_data, str):
                # Find the first occurrence of "Human:" and truncate
                human_idx = resp_data.find("Human:", 10)
                if human_idx != -1:
                    responses[resp_id] = resp_data[:human_idx]
    # Write back to the same file
    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(input_yaml: str, verbose: bool):
    """Postprocess Putnam local rollouts by removing everything after 'Human:' in model answers."""
    file_path = Path(input_yaml)

    if verbose:
        click.echo(f"Processing file: {file_path}")

    process_yaml_file(file_path)

    if verbose:
        click.echo(f"Successfully processed {file_path}")


if __name__ == "__main__":
    main()
