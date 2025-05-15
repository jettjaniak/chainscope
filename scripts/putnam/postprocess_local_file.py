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
            if "model_answer" in resp_data:
                # Find the first occurrence of "Human:" and truncate
                model_answer = resp_data["model_answer"]
                if isinstance(model_answer, list):
                    for i, text in enumerate(model_answer):
                        human_idx = text.find("Human:")
                        if human_idx != -1:
                            model_answer[i] = text[:human_idx]
                else:  # string case
                    human_idx = model_answer.find("Human:")
                    if human_idx != -1:
                        resp_data["model_answer"] = model_answer[:human_idx]

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
