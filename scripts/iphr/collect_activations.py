#!/usr/bin/env python3


import pickle
from pathlib import Path

import click
from tqdm.auto import tqdm

from chainscope.activations import collect_questions_acts
from chainscope.typing import DATA_DIR
from chainscope.utils import MODELS_MAP, load_model_and_tokenizer


def get_dataset_ids() -> list[str]:
    """Get all dataset IDs from the questions directory."""
    questions_dir = DATA_DIR / "questions"
    dataset_ids = []

    for yaml_path in questions_dir.rglob("*.yaml"):
        dataset_ids.append(yaml_path.stem)

    return sorted(dataset_ids)


def filter_dataset_ids(
    dataset_ids: list[str], prefix: str | None, suffix: str | None
) -> list[str]:
    """Filter dataset IDs by prefix and/or suffix."""
    if not prefix and not suffix:
        return dataset_ids

    filtered_ids = dataset_ids
    if prefix:
        filtered_ids = [dataset_id for dataset_id in filtered_ids if dataset_id.startswith(prefix)]
    if suffix:
        filtered_ids = [dataset_id for dataset_id in filtered_ids if dataset_id.endswith(suffix)]

    return filtered_ids


@click.command()
@click.option("-m", "--model-id", type=str, required=True)
@click.option(
    "-p",
    "--prop-id",
    type=str,
    help="Dataset ID prefix to filter by",
)
@click.option(
    "-s",
    "--dataset-suffix",
    type=str,
    help="Dataset ID suffix to filter by",
)
@click.option("-i", "--instr-id", type=str, required=True)
@click.option("-l", "--layers", type=str, required=True)
@click.option("-o", "--out-dir", type=str, required=True)
@click.option("--test", is_flag=True)
def main(
    model_id: str,
    prop_id: str | None,
    dataset_suffix: str | None,
    instr_id: str,
    layers: str,
    out_dir: Path,
    test: bool,
):
    model_id = MODELS_MAP.get(model_id, model_id)
    model, tokenizer = load_model_and_tokenizer(model_id)
    if layers == "all":
        layers_list = list(range(model.config.num_hidden_layers + 1))
    else:
        layers_list = [int(layer) for layer in layers.split(",")]

    # Get and filter dataset IDs
    dataset_ids = get_dataset_ids()
    dataset_ids = filter_dataset_ids(dataset_ids, prop_id, dataset_suffix)

    if not dataset_ids:
        raise click.BadParameter(
            f"No datasets found matching prefix: {prop_id} and suffix: {dataset_suffix}"
        )

    for dataset_id in tqdm(dataset_ids, desc="Processing datasets"):
        dataset_out_dir = Path(out_dir) / dataset_id / model_id.split("/")[-1]
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        res_by_qid = collect_questions_acts(
            model, tokenizer, dataset_id, instr_id, layers_list, test
        )

        for layer in tqdm(
            layers_list, desc=f"Saving activations for {dataset_id}", leave=False
        ):
            out_file = dataset_out_dir / f"L{layer:02d}.pkl"
            layer_res = {qid: res[layer] for qid, res in res_by_qid.items()}
            with open(out_file, "wb") as f:
                pickle.dump(layer_res, f)


if __name__ == "__main__":
    main()
