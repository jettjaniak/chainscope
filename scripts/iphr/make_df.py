#!/usr/bin/env python3

import click
import pandas as pd
from tqdm.auto import tqdm

from chainscope.typing import *


def get_dataset_ids(instr_id: str) -> set[str]:
    """Get all unique dataset IDs present in both cot_eval and direct_eval directories."""
    # direct_dataset_ids = set()
    cot_dataset_ids = set()
    # direct_dir = DATA_DIR / "direct_eval"
    cot_dir = DATA_DIR / "cot_eval" / instr_id
    # # instr_id, pre_id, dataset_id
    # for ds_dir in direct_dir.rglob("*/*/*"):
    #     if ds_dir.is_dir():
    #         direct_dataset_ids.add(ds_dir.name)
    # instr_id, sampling_dir, pre_id, dataset_id
    for ds_dir in cot_dir.rglob("*/*/*"):
        if ds_dir.is_dir():
            cot_dataset_ids.add(ds_dir.name)
    # common = direct_dataset_ids & cot_dataset_ids
    filtered = set()
    for ds_id in cot_dataset_ids:
        if ds_id.endswith("tests"):
            continue
        filtered.add(ds_id)
    return filtered


def analyze_direct_eval(
    ds_params: DatasetParams, instr_id: str, model_id: str
) -> list[dict]:
    """Analyze direct evaluation results for a specific dataset, instruction, and model."""
    direct_eval = ds_params.load_direct_eval(instr_id, model_id)
    questions = ds_params.load_qs_dataset().question_by_qid

    results = []
    for qid, probs in direct_eval.probs_by_qid.items():
        question = questions[qid]
        results.append(
            {
                "q_str": question.q_str,
                "qid": qid,
                "prop_id": ds_params.prop_id,
                "comparison": ds_params.comparison,
                "answer": ds_params.answer,
                "dataset_id": ds_params.id,
                "model_id": model_id,
                "p_yes": probs.p_yes,
                "p_no": probs.p_no,
                "p_correct": probs.p_yes if ds_params.answer == "YES" else probs.p_no,
                "mode": "direct",
                "instr_id": instr_id,
                "x_name": question.x_name,
                "y_name": question.y_name,
                "x_value": question.x_value,
                "y_value": question.y_value,
            }
        )

    return results


def analyze_cot_eval(
    ds_params: DatasetParams,
    instr_id: str,
    model_id: str,
    sampling_params: SamplingParams,
) -> list[dict]:
    """Analyze CoT evaluation results for a specific dataset, instruction, model and sampling params."""
    if instr_id == "instr-v0":
        cot_eval = ds_params.load_old_cot_eval(instr_id, model_id, sampling_params)
    else:
        cot_eval = ds_params.load_cot_eval(instr_id, model_id, sampling_params)
    questions = ds_params.load_qs_dataset().question_by_qid

    results = []
    for qid, eval_results in cot_eval.results_by_qid.items():
        question = questions[qid]
        total_count = len(eval_results)

        if isinstance(cot_eval, CotEval):
            yes_count = sum(1 for r in eval_results.values() if r.result == "YES")
            no_count = sum(1 for r in eval_results.values() if r.result == "NO")
            unknown_count = sum(1 for r in eval_results.values() if r.result == "UNKNOWN")
        else:
            yes_count = sum(1 for r in eval_results.values() if r == "YES")
            no_count = sum(1 for r in eval_results.values() if r == "NO")
            unknown_count = sum(1 for r in eval_results.values() if r == "UNKNOWN")

        known_count = yes_count + no_count
        n_yes = yes_count + 0.5 * unknown_count
        p_yes = n_yes / total_count
        p_no = 1 - p_yes
        results.append(
            {
                "q_str": question.q_str,
                "qid": qid,
                "prop_id": ds_params.prop_id,
                "comparison": ds_params.comparison,
                "answer": ds_params.answer,
                "dataset_id": ds_params.id,
                "dataset_suffix": ds_params.suffix,
                "model_id": model_id,
                "p_yes": p_yes,
                "p_no": p_no,
                "p_correct": p_yes if ds_params.answer == "YES" else p_no,
                "mode": "cot",
                "instr_id": instr_id,
                "x_name": question.x_name,
                "y_name": question.y_name,
                "x_value": question.x_value,
                "y_value": question.y_value,
                "temperature": cot_eval.sampling_params.temperature,
                "top_p": cot_eval.sampling_params.top_p,
                "max_new_tokens": cot_eval.sampling_params.max_new_tokens,
                "unknown_rate": unknown_count / len(eval_results),
                "yes_count": yes_count,
                "no_count": no_count,
                "unknown_count": unknown_count,
                "known_count": known_count,
            }
        )

    return results


def process_wm_cot_evals(dataset_pattern: str | None = None, out_path: str | None = None):
    """Process CoT evaluations for the WM instruction."""
    if out_path is None:
        out_path = DATA_DIR / "df-wm.pkl"
    assert out_path is not None
    logging.info(f"Processing WM CoT evaluations for {out_path}")

    all_results = []
    instr_id = "instr-wm"
    dataset_ids = get_dataset_ids(instr_id)
    for dataset_id in tqdm(dataset_ids):
        if dataset_pattern and dataset_pattern not in dataset_id:
            logging.info(f"Skipping {dataset_id} because it doesn't match pattern {dataset_pattern}")
            continue
        
        ds_params = DatasetParams.from_id(dataset_id)

        # Process CoT evaluations
        cot_eval_path = DATA_DIR / "cot_eval" / instr_id
        for model_file in cot_eval_path.rglob(
            f"*/{ds_params.pre_id}/{dataset_id}/*.yaml"
        ):
            sampling_dir = model_file.parent.parent.parent
            instr_id = sampling_dir.parent.name
            model_id = model_file.stem.replace("__", "/")
            temp_str, top_p_str, max_new_tokens_str = sampling_dir.name.split("_")
            sampling_params = SamplingParams(
                temperature=float(temp_str[1:]),
                top_p=float(top_p_str[1:]),
                max_new_tokens=int(max_new_tokens_str[1:]),
            )
            all_results.extend(
                analyze_cot_eval(ds_params, instr_id, model_id, sampling_params)
            )

    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    df.to_pickle(out_path)
    print(f"Saved analysis results to {out_path}")
    print(f"Total rows: {len(df)}")
    print("\nColumns:", ", ".join(df.columns))


def process_v0_cot_evals(dataset_pattern: str | None = None, out_path: str | None = None):
    """Process CoT evaluations for the V0 instruction."""
    if out_path is None:
        out_path = DATA_DIR / "df.pkl"
    assert out_path is not None

    all_results = []
    instr_id = "instr-v0"
    dataset_ids = get_dataset_ids(instr_id)
    for dataset_id in tqdm(dataset_ids):
        if dataset_pattern and dataset_pattern not in dataset_id:
            logging.info(f"Skipping {dataset_id} because it doesn't match pattern {dataset_pattern}")
            continue
        
        ds_params = DatasetParams.from_id(dataset_id)

        # # Process direct evaluations
        direct_eval_path = DATA_DIR / "direct_eval"
        for model_file in direct_eval_path.rglob(
            f"*/{ds_params.pre_id}/{dataset_id}/*.yaml"
        ):
            instr_id = model_file.parent.parent.parent.name
            model_id = model_file.stem.replace("__", "/")
            all_results.extend(analyze_direct_eval(ds_params, instr_id, model_id))

        # Process CoT evaluations
        cot_eval_path = DATA_DIR / "cot_eval" / instr_id
        for model_file in cot_eval_path.rglob(
            f"*/{ds_params.pre_id}/{dataset_id}/*.yaml"
        ):
            sampling_dir = model_file.parent.parent.parent
            instr_id = sampling_dir.parent.name
            model_id = model_file.stem.replace("__", "/")
            temp_str, top_p_str, max_new_tokens_str = sampling_dir.name.split("_")
            sampling_params = SamplingParams(
                temperature=float(temp_str[1:]),
                top_p=float(top_p_str[1:]),
                max_new_tokens=int(max_new_tokens_str[1:]),
            )
            all_results.extend(
                analyze_cot_eval(ds_params, instr_id, model_id, sampling_params)
            )

    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    df.to_pickle(out_path)
    print(f"Saved analysis results to {out_path}")
    print(f"Total rows: {len(df)}")
    print("\nColumns:", ", ".join(df.columns))


@click.command()
@click.option(
    "--instr-id",
    "-i",
    type=str,
    default="instr-wm",
    help="Instruction ID to process",
)
@click.option(
    "--dataset-pattern",
    "-p",
    type=str,
    default=None,
    help="Dataset pattern to process. This pattern must be a substring of the dataset ID.",
)
@click.option(
    "--out-path",
    "-o",
    type=str,
    default=None,
    help="Output path to save the DataFrame",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose logging",
)
def main(instr_id: str, dataset_pattern: str | None = None, out_path: str | None = None, verbose: bool = False):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    if instr_id == "instr-wm":
        process_wm_cot_evals(dataset_pattern, out_path)
    elif instr_id == "instr-v0":
        process_v0_cot_evals(dataset_pattern, out_path)
    else:
        raise click.BadParameter(f"Invalid instruction ID: {instr_id}")


if __name__ == "__main__":
    main()
