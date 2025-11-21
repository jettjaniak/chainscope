#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml
from beartype import beartype
from tqdm import tqdm

from chainscope.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


@beartype
def is_new_format(raw: Any) -> bool:
    return isinstance(raw, dict) and "questions_by_qid" in raw


@beartype
def infer_dataset_suffix(stem: str, prop_id: str | None) -> str | None:
    if prop_id is None:
        return None
    if stem == prop_id:
        return None
    prefix = f"{prop_id}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return None


@beartype
def gather_model_id(file_path: Path, existing: dict[str, str]) -> str:
    model_dir = file_path.parent
    cached = existing.get(model_dir.name)
    if cached is not None:
        return cached

    for sibling in model_dir.glob("*.yaml"):
        if sibling == file_path:
            continue
        with sibling.open("r") as handle:
            sibling_raw = yaml.safe_load(handle)
        if is_new_format(sibling_raw):
            candidate = sibling_raw.get("model_id")
            if isinstance(candidate, str) and candidate:
                existing[model_dir.name] = candidate
                return candidate

    existing[model_dir.name] = model_dir.name
    return model_dir.name


@beartype
def build_q1_all_responses(
    faithful: dict[str, dict[str, Any]],
    unfaithful: dict[str, dict[str, Any]],
    unknown: dict[str, dict[str, Any]],
) -> dict[str, str]:
    aggregated: dict[str, str] = {}
    for group in (faithful, unfaithful, unknown):
        for rid, payload in group.items():
            response = payload.get("response")
            if isinstance(response, str):
                aggregated[rid] = response
    return aggregated


@beartype
def build_q2_all_responses(
    correct: dict[str, dict[str, Any]],
    incorrect: dict[str, dict[str, Any]],
) -> dict[str, str]:
    aggregated: dict[str, str] = {}
    for group in (correct, incorrect):
        for rid, payload in group.items():
            response = payload.get("response")
            if isinstance(response, str):
                aggregated[rid] = response
    return aggregated


@beartype
def migrate_entry(
    qid: str,
    qdata: dict[str, Any],
    prop_id: str,
    dataset_suffix: str | None,
) -> tuple[str, dict[str, Any]]:
    prompt = qdata.get("prompt")
    assert isinstance(prompt, str)

    faithful = qdata.get("faithful_responses") or {}
    unfaithful = qdata.get("unfaithful_responses") or {}
    unknown = qdata.get("unknown_responses") or {}

    assert isinstance(faithful, dict)
    assert isinstance(unfaithful, dict)
    assert isinstance(unknown, dict)

    metadata = qdata.get("metadata") or {}
    assert isinstance(metadata, dict)

    meta = metadata.copy()
    meta.setdefault("prop_id", prop_id)
    meta.setdefault("comparison", meta.get("comparison", "gt"))
    dataset_id = meta.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        meta["dataset_id"] = f"{prop_id}_unknown"
    meta["dataset_suffix"] = dataset_suffix
    if "group_p_yes_mean" not in meta:
        raise ValueError(f"Missing group_p_yes_mean for {qid}")
    meta.setdefault("is_oversampled", False)
    reversed_q_id = meta.get("reversed_q_id")
    if reversed_q_id is None:
        raise ValueError(f"Missing reversed_q_id for {qid}")
    reversed_dataset_id = meta.get("reversed_q_dataset_id")
    if not isinstance(reversed_dataset_id, str) or not reversed_dataset_id:
        meta["reversed_q_dataset_id"] = f"{reversed_q_id}_dataset"
    if "reversed_q_dataset_suffix" not in meta:
        meta["reversed_q_dataset_suffix"] = dataset_suffix

    correct = meta.get("reversed_q_correct_responses") or {}
    incorrect = meta.get("reversed_q_incorrect_responses") or {}
    assert isinstance(correct, dict)
    assert isinstance(incorrect, dict)

    q1_all = meta.get("q1_all_responses")
    if not isinstance(q1_all, dict) or not q1_all:
        meta["q1_all_responses"] = build_q1_all_responses(faithful, unfaithful, unknown)
    q2_all = meta.get("q2_all_responses")
    if not isinstance(q2_all, dict) or not q2_all:
        meta["q2_all_responses"] = build_q2_all_responses(correct, incorrect)

    question_payload = {
        "prompt": prompt,
        "faithful_responses": faithful,
        "unfaithful_responses": unfaithful,
        "unknown_responses": unknown,
        "metadata": meta,
    }
    return qid, question_payload


@beartype
def migrate_file(
    file_path: Path,
    dry_run: bool,
    model_id_cache: dict[str, str],
    verbose: bool,
) -> bool:
    with file_path.open("r") as handle:
        raw = yaml.safe_load(handle)

    if is_new_format(raw):
        if verbose:
            click.echo(f"[skip] {file_path} already migrated.")
        return False

    assert isinstance(raw, dict) and raw, f"Expected non-empty dict for {file_path}"

    first_key = next(iter(raw.keys()))
    first_entry = raw[first_key]
    assert isinstance(first_entry, dict)
    metadata = first_entry.get("metadata") or {}
    assert isinstance(metadata, dict)

    prop_id = metadata.get("prop_id")
    if not isinstance(prop_id, str) or not prop_id:
        prop_id = file_path.stem.split("_", 1)[0]

    dataset_suffix = infer_dataset_suffix(file_path.stem, prop_id)

    model_id = gather_model_id(file_path, model_id_cache)

    questions: dict[str, dict[str, Any]] = {}
    for qid, qdata in raw.items():
        assert isinstance(qid, str)
        assert isinstance(qdata, dict)
        migrated_qid, payload = migrate_entry(qid, qdata, prop_id, dataset_suffix)
        questions[migrated_qid] = payload

    new_payload = {
        "model_id": model_id,
        "prop_id": prop_id,
        "dataset_suffix": dataset_suffix,
        "questions_by_qid": questions,
    }

    if dry_run:
        click.echo(f"[dry-run] Would migrate {file_path}")
        return True

    with file_path.open("w") as handle:
        yaml.safe_dump(new_payload, handle, sort_keys=False)

    click.echo(f"[migrated] {file_path}")
    return True


@click.command()
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("d/faithfulness"),
    show_default=True,
    help="Root directory containing faithfulness datasets to migrate.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report files that would be migrated without writing changes.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print additional information while scanning.",
)
def main(root: Path, dry_run: bool, verbose: bool) -> None:
    """Migrate legacy unfaithfulness datasets to the new dataclass format."""
    model_id_cache: dict[str, str] = {}
    yaml_files = sorted(root.glob("*/*.yaml"))
    assert yaml_files, f"No YAML files found under {root}"

    migrated = 0
    failures: list[tuple[Path, Exception]] = []
    for file_path in tqdm(yaml_files, desc="Migrating datasets"):
        try:
            migrated_now = migrate_file(
                file_path=file_path,
                dry_run=dry_run,
                model_id_cache=model_id_cache,
                verbose=verbose,
            )
            if migrated_now:
                migrated += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures.append((file_path, exc))
            click.echo(f"[failed] {file_path}: {exc}")
            continue

    click.echo(f"Processed {len(yaml_files)} files, migrated {migrated}.")
    if failures:
        click.echo("Failures encountered:")
        for path, exc in failures:
            click.echo(f"- {path}: {exc}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
