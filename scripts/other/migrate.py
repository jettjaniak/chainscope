#!/usr/bin/env python3

from dataclass_wizard import DumpMeta, LoadMeta

from chainscope.typing import (
    DATA_DIR,
    CotEval,
    CotEvalResult,
    CotResponses,
    DatasetParams,
    DirectEval,
    DirectEvalProbs,
    OldCotEval,
    Properties,
    QsDataset,
    Question,
    SamplingParams,
)


def migrate_questions() -> None:
    """Migrate qs_dataset files to include DatasetParams structure."""
    qs_dataset_dir = DATA_DIR / "questions"

    for dataset_file in qs_dataset_dir.rglob("*.yaml"):
        print(f"Processing {dataset_file}")
        try:
            # Load with LISP key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(QsDataset)
            DumpMeta(key_transform="LISP").bind_to(QsDataset)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(Question)
            DumpMeta(key_transform="LISP").bind_to(Question)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="LISP").bind_to(DatasetParams)

            qs_dataset = QsDataset.load_from_path(dataset_file)

            # dump with SNAKE key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(QsDataset)
            DumpMeta(key_transform="SNAKE").bind_to(QsDataset)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(Question)
            DumpMeta(key_transform="SNAKE").bind_to(Question)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="SNAKE").bind_to(DatasetParams)

            qs_dataset.to_yaml_file(dataset_file)
        except Exception:
            print(f"Skipping {dataset_file} due to error")


def migrate_direct_eval() -> None:
    """Migrate direct_eval files to include DatasetParams structure."""
    direct_eval_dir = DATA_DIR / "direct_eval"

    for eval_file in direct_eval_dir.rglob("*.yaml"):
        print(f"Processing {eval_file}")
        try:
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(DirectEval)
            DumpMeta(key_transform="LISP").bind_to(DirectEval)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(DirectEvalProbs)
            DumpMeta(key_transform="LISP").bind_to(DirectEvalProbs)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="LISP").bind_to(DatasetParams)

            direct_eval = DirectEval.load(eval_file)

            # dump with SNAKE key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(DirectEval)
            DumpMeta(key_transform="SNAKE").bind_to(DirectEval)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(DirectEvalProbs)
            DumpMeta(key_transform="SNAKE").bind_to(DirectEvalProbs)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="SNAKE").bind_to(DatasetParams)

            direct_eval.to_yaml_file(eval_file)
        except Exception:
            print(f"Skipping {eval_file} due to error")


def migrate_cot_responses() -> None:
    """Migrate cot_responses files to include DatasetParams structure."""
    cot_responses_dir = DATA_DIR / "cot_responses"

    for response_file in cot_responses_dir.rglob("*.yaml"):
        print(f"Processing {response_file}")
        try:
            # Load with LISP key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(CotResponses)
            DumpMeta(key_transform="LISP").bind_to(CotResponses)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="LISP").bind_to(DatasetParams)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(SamplingParams)
            DumpMeta(key_transform="LISP").bind_to(SamplingParams)

            cot_responses = CotResponses.load(response_file)

            # dump with SNAKE key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(CotResponses)
            DumpMeta(key_transform="SNAKE").bind_to(CotResponses)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="SNAKE").bind_to(DatasetParams)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(SamplingParams)
            DumpMeta(key_transform="SNAKE").bind_to(SamplingParams)

            cot_responses.to_yaml_file(response_file)
        except Exception:
            print(f"Skipping {response_file} due to error")


def migrate_cot_evals() -> None:
    """Migrate cot_evals files to include DatasetParams structure."""
    cot_evals_dir = DATA_DIR / "cot_eval" / "instr-wm"

    for eval_file in cot_evals_dir.rglob("*.yaml"):
        print(f"Processing {eval_file}")
        try:
            # Load with LISP key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(CotEval)
            DumpMeta(key_transform="LISP").bind_to(CotEval)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(CotEvalResult)
            DumpMeta(key_transform="LISP").bind_to(CotEvalResult)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="LISP").bind_to(DatasetParams)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(SamplingParams)
            DumpMeta(key_transform="LISP").bind_to(SamplingParams)

            cot_evals = CotEval.load(eval_file)

            # dump with SNAKE key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(CotEval)
            DumpMeta(key_transform="SNAKE").bind_to(CotEval)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(CotEvalResult)
            DumpMeta(key_transform="SNAKE").bind_to(CotEvalResult)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="SNAKE").bind_to(DatasetParams)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(SamplingParams)
            DumpMeta(key_transform="SNAKE").bind_to(SamplingParams)

            cot_evals.to_yaml_file(eval_file)
        except Exception:
            print(f"Skipping {eval_file} due to error")


def migrate_properties() -> None:
    """Migrate properties files to include DatasetParams structure."""
    properties_dir = DATA_DIR / "properties"

    for property_file in properties_dir.rglob("*.yaml"):
        print(f"Processing {property_file}")
        try:
            # Load with LISP key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(Properties)
            DumpMeta(key_transform="LISP").bind_to(Properties)

            properties = Properties.load_from_path(property_file)

            # dump with SNAKE key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(Properties)
            DumpMeta(key_transform="SNAKE").bind_to(Properties)

            properties.to_yaml_file(property_file)
        except Exception:
            print(f"Skipping {property_file} due to error")


def migrate_old_cot_evals() -> None:
    """Migrate cot_evals files to include DatasetParams structure."""
    cot_evals_dir = DATA_DIR / "cot_eval" / "instr-v0"

    for eval_file in cot_evals_dir.rglob("*.yaml"):
        print(f"Processing {eval_file}")
        try:
            # Load with LISP key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(OldCotEval)
            DumpMeta(key_transform="LISP").bind_to(OldCotEval)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="LISP").bind_to(DatasetParams)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="LISP"
            ).bind_to(SamplingParams)
            DumpMeta(key_transform="LISP").bind_to(SamplingParams)

            cot_evals = OldCotEval.load(eval_file)

            # dump with SNAKE key transform
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(OldCotEval)
            DumpMeta(key_transform="SNAKE").bind_to(OldCotEval)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(DatasetParams)
            DumpMeta(key_transform="SNAKE").bind_to(DatasetParams)
            LoadMeta(
                v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
            ).bind_to(SamplingParams)
            DumpMeta(key_transform="SNAKE").bind_to(SamplingParams)

            cot_evals.to_yaml_file(eval_file)
        except Exception as e:
            print(f"Skipping {eval_file} due to error: {e}")


if __name__ == "__main__":
    # migrate_questions()
    # migrate_direct_eval()
    # migrate_cot_responses()
    # migrate_cot_evals()
    migrate_old_cot_evals()
    # migrate_properties()
