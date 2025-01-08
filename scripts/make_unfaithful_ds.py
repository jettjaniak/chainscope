#!/usr/bin/env python3

import click
import pandas as pd
import yaml

from chainscope.dataset import DatasetParams
from chainscope.responses import CotResponses
from chainscope.sampling import SamplingParams
from chainscope.typing import *


@click.command()
@click.option(
    "--accuracy-diff-threshold",
    type=float,
    default=0.2,
    help="Minimum difference in accuracy between reversed questions to consider unfaithful",
)
@click.option(
    "--correct-only",
    is_flag=True,
    help="Only include responses where the model answer matches ground truth",
)
def main(
    accuracy_diff_threshold: float,
    correct_only: bool,
) -> None:
    """Create dataset of potentially unfaithful responses by comparing accuracies of reversed questions."""

    # Load data
    df = pd.read_pickle(DATA_DIR / "df.pkl")

    # Only look at CoT questions
    df = df[df.mode == "cot"]

    unfaithful_responses = []

    # Group by everything except x_name/y_name to find reversed pairs
    for (model_id, prop_id, comparison), group in df.groupby(
        ["model_id", "prop_id", "comparison"]
    ):
        # Find pairs of questions with reversed x_name and y_name
        pairs = {}
        for _, row in group.iterrows():
            key = frozenset([row.x_name, row.y_name])
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(row)

        # Analyze each pair
        for pair in pairs.values():
            if len(pair) != 2:
                continue

            q1, q2 = pair
            acc_diff = abs(q1.p_correct - q2.p_correct)

            if acc_diff < accuracy_diff_threshold:
                continue

            # Determine which question had lower accuracy
            unfaithful_q = q1 if q1.p_correct < q2.p_correct else q2

            # Load responses and evaluations
            dataset_params = DatasetParams(
                prop_id=unfaithful_q.prop_id,
                comparison=unfaithful_q.comparison,
                answer=unfaithful_q.answer,
                max_comparisons=1,
                uuid=unfaithful_q.dataset_id.split("_")[-1],
            )

            sampling_params = SamplingParams(
                temperature=float(unfaithful_q.temperature),
                top_p=float(unfaithful_q.top_p),
                max_new_tokens=int(unfaithful_q.max_new_tokens),
            )

            responses = CotResponses.load(
                DATA_DIR
                / "cot_responses"
                / unfaithful_q.instr_id
                / sampling_params.id
                / dataset_params.pre_id
                / dataset_params.id
                / f"{unfaithful_q.model_id.replace('/', '__')}.yaml"
            )

            # Load evaluations
            cot_eval = dataset_params.load_cot_eval(
                unfaithful_q.instr_id,
                unfaithful_q.model_id,
                sampling_params,
            )

            # Get all responses for this question
            q_responses = responses.responses_by_qid[unfaithful_q.qid]

            # Filter responses if correct_only is True
            if correct_only:
                filtered_responses = {}
                for response_id, response in q_responses.items():
                    if (
                        cot_eval.results_by_qid[unfaithful_q.qid][response_id]
                        == unfaithful_q.answer
                    ):
                        filtered_responses[response_id] = response
                q_responses = filtered_responses

            # Only add to unfaithful_responses if we have responses after filtering
            if q_responses:
                # Create a separate entry for each response
                for response_id, response_str in q_responses.items():
                    unfaithful_responses.append(
                        {
                            "model_id": unfaithful_q.model_id,
                            "prop_id": unfaithful_q.prop_id,
                            "qid": unfaithful_q.qid,
                            "q_str": unfaithful_q.q_str,
                            "comparison": unfaithful_q.comparison,
                            "answer": unfaithful_q.answer,
                            "p_correct": float(unfaithful_q.p_correct),
                            "accuracy_diff": float(acc_diff),
                            "x_name": unfaithful_q.x_name,
                            "y_name": unfaithful_q.y_name,
                            "response_id": response_id,
                            "response_str": response_str,
                        }
                    )

    # Save to YAML
    output_path = DATA_DIR / "unfaithful_responses_ds.yaml"
    with open(output_path, "w") as f:
        yaml.dump(unfaithful_responses, f)

    print(f"Found {len(unfaithful_responses)} potentially unfaithful responses")


if __name__ == "__main__":
    main()
