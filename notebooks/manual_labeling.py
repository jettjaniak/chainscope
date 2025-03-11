# %%

import random

from chainscope.typing import CoTPath, CoTPathEval, ProblemDataset
from chainscope.utils import MODELS_MAP

# %%

model_aliases = ["GPT4O", "C3.5S"]
dataset_ids = ["gsm8k"]  # "math" and "mmlu" are way harder

# %%

# Load all CoTPathEval objects
cot_path_evals: dict[tuple[str, str], CoTPathEval] = {}
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path_evals[(model_id, dataset_id)] = CoTPathEval.load(model_id, dataset_id)

# Load the responses
cot_paths: dict[tuple[str, str], CoTPath] = {}
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path = CoTPath.load(model_id, dataset_id)
        cot_paths[(model_id, dataset_id)] = cot_path

# Load datasets
datasets: dict[str, ProblemDataset] = {}
for dataset_id in dataset_ids:
    datasets[dataset_id] = ProblemDataset.load(dataset_id)

# %%

# Collect all unfaithful responses
unfaithful_responses: dict[
    tuple[str, str], dict[str, dict[str, list[int]]]
] = {}  # (model_id, dataset_id) -> qid -> response_uuid -> [unfaithful_steps]

for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path_eval = cot_path_evals[(model_id, dataset_id)]

        unfaithful_responses[(model_id, dataset_id)] = {}

        for qid in cot_path_eval.third_pass_eval_by_qid.keys():
            assert (
                len(cot_path_eval.third_pass_eval_by_qid[qid]) == 1
            ), "Expected 1 response per qid"
            response_uuid = list(cot_path_eval.third_pass_eval_by_qid[qid].keys())[0]

            # Collect all unfaithful steps
            unfaithful_steps = []
            for step_num, step_status in cot_path_eval.third_pass_eval_by_qid[qid][
                response_uuid
            ].steps_status.items():
                if step_status.is_unfaithful:
                    unfaithful_steps.append(step_num)

            if unfaithful_steps:
                unfaithful_responses[(model_id, dataset_id)][qid] = {}
                unfaithful_responses[(model_id, dataset_id)][qid][response_uuid] = (
                    unfaithful_steps
                )

        print(
            f"Number of unfaithful responses for {model_alias} on {dataset_id}: {len(unfaithful_responses[(model_id, dataset_id)])}"
        )

# %%

# Collect responses that have correct answer and all steps correct
correct_responses: dict[
    tuple[str, str], dict[str, str]
] = {}  # (model_id, dataset_id) -> qid -> response_uuid

for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path_eval = cot_path_evals[(model_id, dataset_id)]

        correct_responses[(model_id, dataset_id)] = {}

        for qid in cot_path_eval.answer_correct_by_qid.keys():
            assert (
                len(cot_path_eval.answer_correct_by_qid[qid]) == 1
            ), "Expected 1 response per qid"
            response_uuid = list(cot_path_eval.answer_correct_by_qid[qid].keys())[0]

            answer_correct = (
                cot_path_eval.answer_correct_by_qid[qid][response_uuid].answer_correct
                == "CORRECT"
            )

            if (
                qid not in cot_path_eval.first_pass_eval_by_qid
                or response_uuid not in cot_path_eval.first_pass_eval_by_qid[qid]
            ):
                continue

            all_steps_correct = True
            for step_num, step_status in cot_path_eval.first_pass_eval_by_qid[qid][
                response_uuid
            ].steps_status.items():
                if step_status.node_status != "CORRECT":
                    all_steps_correct = False
                    break

            if all_steps_correct and answer_correct:
                correct_responses[(model_id, dataset_id)][qid] = response_uuid

        print(
            f"Number of correct responses for {model_alias} on {dataset_id}: {len(correct_responses[(model_id, dataset_id)])}"
        )

# %%

# Collect responses that have at least one incorrect step, and are not marked as unfaithful
incorrect_responses: dict[
    tuple[str, str], dict[str, str]
] = {}  # (model_id, dataset_id) -> qid -> response_uuid

for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path_eval = cot_path_evals[(model_id, dataset_id)]

        incorrect_responses[(model_id, dataset_id)] = {}

        for qid in cot_path_eval.first_pass_eval_by_qid.keys():
            assert (
                len(cot_path_eval.first_pass_eval_by_qid[qid]) == 1
            ), "Expected 1 response per qid"
            response_uuid = list(cot_path_eval.first_pass_eval_by_qid[qid].keys())[0]

            at_least_one_incorrect = False
            for step_num, step_status in cot_path_eval.first_pass_eval_by_qid[qid][
                response_uuid
            ].steps_status.items():
                if step_status.node_status != "CORRECT":
                    at_least_one_incorrect = True
                    break

            has_unfaithful_steps = (
                unfaithful_responses[(model_id, dataset_id)]
                .get(qid, {})
                .get(response_uuid, [])
            ) != []

            if at_least_one_incorrect and not has_unfaithful_steps:
                incorrect_responses[(model_id, dataset_id)][qid] = response_uuid

        print(
            f"Number of incorrect responses for {model_alias} on {dataset_id}: {len(incorrect_responses[(model_id, dataset_id)])}"
        )


# %%

# Set random seed
random.seed(42)

# Collect responses for manual labeling, based on the 3 buckets defined above
unfaithful_qids_per_bucket = 50
correct_qids_per_bucket = 50
incorrect_qids_per_bucket = 50

responses_for_manual_labeling: dict[
    tuple[str, str], dict[str, str]
] = {}  # (model_id, dataset_id) -> qid -> response_uuid

for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path_eval = cot_path_evals[(model_id, dataset_id)]

        responses_for_manual_labeling[(model_id, dataset_id)] = {}

        # Unfaithful responses - sample qids
        for qid in random.sample(
            list(unfaithful_responses[(model_id, dataset_id)].keys()),
            min(
                unfaithful_qids_per_bucket,
                len(unfaithful_responses[(model_id, dataset_id)]),
            ),
        ):
            assert (
                len(unfaithful_responses[(model_id, dataset_id)][qid]) == 1
            ), "Expected 1 unfaithful response per qid"
            response_uuid = list(
                unfaithful_responses[(model_id, dataset_id)][qid].keys()
            )[0]
            responses_for_manual_labeling[(model_id, dataset_id)][qid] = response_uuid
            print(f"Qid={qid} in responses for manual labeling is unfaithful")

        # Correct responses - sample qids
        for qid in random.sample(
            list(correct_responses[(model_id, dataset_id)].keys()),
            min(
                correct_qids_per_bucket,
                len(correct_responses[(model_id, dataset_id)]),
            ),
        ):
            responses_for_manual_labeling[(model_id, dataset_id)][qid] = (
                correct_responses[(model_id, dataset_id)][qid]
            )

        # Incorrect responses - sample qids
        for qid in random.sample(
            list(incorrect_responses[(model_id, dataset_id)].keys()),
            min(
                incorrect_qids_per_bucket,
                len(incorrect_responses[(model_id, dataset_id)]),
            ),
        ):
            responses_for_manual_labeling[(model_id, dataset_id)][qid] = (
                incorrect_responses[(model_id, dataset_id)][qid]
            )

        print(
            f"Number of responses for manual labeling for {model_alias} on {dataset_id}: {len(responses_for_manual_labeling[(model_id, dataset_id)])}"
        )


print(
    f"Total number of responses for manual labeling: {sum([len(v) for v in responses_for_manual_labeling.values()])}"
)
# %%

# Set random seed
random.seed(42)

# Print qid and response_uuid for each response selected for manual labeling
manual_labeling_items = []
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        for qid, response_uuid in responses_for_manual_labeling[
            (model_id, dataset_id)
        ].items():
            manual_labeling_items.append((model_id, dataset_id, qid, response_uuid))


# Shuffle the list
random.shuffle(manual_labeling_items)

for item in manual_labeling_items:
    print(f"{item[2]};{item[3]}")
# %%


def show_path(qid, response_uuid, model_id, dataset_id):
    cot_path = cot_paths[(model_id, dataset_id)]

    print(f"Question ID: {qid}")
    print(f"Response ID: {response_uuid}\n")
    print(f"Question: `{datasets[dataset_id].problems_by_qid[qid].q_str}`\n")
    print(
        f"Ground truth answer: `{datasets[dataset_id].problems_by_qid[qid].answer}`\n"
    )

    response_str = "Response: `"
    for step_id in cot_path.cot_path_by_qid[qid][response_uuid].keys():
        step = cot_path.cot_path_by_qid[qid][response_uuid][step_id]
        response_str += f"{step}\n"
    # Remove the last newline
    response_str = response_str[:-1]
    response_str += "`"
    print(response_str)


# %%

# Print the responses
all_responses_str = ""
for item in manual_labeling_items:
    model_id, dataset_id, qid, response_uuid = item
    show_path(qid, response_uuid, model_id, dataset_id)
    print("""

----------------------------------------------------------------------------------

""")
# %%


def show_path_with_automatic_analysis(qid, response_uuid, model_id, dataset_id):
    dataset = datasets[dataset_id]
    cot_path = cot_paths[(model_id, dataset_id)]
    cot_path_eval = cot_path_evals[(model_id, dataset_id)]

    print(f"{qid}, {response_uuid}")
    print(f"Question: `{dataset.problems_by_qid[qid].q_str}`")
    print()
    print(f"Ground truth answer: `{dataset.problems_by_qid[qid].answer}`")
    print()
    # Print each step with its evaluation
    for step_id in cot_path.cot_path_by_qid[qid][response_uuid].keys():
        step = cot_path.cot_path_by_qid[qid][response_uuid][step_id]
        print(f"Step {step_id}:")
        print(f" - Content: `{step}`")

        print(
            f" - First pass status: {cot_path_eval.first_pass_eval_by_qid[qid][response_uuid].steps_status[step_id].node_status}"
        )
        print(
            f" - First pass explanation: {cot_path_eval.first_pass_eval_by_qid[qid][response_uuid].steps_status[step_id].explanation}"
        )

        if (
            step_id
            in cot_path_eval.second_pass_eval_by_qid[qid][response_uuid].steps_status
        ):
            step_eval = cot_path_eval.second_pass_eval_by_qid[qid][
                response_uuid
            ].steps_status[step_id]
            print(
                f" - Second pass status: {step_eval.node_status}, severity: {step_eval.node_severity}"
            )
            if step_eval.explanation:
                print(f" - Second pass explanation: {step_eval.explanation}")
            else:
                print(" - Second pass explanation is missing for this step.")

            if cot_path_eval.second_pass_eval_by_qid[qid][response_uuid].steps_status[
                step_id
            ].node_status == "UNFAITHFUL" and cot_path_eval.second_pass_eval_by_qid[
                qid
            ][response_uuid].steps_status[step_id].node_severity in [
                "MINOR",
                "MAJOR",
                "CRITICAL",
            ]:
                step_eval = cot_path_eval.third_pass_eval_by_qid[qid][
                    response_uuid
                ].steps_status[step_id]
                print(
                    f" - Third pass status: {step_eval.node_severity}, is_unfaithful: {step_eval.is_unfaithful}"
                )
                if step_eval.explanation:
                    print(f" - Third pass explanation: {step_eval.explanation}")
                else:
                    print(" - Third pass explanation is missing for this step.")
            else:
                print(" - Third pass eval is missing for this step")
        else:
            print(" - Second pass eval is missing for this step")


# %%

# Show the paths with automatic analysis for 10 random unfaithful responses of each model
random_selection_size = 10
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in ["gsm8k"]:
        unfaithful_responses_for_this_model_and_dataset = []
        for qid in unfaithful_responses[(model_id, dataset_id)].keys():
            responses = unfaithful_responses[(model_id, dataset_id)][qid]
            assert len(responses) == 1, "Expected 1 unfaithful response per qid"
            response_uuid = list(responses.keys())[0]
            unfaithful_responses_for_this_model_and_dataset.append((qid, response_uuid))
        print(
            f"Model {model_alias} has {len(unfaithful_responses_for_this_model_and_dataset)} unfaithful responses for dataset {dataset_id}\n"
        )
        sample_size = min(
            random_selection_size,
            len(unfaithful_responses_for_this_model_and_dataset),
        )
        for qid, response_uuid in random.sample(
            unfaithful_responses_for_this_model_and_dataset, sample_size
        ):
            show_path_with_automatic_analysis(qid, response_uuid, model_id, dataset_id)
            print("""

----------------------------------------------------------------------------------

""")

# %%
