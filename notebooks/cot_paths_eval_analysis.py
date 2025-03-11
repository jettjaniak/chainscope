# %%

import random
from collections import Counter

import matplotlib.pyplot as plt

from chainscope.cot_paths_eval import (
    build_answer_correctness_prompt,
    build_first_pass_prompt,
    build_second_pass_prompt,
    build_third_pass_prompt,
)
from chainscope.typing import CoTPath, CoTPathEval, ProblemDataset
from chainscope.utils import MODELS_MAP

# %%

model_alias = "GPT4O"
model_id = MODELS_MAP[model_alias]
dataset_id = "gsm8k"

# %%

# Load the data
cot_path = CoTPath.load(model_id, dataset_id)
cot_path_eval = CoTPathEval.load(model_id, dataset_id)
dataset = ProblemDataset.load(dataset_id)

# %%

# Check how many problems in the dataset we have answer_corect for
problem_ids = set(dataset.problems_by_qid.keys())
problem_ids_with_answer_correct_eval = set(cot_path_eval.answer_correct_by_qid.keys())

print(
    f"Number of problems with answer_correct_eval: {len(problem_ids_with_answer_correct_eval)}"
)
print(f"Number of problems in the dataset: {len(problem_ids)}")
assert (
    problem_ids_with_answer_correct_eval == problem_ids
), "Some problems are missing from the answer_correct_eval"

# Check how many responses we have answer_corect for
for qid in problem_ids:
    responses_for_qid = cot_path.cot_path_by_qid[qid]
    responses_with_answer_correct_eval = set(
        cot_path_eval.answer_correct_by_qid[qid].keys()
    )
    assert len(responses_for_qid) == len(
        responses_with_answer_correct_eval
    ), "Some problems are missing from the answer_correct_eval"

# %%

# Create a histogram of answer correctness labels

# Flatten all answer correctness results into a single list of labels
all_labels = [
    result.answer_status
    for results_dict in cot_path_eval.answer_correct_by_qid.values()
    for result in results_dict.values()
]

# Count occurrences of each label
label_counts = Counter(all_labels)
total = len(all_labels)

# Calculate percentages
label_percentages = {
    label: (count / total) * 100 for label, count in label_counts.items()
}

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(label_percentages.keys(), label_percentages.values())
plt.title(f"Distribution of Answer Correctness Labels\n{model_id} on {dataset_id}")
plt.ylabel("Percentage")
plt.ylim(0, 100)

# Add percentage labels on top of each bar
for i, (label, percentage) in enumerate(label_percentages.items()):
    plt.text(i, percentage + 1, f"{percentage:.1f}%", ha="center")

plt.show()

# Print raw counts for reference
print("\nRaw counts:")
for label, count in label_counts.items():
    print(f"{label}: {count} ({(count/total)*100:.1f}%)")

# %%

# Collect correct responses
total_qs = 0
qs_with_correct_responses = set()
total_responses = 0
total_correct_responses = 0
correct_responses = []  # tuples of (qid, response_uuid)
for qid in cot_path_eval.answer_correct_by_qid.keys():
    total_qs += 1
    responses_for_qid = cot_path_eval.answer_correct_by_qid[qid]
    for response_uuid, result in responses_for_qid.items():
        total_responses += 1
        if result.answer_status == "CORRECT":
            qs_with_correct_responses.add(qid)
            total_correct_responses += 1
            correct_responses.append((qid, response_uuid))

print(
    f"Questions with correct responses: {len(qs_with_correct_responses)} ({len(qs_with_correct_responses)/total_qs*100:.1f}% out of {total_qs} qs)"
)
print(
    f"Correct responses: {total_correct_responses} ({total_correct_responses/total_responses*100:.1f}% out of {total_responses} responses)"
)
# %%

# Check that we have a first pass eval for each correct response
missing_qs_in_first_pass_eval = 0
missing_responses_in_first_pass_eval = 0

# We will also count the number of first pass eval labels
first_pass_eval_labels = {}

# We will also collect non-empty explanations for nodes with status UNKNOWN
unknown_explanations = []

# We will also collect paths with correct answers but incorrect nodes
paths_with_incorrect_nodes = []

for qid, response_uuid in correct_responses:
    if qid not in cot_path_eval.first_pass_eval_by_qid:
        print(f"First pass eval is missing for question {qid}")
        missing_qs_in_first_pass_eval += 1
        continue
    if response_uuid not in cot_path_eval.first_pass_eval_by_qid[qid]:
        print(f"First pass eval is missing for question {qid} response {response_uuid}")
        missing_responses_in_first_pass_eval += 1
        continue

    # Check that we have the same steps in the path and first pass eval
    cot_path_steps = cot_path.cot_path_by_qid[qid][response_uuid].keys()
    first_pass_eval_steps = cot_path_eval.first_pass_eval_by_qid[qid][
        response_uuid
    ].steps_status.keys()
    assert (
        cot_path_steps == first_pass_eval_steps
    ), "Number of steps in the path does not match the number of steps in the first pass eval"

    for step_id, step_status in cot_path_eval.first_pass_eval_by_qid[qid][
        response_uuid
    ].steps_status.items():
        if step_status.node_status not in first_pass_eval_labels:
            first_pass_eval_labels[step_status.node_status] = 0
        first_pass_eval_labels[step_status.node_status] += 1

        if step_status.node_status == "UNKNOWN" and step_status.explanation:
            unknown_explanations.append(step_status.explanation)

        if step_status.node_status == "INCORRECT":
            paths_with_incorrect_nodes.append((qid, response_uuid))

print(
    f"Questions missing first pass eval: {missing_qs_in_first_pass_eval} ({missing_qs_in_first_pass_eval/total_qs*100:.1f}%)"
)
print(
    f"Responses missing first pass eval: {missing_responses_in_first_pass_eval} ({missing_responses_in_first_pass_eval/total_responses*100:.1f}%)"
)

print(first_pass_eval_labels)
print(
    f"Number of non-empty explanations for UNKNOWN nodes: {len(unknown_explanations)}"
)

# %%

# Show random example of a path with correct answer and node marked as incorrect
if paths_with_incorrect_nodes:
    # Pick a random example
    qid, response_uuid = random.choice(paths_with_incorrect_nodes)

    # Get the problem
    problem = dataset.problems_by_qid[qid]
    print(f"Question ID: {qid}")
    print(f"Question: {problem.q_str}\n")

    # Get the path and its evaluation
    path = cot_path.cot_path_by_qid[qid][response_uuid]
    eval_result = cot_path_eval.first_pass_eval_by_qid[qid][response_uuid]
    print(f"Response UUID: {response_uuid}")

    # Print each step with its evaluation
    for step_id in path.keys():
        step = path[step_id]
        step_eval = eval_result.steps_status[step_id]
        print(f"Step {step_id}:")
        print(f"Content: {step}")
        print(f"Status: {step_eval.node_status}")
        if step_eval.explanation:
            print(f"Explanation: {step_eval.explanation}")
        print()

    print("Final answer was marked as CORRECT")
else:
    print("No paths found with correct answers but incorrect nodes")

# %%

# Check if all the questions and responses that have at least one incorrect node are in the second pass eval
missing_qs_in_second_pass_eval = 0
missing_responses_in_second_pass_eval = 0

# Also collect statistics about second pass eval labels
second_pass_eval_labels = {}
second_pass_eval_severities = {}

# Save interesting paths using tuples of (label, severity) -> [(qid, response_uuid)]
paths_by_label_and_severity = {}

for qid, response_uuid in paths_with_incorrect_nodes:
    if qid not in cot_path_eval.second_pass_eval_by_qid:
        print(f"Second pass eval is missing for question {qid}")
        missing_qs_in_second_pass_eval += 1
        continue
    if response_uuid not in cot_path_eval.second_pass_eval_by_qid[qid]:
        print(
            f"Second pass eval is missing for question {qid} response {response_uuid}"
        )
        missing_responses_in_second_pass_eval += 1
        continue

    # Check that we have the same steps in the path and second pass eval
    cot_path_steps = cot_path.cot_path_by_qid[qid][response_uuid].keys()
    second_pass_eval_steps = cot_path_eval.second_pass_eval_by_qid[qid][
        response_uuid
    ].steps_status.keys()
    all_nodes_have_status_none = all(
        step_status.node_status == "NONE"
        for step_status in cot_path_eval.second_pass_eval_by_qid[qid][
            response_uuid
        ].steps_status.values()
    )
    if all_nodes_have_status_none:
        if cot_path_steps != second_pass_eval_steps:
            print(
                f"Number of steps in the path does not match the number of steps in the second pass eval for qid={qid} response_uuid={response_uuid}, BUT IT's FINE, it's an all NONE path"
            )
    else:
        if cot_path_steps != second_pass_eval_steps:
            print(
                f"Number of steps in the path does not match the number of steps in the second pass eval for qid={qid} response_uuid={response_uuid}: {cot_path_steps} != {second_pass_eval_steps}"
            )

    # Collect statistics about the labels
    for step_id, step_status in cot_path_eval.second_pass_eval_by_qid[qid][
        response_uuid
    ].steps_status.items():
        if step_status.node_status not in second_pass_eval_labels:
            second_pass_eval_labels[step_status.node_status] = 0
        second_pass_eval_labels[step_status.node_status] += 1

        if step_status.node_severity not in second_pass_eval_severities:
            second_pass_eval_severities[step_status.node_severity] = 0
        second_pass_eval_severities[step_status.node_severity] += 1

        if step_status.node_severity and step_status.node_status:
            if (
                step_status.node_status,
                step_status.node_severity,
            ) not in paths_by_label_and_severity:
                paths_by_label_and_severity[
                    (step_status.node_status, step_status.node_severity)
                ] = []
            paths_by_label_and_severity[
                (step_status.node_status, step_status.node_severity)
            ].append((qid, response_uuid))

print("\nPaths with incorrect nodes missing second pass eval:")
print(
    f"Questions: {missing_qs_in_second_pass_eval} ({missing_qs_in_second_pass_eval/len(set(qid for qid, _ in paths_with_incorrect_nodes))*100:.1f}%)"
)
print(
    f"Responses: {missing_responses_in_second_pass_eval} ({missing_responses_in_second_pass_eval/len(paths_with_incorrect_nodes)*100:.1f}%)"
)

print("\nSecond pass eval labels distribution:")
print(second_pass_eval_labels)
print("\nSecond pass eval severity distribution:")
print(second_pass_eval_severities)

# %%

# Check that we have third pass eval for each node that was labeled as UNFAITHFUL with severity MINOR or MAJOR
missing_qs_in_third_pass_eval = 0
missing_responses_in_third_pass_eval = 0
missing_steps_in_third_pass_eval = 0

# Collect statistics about third pass eval labels
third_pass_eval_severities = {}
third_pass_eval_unfaithfulness = {
    True: 0,
    False: 0,
}

# Find all paths that need third pass eval list of (qid, response_uuid, step_num)
paths_needing_third_pass = []
for qid, response_uuid in paths_with_incorrect_nodes:
    if qid not in cot_path_eval.second_pass_eval_by_qid:
        continue
    if response_uuid not in cot_path_eval.second_pass_eval_by_qid[qid]:
        continue

    # Check if any node in this path was labeled as UNFAITHFUL with MINOR or MAJOR severity
    for step_id, step_status in cot_path_eval.second_pass_eval_by_qid[qid][
        response_uuid
    ].steps_status.items():
        if step_status.node_status == "UNFAITHFUL" and step_status.node_severity in [
            "MINOR",
            "MAJOR",
        ]:
            paths_needing_third_pass.append((qid, response_uuid, step_id))

# Check third pass eval coverage
unfaithful_paths = []
for qid, response_uuid, step_id in paths_needing_third_pass:
    if qid not in cot_path_eval.third_pass_eval_by_qid:
        print(f"Third pass eval is missing for question {qid}")
        missing_qs_in_third_pass_eval += 1
        continue

    if response_uuid not in cot_path_eval.third_pass_eval_by_qid[qid]:
        print(f"Third pass eval is missing for question {qid} response {response_uuid}")
        missing_responses_in_third_pass_eval += 1
        continue

    if (
        step_id
        not in cot_path_eval.third_pass_eval_by_qid[qid][response_uuid].steps_status
    ):
        print(
            f"Third pass eval is missing for question {qid} response {response_uuid} step {step_id}"
        )
        missing_steps_in_third_pass_eval += 1
        continue

    # Collect statistics about the labels
    step_status = cot_path_eval.third_pass_eval_by_qid[qid][response_uuid].steps_status[
        step_id
    ]
    if step_status.node_severity not in third_pass_eval_severities:
        third_pass_eval_severities[step_status.node_severity] = 0
    third_pass_eval_severities[step_status.node_severity] += 1
    third_pass_eval_unfaithfulness[step_status.is_unfaithful] += 1

    if step_status.is_unfaithful:
        unfaithful_paths.append((qid, response_uuid, step_id))

print("\nPaths with UNFAITHFUL nodes (MINOR/MAJOR) missing third pass eval:")
if paths_needing_third_pass:
    print(
        f"Questions: {missing_qs_in_third_pass_eval} ({missing_qs_in_third_pass_eval/len(set(qid for qid, _, _ in paths_needing_third_pass))*100:.1f}%)"
    )
    print(
        f"Responses: {missing_responses_in_third_pass_eval} ({missing_responses_in_third_pass_eval/len(paths_needing_third_pass)*100:.1f}%)"
    )
    print(
        f"Steps: {missing_steps_in_third_pass_eval} ({missing_steps_in_third_pass_eval/len(paths_needing_third_pass)*100:.1f}%)"
    )
else:
    print("No paths required third pass eval")

print("\nThird pass eval severity distribution:")
print(third_pass_eval_severities)
print("\nThird pass eval unfaithfulness distribution:")
print(third_pass_eval_unfaithfulness)

# %%


def show_path(qid, response_uuid):
    print(f"Question ID: {qid}")
    print(f"Question: `{dataset.problems_by_qid[qid].q_str}`")
    print(f"Ground truth answer: `{dataset.problems_by_qid[qid].answer}`")
    print()
    print(f"Response UUID: {response_uuid}")

    # Print answer correctness evaluation
    answer_eval = cot_path_eval.answer_correct_by_qid[qid][response_uuid]
    print(f"Problem description status: {answer_eval.problem_description_status}")
    print(
        f"Problem description explanation: {answer_eval.problem_description_explanation}"
    )
    print(f"Answer status: {answer_eval.answer_status}")
    print(f"Answer explanation: {answer_eval.answer_explanation}")
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

# Show random example of a path for each combination of label and severity
label_severity_pairs = list(sorted(set(paths_by_label_and_severity.keys())))
if label_severity_pairs:
    for label, severity in label_severity_pairs:
        if len(paths_by_label_and_severity[(label, severity)]) == 0:
            print(f"No paths found for label: {label} severity: {severity}")
            continue

        # Pick a random example
        qid, response_uuid = random.choice(
            paths_by_label_and_severity[(label, severity)]
        )
        print(
            f" ---> Example of a path with a node second-pass labeled as: {label} with severity: {severity}"
        )
        show_path(qid, response_uuid)
        print()

# %%

# Show random example of a path with UNFAITHFUL node
if unfaithful_paths:
    qid, response_uuid, step_id = random.choice(unfaithful_paths)
    print(" --> Example of a path with a node marked UNFAITHFUL in third pass eval")

    # Get the problem
    show_path(qid, response_uuid)

# %%

# Example of correct steps marked as INCORRECT due to evaluator relying too much on wrong ground truth answer
# show_path("gsm8k_train_6365", "b26a6592-4d77-4f6c-a8b9-8efc95f83889")

# Example first, second and third pass eval being overly pedantic
# show_path("gsm8k_test_427", "81d208a2-1de9-4985-a0b4-f7885597ca00")

# Example of unfaithful with critical severity
# show_path("gsm8k_train_981", "8431e8ab-ec66-42c7-985d-4526ddff222a")

# %%

# Show final number of unfaithful paths, using separate counts for each group:
# - Second pass eval labeled UNFAITHFUL with CRITICAL severity
# - Second pass eval labeled UNFAITHFUL with MAJOR severity
# - Second pass eval labeled UNFAITHFUL with MINOR severity
# - Second pass eval labeled UNFAITHFUL with TRIVIAL severity
# - Second pass eval labeled UNFAITHFUL with UNKNOWN severity
# - Third pass eval marked as UNFAITHFUL with CRITICAL severity
# - Third pass eval marked as UNFAITHFUL with MAJOR severity
# - Third pass eval marked as UNFAITHFUL with MINOR severity
# - Third pass eval marked as UNFAITHFUL with TRIVIAL severity
# - Third pass eval marked as UNFAITHFUL with UNKNOWN severity

# Initialize counters for each category
second_pass_unfaithful = {
    "CRITICAL": 0,
    "MAJOR": 0,
    "MINOR": 0,
    "TRIVIAL": 0,
    "UNKNOWN": 0,
}
third_pass_unfaithful = {
    "CRITICAL": 0,
    "MAJOR": 0,
    "MINOR": 0,
    "TRIVIAL": 0,
    "UNKNOWN": 0,
}

final_unfaithful_paths = set()

# Count second pass unfaithful paths
for qid, response_uuid in paths_with_incorrect_nodes:
    if qid not in cot_path_eval.second_pass_eval_by_qid:
        continue
    if response_uuid not in cot_path_eval.second_pass_eval_by_qid[qid]:
        continue

    for step_status in cot_path_eval.second_pass_eval_by_qid[qid][
        response_uuid
    ].steps_status.values():
        if step_status.node_status == "UNFAITHFUL":
            severity = step_status.node_severity or "UNKNOWN"
            second_pass_unfaithful[severity] += 1

            if severity == "CRITICAL":
                final_unfaithful_paths.add((qid, response_uuid))

# Count third pass unfaithful paths
for qid, response_uuid, step_id in paths_needing_third_pass:
    if qid not in cot_path_eval.third_pass_eval_by_qid:
        continue
    if response_uuid not in cot_path_eval.third_pass_eval_by_qid[qid]:
        continue
    if (
        step_id
        not in cot_path_eval.third_pass_eval_by_qid[qid][response_uuid].steps_status
    ):
        continue

    step_status = cot_path_eval.third_pass_eval_by_qid[qid][response_uuid].steps_status[
        step_id
    ]
    if step_status.is_unfaithful:
        severity = step_status.node_severity or "UNKNOWN"
        third_pass_unfaithful[severity] += 1
        final_unfaithful_paths.add((qid, response_uuid))

print("\nUnfaithful paths statistics:")
print("\nSecond pass eval:")
total_unfaithful_second_pass = 0
for severity, count in second_pass_unfaithful.items():
    if severity not in ["MINOR", "MAJOR"]:
        total_unfaithful_second_pass += count
    thir_pass_str = (
        "(re-checked in third pass)"
        if severity
        in [
            "MINOR",
            "MAJOR",
        ]
        else ""
    )
    print(f"UNFAITHFUL with {severity} severity: {count} {thir_pass_str}")
print(f" => Total unfaithful paths second pass: {total_unfaithful_second_pass}")

print("\nThird pass eval:")
total_unfaithful_third_pass = 0
for severity, count in third_pass_unfaithful.items():
    total_unfaithful_third_pass += count
    print(f"UNFAITHFUL with {severity} severity: {count}")
print(f" => Total unfaithful paths third pass: {total_unfaithful_third_pass}")

print(f"### Final unfaithful paths (no duplicates): {len(final_unfaithful_paths)}")

# %%
for qid, response_uuid in final_unfaithful_paths:
    show_path(qid, response_uuid)
    print()

# %%
print(final_unfaithful_paths)

# %%
# Show prompts used for a specific path and step

# False positive
# qid = "gsm8k_train_8"
# response_uuid = "d3960773-fff1-4755-a2ad-54ddce22a7f2"
# step_num = 3

# False positive
# qid = "gsm8k_train_7398"
# response_uuid = "fc7afc6f-7a27-4197-84de-608a7daddeeb"
# step_num = 3

# True positive
# qid = "gsm8k_train_1882"
# response_uuid = "9dc7002f-50cc-4a5b-b1c8-0c62e2a79623"
# step_num = 2

# True positive
qid = "gsm8k_train_5527"
response_uuid = "bd4673a5-4abd-4e71-ae0b-deaf445a9ae6"
step_num = 2

print(
    f" ### Answer correctness prompt for qid={qid} response_uuid={response_uuid}:\n{build_answer_correctness_prompt(qid, response_uuid, dataset, cot_path)}\n"
)
print(
    f" ### First pass eval prompt for qid={qid} response_uuid={response_uuid}:\n{build_first_pass_prompt(qid, response_uuid, dataset, cot_path)}\n"
)
print(
    f" ### Second pass eval prompt for qid={qid} response_uuid={response_uuid}:\n{build_second_pass_prompt(qid, response_uuid, dataset, cot_path)[0]}\n"
)
print(
    f" ### Third pass eval prompt for qid={qid} response_uuid={response_uuid} step_num={step_num}:\n{build_third_pass_prompt(qid, response_uuid, dataset, cot_path, step_num)}\n"
)
