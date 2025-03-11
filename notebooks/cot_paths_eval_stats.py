# %%W

from chainscope.typing import CoTPath, CoTPathEval, ProblemDataset
from chainscope.utils import MODELS_MAP

# %%

model_aliases = ["GPT4O", "C3.5S", "DSV3", "GP1.5", "L70"]
dataset_ids = ["gsm8k", "math", "mmlu"]

# %%

# Load datasets
datasets: dict[str, ProblemDataset] = {}
for dataset_id in dataset_ids:
    datasets[dataset_id] = ProblemDataset.load(dataset_id)

cot_paths: dict[tuple[str, str], CoTPath] = {}
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path = CoTPath.load(model_id, dataset_id)
        cot_paths[(model_id, dataset_id)] = cot_path

cot_paths_evals: dict[tuple[str, str], CoTPathEval] = {}
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        cot_path_eval = CoTPathEval.load(model_id, dataset_id)
        cot_paths_evals[(model_id, dataset_id)] = cot_path_eval

# %%

unfaithful_responses: dict[
    tuple[str, str], list[tuple[str, str, list[int]]]
] = {}  # (model_id, dataset_id) -> [(qid, response_uuid, [step_num])]

correct_responses: dict[tuple[str, str], int] = {}

for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        print(f" #### Processing model {model_alias} on {dataset_id}:")
        unfaithful_responses[(model_id, dataset_id)] = []
        correct_responses[(model_id, dataset_id)] = 0

        cot_path_eval = cot_paths_evals[(model_id, dataset_id)]

        # Count how many responses have correct answer:
        for qid in cot_path_eval.answer_correct_by_qid.keys():
            if len(cot_path_eval.answer_correct_by_qid[qid]) > 1:
                print(f"More than 1 response for qid {qid} in answer correct eval")
                continue
            if len(cot_path_eval.answer_correct_by_qid[qid]) == 0:
                print(f"No response for qid {qid} in answer correct eval")
                continue

            response_uuid = list(cot_path_eval.answer_correct_by_qid[qid].keys())[0]
            if (
                cot_path_eval.answer_correct_by_qid[qid][response_uuid].answer_status
                == "CORRECT"
            ):
                correct_responses[(model_id, dataset_id)] += 1

        # Count how many responses have at least one unfaithful node in third pass eval
        for qid in cot_path_eval.third_pass_eval_by_qid.keys():
            if len(cot_path_eval.third_pass_eval_by_qid[qid]) > 1:
                print(f"More than 1 response for qid {qid} in third pass eval")
                continue
            if len(cot_path_eval.third_pass_eval_by_qid[qid]) == 0:
                # This is completely normal and it can happen for many reasons:
                # - No incorrect steps in first pass
                # - No unfaithful steps in second pass
                continue

            response_uuid = list(cot_path_eval.third_pass_eval_by_qid[qid].keys())[0]
            unfaithful_steps = []
            for step_num, step_status in cot_path_eval.third_pass_eval_by_qid[qid][
                response_uuid
            ].steps_status.items():
                if step_status.is_unfaithful:
                    unfaithful_steps.append(step_num)
            if unfaithful_steps:
                unfaithful_responses[(model_id, dataset_id)].append(
                    (qid, response_uuid, unfaithful_steps)
                )

# %%

# Print the number of unfaithful responses over correct responses for each model and dataset
for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        unf_cnt = len(unfaithful_responses[(model_id, dataset_id)])
        correct_cnt = correct_responses[(model_id, dataset_id)]
        pct = unf_cnt / correct_cnt if correct_cnt > 0 else 0
        print(
            f"Unfaithful responses for {model_alias} on {dataset_id}: {unf_cnt} ({pct:.2%} of {correct_cnt} correct)"
        )
# %%


def show_path_with_unfaithful_step(
    qid, response_uuid, unfaithful_steps, model_alias, dataset_id
):
    model_id = MODELS_MAP[model_alias]
    cot_path = cot_paths[(model_id, dataset_id)]
    cot_path_eval = cot_paths_evals[(model_id, dataset_id)]

    print(f"Response ID: {response_uuid}\n")

    print(f"Question ID: {qid}")
    print(f"\\textbf{{Question}}: {datasets[dataset_id].problems_by_qid[qid].q_str}\n")

    response_str = f"\\textbf{{{model_alias}}}: \n\n"
    for step_id in cot_path.cot_path_by_qid[qid][response_uuid].keys():
        step = cot_path.cot_path_by_qid[qid][response_uuid][step_id]
        step = step.replace("\n", "\n\n")
        if step_id in unfaithful_steps:
            # Reason for unfaithfulness:
            step += "\n\n\\vspace{0.1cm}\n"
            step += f"\\hspace{{1cm}}\\textit{{# {cot_path_eval.third_pass_eval_by_qid[qid][response_uuid].steps_status[step_id].explanation}}}"
            step += "\n\\vspace{0.1cm}"
        response_str += f"{step}\n\n"
    # Remove the last newline
    response_str = response_str[:-1]

    response_str = response_str.replace("×", "$\\times$")

    print(response_str)


# %%

for model_alias in model_aliases:
    model_id = MODELS_MAP[model_alias]
    for dataset_id in dataset_ids:
        print(f" #### Unfaithful responses for {model_alias} on {dataset_id}:")
        unf_resps = unfaithful_responses[(model_id, dataset_id)]
        for qid, response_uuid, unfaithful_steps in unf_resps:
            show_path_with_unfaithful_step(
                qid, response_uuid, unfaithful_steps, model_alias, dataset_id
            )
            print()
            print("-" * 100)
            print()

# %%
