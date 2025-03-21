# %%
import json

import pandas as pd

from chainscope.typing import *

# %%

# Cache for responses and evaluations
_responses_cache: dict[Path, CotResponses] = {}
_evals_cache: dict[tuple[str, str, str, str], CotEval] = {}
_faithfulness_cache: dict[Path, Any] = {}
_qs_dataset_cache: dict[Path, QsDataset] = {}

# %%

model_id = "anthropic/claude-3.5-haiku"

# %%

# Load the data
df = pd.read_pickle(DATA_DIR / "df-wm.pkl")
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

df = df[df["mode"] == "cot"]

# Filter by model
df = df[df["model_id"] == model_id]

assert (
    len(df) > 0
), f"No data found, models are: {pd.read_pickle(DATA_DIR / 'df-wm.pkl')['model_id'].unique()}"


# %%
# Function to load responses and eval for a row
def load_responses_and_eval(
    row,
) -> tuple[
    str,
    str,
    dict[str, MathResponse | AtCoderResponse | str],
    dict[str, CotEvalResult],
    Literal["YES", "NO"],
]:
    q_id = row["qid"]

    dataset_params = DatasetParams(
        prop_id=row["prop_id"],
        comparison=row["comparison"],
        answer=row["answer"],
        max_comparisons=1,
        uuid=row["dataset_id"].split("_")[-1],
    )

    expected_answer = dataset_params.answer

    sampling_params = SamplingParams(
        temperature=float(row["temperature"]),
        top_p=float(row["top_p"]),
        max_new_tokens=int(row["max_new_tokens"]),
    )

    # Construct response file path
    response_path = (
        DATA_DIR
        / "cot_responses"
        / row["instr_id"]
        / sampling_params.id
        / dataset_params.pre_id
        / dataset_params.id
        / f"{row['model_id'].replace('/', '__')}.yaml"
    )

    # Load responses from cache or disk
    if response_path not in _responses_cache:
        _responses_cache[response_path] = CotResponses.load(response_path)
    responses = _responses_cache[response_path]

    # Create cache key for evaluations
    eval_cache_key = (
        row["instr_id"],
        row["model_id"],
        dataset_params.id,
        sampling_params.id,
    )

    # Load evaluations from cache or disk
    if eval_cache_key not in _evals_cache:
        _evals_cache[eval_cache_key] = dataset_params.load_cot_eval(
            row["instr_id"],
            row["model_id"],
            sampling_params,
        )
    cot_eval = _evals_cache[eval_cache_key]

    # Load dataset
    qs_dataset_path = dataset_params.qs_dataset_path
    if qs_dataset_path not in _qs_dataset_cache:
        _qs_dataset_cache[qs_dataset_path] = QsDataset.load_from_path(qs_dataset_path)
    qs_dataset = _qs_dataset_cache[qs_dataset_path]
    q_str = qs_dataset.question_by_qid[q_id].q_str

    return (
        q_id,
        q_str,
        responses.responses_by_qid[q_id],
        cot_eval.results_by_qid[q_id],
        expected_answer,
    )

# %%
# Load IPHR faithfulness data

faithfulness_files = list(
    DATA_DIR.glob(f"faithfulness/{model_id.split('/')[-1]}/*.yaml")
)

unfaithful_q_ids = []
p_correct_by_qid = {}
reversed_q_ids = {}
for file in faithfulness_files:
    if file not in _faithfulness_cache:
        with open(file, "r") as f:
            file_data = yaml.safe_load(f)
        _faithfulness_cache[file] = file_data

    file_data = _faithfulness_cache[file]
    unfaithful_q_ids.extend(file_data.keys())
    unfaithful_q_ids.extend(
        [item["metadata"]["reversed_q_id"] for item in file_data.values()]
    )
    for key, item in file_data.items():
        p_correct_by_qid[key] = item["metadata"]["p_correct"]
        p_correct_by_qid[item["metadata"]["reversed_q_id"]] = item["metadata"][
            "reversed_q_p_correct"
        ]
        reversed_q_ids[key] = item["metadata"]["reversed_q_id"]
        reversed_q_ids[item["metadata"]["reversed_q_id"]] = key

# %%

assert len(unfaithful_q_ids) > 0, "No unfaithful data found"

print(f"Number of unfaithful q_ids: {len(unfaithful_q_ids)}")

# Example of unfaithful q_id:
print(f" ## Example of unfaithful q_id: {unfaithful_q_ids[0]}.")
print(f"P_correct: {p_correct_by_qid[unfaithful_q_ids[0]]}")
print(f"Reversed q_id: {reversed_q_ids[unfaithful_q_ids[0]]}")
print(f"P_correct (reversed): {p_correct_by_qid[reversed_q_ids[unfaithful_q_ids[0]]]}")

# %%

# Find the pair of unfaithful qs with largest difference in p_correct

# Create sorted pairs of questions
unfaithful_qids_pairs = []
seen_pairs = set()
for q_id in unfaithful_q_ids:
    rev_q_id = reversed_q_ids[q_id]
    if (
        rev_q_id in p_correct_by_qid
        and q_id not in seen_pairs
        and rev_q_id not in seen_pairs
    ):
        pair = tuple(sorted([q_id, rev_q_id]))  # Sort to ensure consistent ordering
        diff = abs(p_correct_by_qid[q_id] - p_correct_by_qid[rev_q_id])
        unfaithful_qids_pairs.append((pair, diff))
        seen_pairs.add(q_id)
        seen_pairs.add(rev_q_id)

# Sort by difference in p_correct
unfaithful_qids_pairs.sort(key=lambda x: x[1], reverse=True)
print(f"Number of unfaithful qids pairs: {len(unfaithful_qids_pairs)}")

# %%

# Load the instruction-wm prompt
wm_template = Instructions.load("instr-wm").cot

# %%

# Print the top K pairs with largest difference
K = 100
for (qid1, qid2), acc_diff in unfaithful_qids_pairs[:K]:
    q1_answer = df.loc[df["qid"] == qid1, "answer"].values[0]
    q2_answer = df.loc[df["qid"] == qid2, "answer"].values[0]
    q1_acc = p_correct_by_qid[qid1]
    q2_acc = p_correct_by_qid[qid2]
    
    q1_str = df.loc[df["qid"] == qid1, "q_str"].values[0]
    q2_str = df.loc[df["qid"] == qid2, "q_str"].values[0]
    q1_prompt = wm_template.format(question=q1_str)[:-1]
    q2_prompt = wm_template.format(question=q2_str)[:-1]

    dataset_id = df.loc[df["qid"] == qid1, "dataset_id"].values[0]
    x_name = df.loc[df["qid"] == qid1, "x_name"].values[0]
    y_name = df.loc[df["qid"] == qid1, "y_name"].values[0]

    print(f"Dataset id: {dataset_id}")
    print(f"First entity: {x_name}")
    print(f"Second entity: {y_name}\n")

    print("First prompt:\n")
    print(f"`{q1_prompt}`\n")
    print(f"Expected answer: {q1_answer}")
    print(f"Model's accuracy: {q1_acc:.3f}")
    print("Example output: ")

    print("\nSecond prompt:\n")
    print(f"`{q2_prompt}`\n")
    print(f"Expected answer: {q2_answer}")
    print(f"Model's accuracy: {q2_acc:.3f}")
    print("Example output: ")

    print("-" * 100)
# %%

# build a JSON with the data we printed above
model_unfaithful_pairs_data = []
for (qid1, qid2), acc_diff in unfaithful_qids_pairs:
    q1_answer = df.loc[df["qid"] == qid1, "answer"].values[0]
    q2_answer = df.loc[df["qid"] == qid2, "answer"].values[0]
    q1_acc = p_correct_by_qid[qid1]
    q2_acc = p_correct_by_qid[qid2]
    
    q1_str = df.loc[df["qid"] == qid1, "q_str"].values[0]
    q2_str = df.loc[df["qid"] == qid2, "q_str"].values[0]
    q1_prompt = wm_template.format(question=q1_str)[:-1]
    q2_prompt = wm_template.format(question=q2_str)[:-1]

    dataset_id = df.loc[df["qid"] == qid1, "dataset_id"].values[0]
    x_name = df.loc[df["qid"] == qid1, "x_name"].values[0]
    y_name = df.loc[df["qid"] == qid1, "y_name"].values[0]

    model_unfaithful_pairs_data.append({
        "dataset_id": dataset_id,
        "first_entity": x_name,
        "second_entity": y_name,
        "q1_str": q1_str.split("\n\n")[1],
        "q2_str": q2_str.split("\n\n")[1],
        "q1_prompt": q1_prompt,
        "q2_prompt": q2_prompt,
        "q1_answer": q1_answer,
        "q2_answer": q2_answer,
        "q1_acc": q1_acc,
        "q2_acc": q2_acc,
        "acc_diff": acc_diff,
    })

print(f"Number of unfaithful pairs: {len(model_unfaithful_pairs_data)}")

# save the JSON
model_name = model_id.split("/")[-1]
with open(f"unfaithful_pairs_data_{model_name}.json", "w") as f:
    json.dump(model_unfaithful_pairs_data, f)


# %%
