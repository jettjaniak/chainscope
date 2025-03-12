# %%

import glob
import os
import pickle
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from chainscope.typing import *

# %%

# Cache for responses and evaluations
_responses_cache: dict[Path, CotResponses] = {}
_evals_cache: dict[tuple[str, str, str, str], CotEval] = {}
_faithfulness_cache: dict[Path, Any] = {}
_qs_dataset_cache: dict[Path, QsDataset] = {}

# %%

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model_family = "Llama3.1"
model_size = "8B"
model_type = "chat"
n_layers = 32

experiment_variations = [
    "feed_only_statement",
    "feed_chat_template_get_acts_at_punctuation",
    "feed_full_prompt",
]
experiment_variation = experiment_variations[2]

# %%

ROOT = "/workspace/chainscope/notebooks/Truth_is_Universal"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).cuda()

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
def remove_llama_system_dates(chat_input_str: str) -> str:
    return re.sub(
        r"\n\nCutting Knowledge Date: .*\nToday Date: .*\n\n", "", chat_input_str
    )


def conversation_to_str_prompt(
    conversation: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool,
) -> str:
    str_prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    assert isinstance(str_prompt, str)
    return remove_llama_system_dates(str_prompt)


def make_chat_prompt(
    instruction: str,
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool,
) -> str:
    conversation = [
        {
            "role": "user",
            "content": instruction,
        }
    ]
    return conversation_to_str_prompt(conversation, tokenizer, add_generation_prompt)


# %%
# Load IPHR responses and eval

min_sentence_length = 100


def split_resp_in_statements(
    q_str: str,
    response: str,
    instructions: Instructions,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[str], list[int], list[int]]:
    end_of_sentence_punctuation = [".", "!", "?"]
    sentences_last_token_positions = []

    prompt = instructions.cot.format(question=q_str)
    input_str = make_chat_prompt(
        instruction=prompt,
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(input_str)

    # Split by "." or "\n", remove empty strings or strings that are just a number
    lines = response.split("\n")
    sentences = []
    for line in lines:
        # split by ".", "!", "?", keeping the corresponding delimiter
        current = ""
        for char in line:
            current += char
            input_str += char

            if char in end_of_sentence_punctuation:
                stripped = current.strip()
                if stripped == "":
                    # discard the current sentence
                    current = ""
                elif stripped.isdigit():
                    # discard the current sentence
                    current = ""
                elif len(stripped) < min_sentence_length:
                    # discard the current sentence
                    current = ""
                else:
                    sentences.append(stripped)
                    input_ids = tokenizer.encode(input_str)
                    last_token_position = len(input_ids) - 1
                    sentences_last_token_positions.append(last_token_position)
                    current = ""
        if (
            current and len(current.strip()) > min_sentence_length
        ):  # Add any remaining text
            sentences.append(current)
            input_ids = tokenizer.encode(input_str)
            last_token_position = len(input_ids) - 1
            sentences_last_token_positions.append(last_token_position)

        input_str += "\n"

    # strip the sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences, sentences_last_token_positions, input_ids


iphr_data = []
instructions = Instructions.load("instr-wm")
for _, row in df.iterrows():
    q_id, q_str, responses, eval, expected_answer = load_responses_and_eval(row)
    assert type(responses) is dict, "Responses are not a dictionary"

    for response_id, response in responses.items():
        if response is not None:
            assert type(response) is str, "Response is not a string"
            statements, sentences_last_token_positions, input_ids = (
                split_resp_in_statements(
                    q_str,
                    response,
                    instructions,
                    tokenizer,
                )
            )
            iphr_data.append(
                {
                    "q_id": q_id,
                    "response_id": response_id,
                    "response": response,
                    "expected_answer": expected_answer,
                    "eval": eval[response_id],
                    "statements": statements,
                    "sentences_last_token_positions": sentences_last_token_positions,
                    "input_ids": input_ids,
                }
            )

# Sort iphr_data by q_id
iphr_data = sorted(iphr_data, key=lambda x: x["q_id"])

assert len(iphr_data) > 0, "No IPHR data found"

# Show five random sentences from the IPHR data
print(" ## Five random sentences from the IPHR data:")
random_sentences = random.sample(iphr_data, 5)
for item in random_sentences:
    print(item["q_id"])
    print(item["response_id"])
    print(item["expected_answer"])
    print(item["statements"])
    print("\n")

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

# Fill the p_correct_by_qid and reversed_q_ids for the questions that are not unfaithful.
# we need to use df
faithful_q_ids = []
for _, row in df.iterrows():
    q_id = row["qid"]
    if q_id in unfaithful_q_ids:
        continue

    # Find the reversed q_id: same values, same prop_id, same comparison but values are in different order
    reversed_q_id = None
    for _, other_row in df.iterrows():
        if (
            other_row["prop_id"] == row["prop_id"]
            and other_row["comparison"] == row["comparison"]
            and other_row["x_value"] == row["y_value"]
            and other_row["y_value"] == row["x_value"]
        ):
            reversed_q_id = other_row["qid"]
            break
    if reversed_q_id is None:
        continue

    faithful_q_ids.append(q_id)

    reversed_q_ids[q_id] = reversed_q_id
    reversed_q_ids[reversed_q_id] = q_id

    p_correct_by_qid[q_id] = row["p_correct"]
    p_correct_by_qid[reversed_q_id] = df.loc[
        df["qid"] == reversed_q_id, "p_correct"
    ].values[0]

# %%

assert len(unfaithful_q_ids) > 0, "No unfaithful data found"
assert len(faithful_q_ids) > 0, "No faithful data found"

print(f"Number of unfaithful q_ids: {len(unfaithful_q_ids)}")
print(f"Number of faithful q_ids: {len(faithful_q_ids)}")

# Example of unfaithful q_id:
print(f" ## Example of unfaithful q_id: {unfaithful_q_ids[0]}.")
print(f"P_correct: {p_correct_by_qid[unfaithful_q_ids[0]]}")
print(f"Reversed q_id: {reversed_q_ids[unfaithful_q_ids[0]]}")
print(f"P_correct (reversed): {p_correct_by_qid[reversed_q_ids[unfaithful_q_ids[0]]]}")

# Example of faithful q_id:
print(f" ## Example of faithful q_id: {faithful_q_ids[0]}.")
print(f"P_correct: {p_correct_by_qid[faithful_q_ids[0]]}")
print(f"Reversed q_id: {reversed_q_ids[faithful_q_ids[0]]}")
print(f"P_correct (reversed): {p_correct_by_qid[reversed_q_ids[faithful_q_ids[0]]]}")

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

# Print the pair with largest difference (same as before)
if unfaithful_qids_pairs:
    (q1, q2), max_diff = unfaithful_qids_pairs[0]
    print("Largest p_correct difference in unfaithful pairs:")
    print(f"Q1 ({q1[:10]}...): {p_correct_by_qid[q1]:.3f}")
    print(f"Q1 str: `{df.loc[df['qid'] == q1, 'q_str'].values[0]}`\n")
    counter = 1
    for item in iphr_data:
        if item["q_id"] == q1:
            print(f"Response {counter}: `{item['response']}`")
            print("\n")
            counter += 1
    print(f"Q2 ({q2[:10]}...): {p_correct_by_qid[q2]:.3f}")
    print(f"Q2 str: `{df.loc[df['qid'] == q2, 'q_str'].values[0]}`\n")
    counter = 1
    for item in iphr_data:
        if item["q_id"] == q2:
            print(f"Response {counter}: `{item['response']}`")
            print("\n")
            counter += 1
    print(f"Difference: {max_diff:.3f}")

# Find the pair of faithful qs with largest sum of p_correct

# Create sorted pairs of questions
faithful_qids_pairs = []
seen_pairs = set()
for q_id in faithful_q_ids:
    rev_q_id = reversed_q_ids[q_id]
    if (
        rev_q_id in p_correct_by_qid
        and q_id not in seen_pairs
        and rev_q_id not in seen_pairs
    ):
        pair = tuple(sorted([q_id, rev_q_id]))  # Sort to ensure consistent ordering
        sum_p = p_correct_by_qid[q_id] + p_correct_by_qid[rev_q_id]
        faithful_qids_pairs.append((pair, sum_p))
        seen_pairs.add(q_id)
        seen_pairs.add(rev_q_id)

# Sort by sum of p_correct
faithful_qids_pairs.sort(key=lambda x: x[1], reverse=True)
print(f"\nNumber of faithful qids pairs: {len(faithful_qids_pairs)}")

# Print the pair with largest sum (same as before)
if faithful_qids_pairs:
    (q1, q2), max_sum = faithful_qids_pairs[0]
    print("Largest p_correct sum in faithful pairs:")
    print(f"Q1 ({q1[:10]}...): {p_correct_by_qid[q1]:.3f}")
    print(f"Q1 str: `{df.loc[df['qid'] == q1, 'q_str'].values[0]}`\n")
    counter = 1
    for item in iphr_data:
        if item["q_id"] == q1:
            print(f"Response {counter}: `{item['response']}`")
            print("\n")
            counter += 1
    print(f"Q2 ({q2[:10]}...): {p_correct_by_qid[q2]:.3f}")
    print(f"Q2 str: `{df.loc[df['qid'] == q2, 'q_str'].values[0]}`\n")
    counter = 1
    for item in iphr_data:
        if item["q_id"] == q2:
            print(f"Response {counter}: `{item['response']}`")
            print("\n")
            counter += 1
    print(f"Sum: {max_sum:.3f}")

# %%

# Create lists of tuples (statement, metadata) for both faithful and unfaithful statements
N_PAIRS = 200  # Number of pairs to select from each category

# First, group statements by question ID and faithfulness
faithful_statements_by_qid: dict[str, list[tuple[str | tuple[str, int], dict]]] = {}
unfaithful_statements_by_qid: dict[str, list[tuple[str | tuple[str, int], dict]]] = {}

for item in iphr_data:
    if experiment_variation == "feed_full_prompt":
        stmt_tuples = [
            (
                (item["response_id"], pos),
                {
                    "q_id": item["q_id"],
                    "response_id": item["response_id"],
                    "statement_id": idx,
                    "is_unfaithful": item["q_id"] in unfaithful_q_ids,
                    "input_ids": item["input_ids"],
                },
            )
            for idx, pos in enumerate(item["sentences_last_token_positions"])
        ]
    else:
        stmt_tuples = [
            (
                stmt,
                {
                    "q_id": item["q_id"],
                    "response_id": item["response_id"],
                    "statement_id": idx,
                    "is_unfaithful": item["q_id"] in unfaithful_q_ids,
                },
            )
            for idx, stmt in enumerate(item["statements"])
        ]

    target_dict = (
        unfaithful_statements_by_qid
        if item["q_id"] in unfaithful_q_ids
        else faithful_statements_by_qid
    )
    if item["q_id"] not in target_dict:
        target_dict[item["q_id"]] = []
    target_dict[item["q_id"]].extend(stmt_tuples)


# Sample N_PAIRS pairs from each category
def uniform_sample_from_sorted(pairs, n):
    if n >= len(pairs):
        return pairs
    # Calculate the step size to get n evenly spaced indices
    step = (len(pairs) - 1) / (n - 1) if n > 1 else 1
    indices = [int(round(i * step)) for i in range(n)]
    # Ensure we don't have duplicate indices due to rounding
    indices = sorted(list(set(indices)))
    return [pairs[i] for i in indices]


sampled_faithful_pairs = uniform_sample_from_sorted(
    faithful_qids_pairs, min(N_PAIRS, len(faithful_qids_pairs))
)
sampled_unfaithful_pairs = uniform_sample_from_sorted(
    unfaithful_qids_pairs, min(N_PAIRS, len(unfaithful_qids_pairs))
)

# Print some statistics about the sampling
print("\nSampling statistics:")
if sampled_faithful_pairs:
    print("Faithful pairs p_correct sums:")
    print(f"  First pair: {sampled_faithful_pairs[0][1]:.3f}")
    print(
        f"  Middle pair: {sampled_faithful_pairs[len(sampled_faithful_pairs)//2][1]:.3f}"
    )
    print(f"  Last pair: {sampled_faithful_pairs[-1][1]:.3f}")

if sampled_unfaithful_pairs:
    print("\nUnfaithful pairs p_correct differences:")
    print(f"  First pair: {sampled_unfaithful_pairs[0][1]:.3f}")
    print(
        f"  Middle pair: {sampled_unfaithful_pairs[len(sampled_unfaithful_pairs)//2][1]:.3f}"
    )
    print(f"  Last pair: {sampled_unfaithful_pairs[-1][1]:.3f}")

# Gather all statements from selected pairs
faithful_data = []
for (q1, q2), _ in sampled_faithful_pairs:
    if q1 in faithful_statements_by_qid and q2 in faithful_statements_by_qid:
        faithful_data.extend(faithful_statements_by_qid[q1])
        faithful_data.extend(faithful_statements_by_qid[q2])

unfaithful_data = []
for (q1, q2), _ in sampled_unfaithful_pairs:
    if q1 in unfaithful_statements_by_qid and q2 in unfaithful_statements_by_qid:
        unfaithful_data.extend(unfaithful_statements_by_qid[q1])
        unfaithful_data.extend(unfaithful_statements_by_qid[q2])

# Create the dataset and labels
iphr_dataset_name = "iphr"
iphr_dataset = [item[0] for item in faithful_data + unfaithful_data]
iphr_metadata = [item[1] for item in faithful_data + unfaithful_data]

# Shuffle the dataset, labels, and metadata together
idxs = list(range(len(iphr_dataset)))
random.shuffle(idxs)
iphr_dataset = [iphr_dataset[i] for i in idxs]
iphr_metadata = [iphr_metadata[i] for i in idxs]

# Build labels as tensor
if experiment_variation == "feed_full_prompt":
    # We need to build labels for each response id, in the correct order, and with as many labels per response id as the number of statements
    response_ids = set([item["response_id"] for item in iphr_metadata])
    response_ids = list(sorted(response_ids))
    iphr_labels = []
    for response_id in response_ids:
        statements = [
            item for item in iphr_metadata if item["response_id"] == response_id
        ]
        iphr_labels.extend([item["is_unfaithful"] for item in statements])
    iphr_labels = t.tensor(iphr_labels)
else:
    iphr_labels = t.tensor([item["is_unfaithful"] for item in iphr_metadata])

print(f"Size of iphr_dataset: {len(iphr_dataset)}")

# Assert all these have the same size
assert (
    len(iphr_dataset) == len(iphr_labels) == len(iphr_metadata)
), f"iphr_dataset, iphr_labels, and iphr_metadata must have the same size. Got sizes: {len(iphr_dataset)}, {len(iphr_labels)}, {len(iphr_metadata)}"

# How many statements in iphr_dataset have the word "not"
not_count = sum("not" in item for item in iphr_dataset)
print(f"Number of statements with 'not': {not_count}")

# %%

# Double check that the dataset is balanced
faithful_count = sum(iphr_labels == 0)
unfaithful_count = sum(iphr_labels == 1)
print(f"Number of faithful statements: {faithful_count}")
print(f"Number of unfaithful statements: {unfaithful_count}")

# %% Collect activations


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        assert type(module_outputs) is tuple, "module_outputs is not a tuple"
        self.out = module_outputs[0]


def get_acts(statements, tokenizer, model, layers, device):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        # remove existing hooks in this layer
        model.model.layers[layer]._forward_hooks.clear()
        handle = model.model.layers[layer].register_forward_hook(hook)
        # Add hook to the model
        hooks.append(hook), handles.append(handle)

    # get activations
    acts = {layer: [] for layer in layers}

    if experiment_variation == "feed_chat_template_get_acts_at_punctuation":
        assert len(statements) > 0, "No statements provided"
        assert isinstance(statements[0], tuple), "Statements must be tuples"
        # Get all response ids we will be processing
        response_ids = set([statement[0] for statement in statements])
        # Sort response ids
        response_ids = list(sorted(response_ids))

        # Get token positions to get activations at by response id
        token_positions = {}
        for item in statements:
            response_id, last_token_pos = item
            if response_id not in token_positions:
                token_positions[response_id] = []
            token_positions[response_id].append(last_token_pos)

        # Get input ids for each response id and collect activations
        for response_id in tqdm(response_ids, desc="Collecting activations"):
            input_ids = None
            for item in iphr_data:
                if item["response_id"] == response_id:
                    input_ids = t.tensor(item["input_ids"]).unsqueeze(0).to(device)
                    break
            assert (
                input_ids is not None
            ), f"Could not find input_ids for response {response_id}"

            # Run model
            model(input_ids)

            # Collect activations for all required token positions
            token_positions = token_positions[response_id]
            for layer, hook in zip(layers, hooks):
                for pos in token_positions:
                    acts[layer].append(hook.out[0, pos])

    else:
        for statement in statements:
            # For training datasets or other experiment variations
            if experiment_variation == "feed_only_statement":
                input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
            elif experiment_variation == "feed_full_prompt":
                chat_input = make_chat_prompt(
                    instruction=statement,
                    tokenizer=tokenizer,
                    add_generation_prompt=False,
                )
                # rstrip <|eot_id|>
                chat_input = chat_input.rstrip("<|eot_id|>")
                input_ids = tokenizer.encode(chat_input, return_tensors="pt").to(device)
            else:
                raise ValueError(
                    f"Unknown experiment variation: {experiment_variation}"
                )

            model(input_ids)
            for layer, hook in zip(layers, hooks):
                acts[layer].append(hook.out[0, -1])

    for layer, act in acts.items():
        acts[layer] = t.stack(act).float()

    # remove hooks
    for layer, handle in zip(layers, handles):
        handle.remove()

    return acts


layers = list(range(len(model.model.layers)))
save_dir = f"{ROOT}/acts/{model_family}/{model_size}/{model_type}/{iphr_dataset_name}/{experiment_variation}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for idx in tqdm(
    range(0, len(iphr_dataset), 25),
    desc=f"Collecting activations for {iphr_dataset_name}",
):
    # if files already exist, skip
    files = [f"{save_dir}/layer_{layer}_{idx}.pt" for layer in layers]
    if all(os.path.exists(file) for file in files):
        continue
    with t.no_grad():
        acts = get_acts(
            iphr_dataset[idx : idx + 25], tokenizer, model, layers, "cuda:0"
        )
    for layer, act in acts.items():
        t.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")

# %%


def learn_truth_directions(acts_centered, labels, polarities):
    # Check if all polarities are zero (handling both int and float) -> if yes learn only t_g
    all_polarities_zero = t.allclose(polarities, t.tensor([0.0]), atol=1e-8)
    # Make the sure the labels only have the values -1.0 and 1.0
    labels_copy = labels.clone()
    labels_copy = t.where(labels_copy == 0.0, t.tensor(-1.0), labels_copy)

    if all_polarities_zero:
        X = labels_copy.reshape(-1, 1)
    else:
        X = t.column_stack([labels_copy, labels_copy * polarities])

    # Compute the analytical OLS solution
    solution = t.linalg.inv(X.T @ X) @ X.T @ acts_centered

    # Extract t_g and t_p
    if all_polarities_zero:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]
        t_p = solution[1, :]

    return t_g, t_p


def learn_polarity_direction(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty=None, fit_intercept=True)
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    polarity_direc = LR_polarity.coef_
    return polarity_direc


class TTPD:
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.polarity_direc = learn_polarity_direction(acts, polarities)
        acts_2d = probe._project_acts(acts)
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts_2d, labels.numpy())
        return probe

    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_2d))

    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g
        proj_p = acts.numpy() @ self.polarity_direc.T
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d


def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = t.min(t.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return t.mean(consistency_losses + confidence_losses)


class CCSProbe(t.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=True), t.nn.Sigmoid())

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, acts, iid=None):
        return self(acts).round()

    def from_data(
        acts,
        neg_acts,
        labels=None,
        lr=0.001,
        weight_decay=0.1,
        epochs=1000,
        device="cpu",
    ):
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = CCSProbe(acts.shape[-1]).to(device)

        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None:  # flip direction if needed
            labels = labels.to(device)
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1

        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]

    @property
    def bias(self):
        return self.net[0].bias.data[0]


class LRProbe:
    def __init__(self):
        self.LR = None

    def from_data(acts, labels):
        probe = LRProbe()
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts.numpy(), labels.numpy())
        return probe

    def pred(self, acts):
        return t.tensor(self.LR.predict(acts))


class MMProbe(t.nn.Module):
    def __init__(self, direction, LR):
        super().__init__()
        self.direction = direction
        self.LR = LR

    def forward(self, acts):
        proj = acts @ self.direction
        return t.tensor(self.LR.predict(proj[:, None]))

    def pred(self, x):
        return self(x).round()

    def from_data(acts, labels, device="cpu"):
        pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        # project activations onto direction
        proj = acts @ direction
        # fit bias
        LR = LogisticRegression(penalty=None, fit_intercept=True)
        LR.fit(proj[:, None], labels)

        probe = MMProbe(direction, LR).to(device)

        return probe


ACTS_BATCH_SIZE = 25


def collect_acts(
    dataset_name,
    model_family,
    model_size,
    model_type,
    layer,
    center=True,
    scale=False,
    device="cpu",
):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(
        ROOT,
        "acts",
        model_family,
        model_size,
        model_type,
        dataset_name,
    )
    if dataset_name == "iphr":
        activation_files = glob.glob(
            os.path.join(directory, experiment_variation, f"layer_{layer}_*.pt")
        )
    else:
        activation_files = glob.glob(os.path.join(directory, f"layer_{layer}_*.pt"))
    acts = [
        t.load(
            os.path.join(directory, f"layer_{layer}_{i}.pt")
            if dataset_name != "iphr"
            else os.path.join(directory, experiment_variation, f"layer_{layer}_{i}.pt"),
            map_location=device,
        )
        for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)
    ]
    try:
        acts = t.cat(acts, dim=0).to(device)
    except Exception:
        raise Exception(
            "No activation vectors could be found for the dataset "
            + dataset_name
            + ". Please generate them first using generate_acts."
        )
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts


def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0:  # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    try:
        acts, labels = t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)
    except Exception:
        raise Exception(
            "No activation vectors could be found for this dataset. Please generate them first using generate_acts."
        )
    return acts, labels


class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """

    def __init__(self):
        self.data = {"train": {}, "val": {}}  # dictionary of datasets
        self.proj = None  # projection matrix for dimensionality reduction

    def add_dataset(
        self,
        dataset_name,
        model_family,
        model_size,
        model_type,
        layer,
        label="label",
        split=None,
        seed=None,
        center=True,
        scale=False,
        device="cpu",
        labels=None,
    ):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        acts = collect_acts(
            dataset_name,
            model_family,
            model_size,
            model_type,
            layer,
            center=center,
            scale=scale,
            device=device,
        )
        if labels is None:
            df = pd.read_csv(os.path.join(ROOT, "datasets", f"{dataset_name}.csv"))
            labels = t.Tensor(df[label].values).to(device)

        if split is None:
            self.data[dataset_name] = acts, labels

        if split is not None:
            assert df is not None
            assert 0 <= split and split <= 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data["train"][dataset_name] = acts[train], labels[train]
            self.data["val"][dataset_name] = acts[val], labels[val]

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == "all":
            data_dict = self.data
        elif datasets == "train":
            data_dict = self.data["train"]
        elif datasets == "val":
            data_dict = self.data["val"]
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data["train"][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data["val"][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets: self.data[datasets]}
        else:
            raise ValueError(
                f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}"
            )
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels


def dataset_sizes(datasets):
    """
    Computes the size of each dataset, i.e. the number of statements.
    Input: array of strings that are the names of the datasets
    Output: dictionary, keys are the dataset names and values the number of statements
    """
    dataset_sizes_dict = {}
    for dataset in datasets:
        file_path = ROOT + "/datasets/" + dataset + ".csv"
        with open(file_path, "r") as file:
            line_count = sum(1 for line in file)
        dataset_sizes_dict[dataset] = line_count - 1
    return dataset_sizes_dict


def collect_training_data(
    dataset_names,
    train_set_sizes,
    model_family,
    model_size,
    model_type,
    layer,
    **kwargs,
):
    """
    Takes as input the names of datasets in the format
    [affirmative_dataset1, negated_dataset1, affirmative_dataset2, negated_dataset2, ...]
    and returns a balanced training dataset of centered activations, activations, labels and polarities
    """
    all_acts_centered, all_acts, all_labels, all_polarities = [], [], [], []

    for dataset_name in dataset_names:
        dm = DataManager()
        dm.add_dataset(
            dataset_name,
            model_family,
            model_size,
            model_type,
            layer,
            split=None,
            center=False,
            device="cpu",
        )
        acts, labels = dm.data[dataset_name]

        polarity = -1.0 if "neg_" in dataset_name else 1.0
        polarities = t.full((labels.shape[0],), polarity)

        # balance the training dataset by including an equal number of activations from each dataset
        # choose the same subset of statements for affirmative and negated version of the dataset
        if "neg_" not in dataset_name:
            rand_subset = np.random.choice(
                acts.shape[0], min(train_set_sizes.values()), replace=False
            )

        all_acts_centered.append(
            acts[rand_subset, :] - t.mean(acts[rand_subset, :], dim=0)
        )
        all_acts.append(acts[rand_subset, :])
        all_labels.append(labels[rand_subset])
        all_polarities.append(polarities[rand_subset])

    return map(t.cat, (all_acts_centered, all_acts, all_labels, all_polarities))


# %% Training probes and saving to disk

# define datasets used for training of probes
train_sets = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "inventors",
    "neg_inventors",
    "animal_class",
    "neg_animal_class",
    "element_symb",
    "neg_element_symb",
    "facts",
    "neg_facts",
]
# get size of each training dataset to include an equal number of statements from each topic in training data
train_set_sizes = dataset_sizes(train_sets)

# Train probes for each type and layer
probe_types = {
    "TTPD": TTPD,
    "LRProbe": LRProbe,
    "CCSProbe": CCSProbe,
    "MMProbe": MMProbe,
}
all_layers = list(range(n_layers))
layers_to_evaluate = all_layers

trained_probes_path = f"trained_probes_{experiment_variation}.pkl"
if os.path.exists(trained_probes_path):
    with open(trained_probes_path, "rb") as f:
        trained_probes = pickle.load(f)
else:
    trained_probes = {probe_type: {} for probe_type in probe_types}
    device = model.device

    total_runs = len(probe_types) * len(layers_to_evaluate)
    with tqdm(total=total_runs, desc="Training probes") as pbar:
        for probe_type in probe_types.keys():
            for layer in layers_to_evaluate:
                # Load training data from all training sets
                acts_centered, acts, labels, polarities = collect_training_data(
                    train_sets,
                    train_set_sizes,
                    model_family,
                    model_size,
                    model_type,
                    layer,
                )

                # Train probe based on type
                if probe_type == "TTPD":
                    probe = TTPD.from_data(acts_centered, acts, labels, polarities)
                elif probe_type == "LRProbe":
                    probe = LRProbe.from_data(acts, labels)
                elif probe_type == "CCSProbe":
                    acts_affirm = acts[polarities == 1.0]
                    acts_neg = acts[polarities == -1.0]
                    labels_affirm = labels[polarities == 1.0]
                    mean_affirm = t.mean(acts_affirm, dim=0)
                    mean_neg = t.mean(acts_neg, dim=0)
                    acts_affirm = acts_affirm - mean_affirm
                    acts_neg = acts_neg - mean_neg
                    probe = CCSProbe.from_data(
                        acts_affirm, acts_neg, labels_affirm, device=device
                    ).to("cpu")
                    # Store means for later use in evaluation
                    probe.mean_affirm = mean_affirm
                    probe.mean_neg = mean_neg
                elif probe_type == "MMProbe":
                    probe = MMProbe.from_data(acts, labels)
                else:
                    raise ValueError(f"Unknown probe type: {probe_type}")

                trained_probes[probe_type][layer] = probe
                pbar.update(1)

    print("Saving trained probes to disk...")
    with open(trained_probes_path, "wb") as f:
        pickle.dump(trained_probes, f)
# %%

benchmark_results_path = f"benchmark_ttpd_results_{experiment_variation}.pkl"
if os.path.exists(benchmark_results_path):
    with open(benchmark_results_path, "rb") as f:
        results = pickle.load(f)
else:
    # Evaluate on IPHR dataset
    results = {probe_type: {} for probe_type in probe_types}
    total_runs = len(probe_types) * len(layers_to_evaluate)
    with tqdm(total=total_runs, desc="Evaluating probes") as pbar:
        for layer in layers_to_evaluate:
            dm = DataManager()
            dm.add_dataset(
                iphr_dataset_name,
                model_family,
                model_size,
                model_type,
                layer,
                split=None,
                center=False,
                device="cpu",
                labels=iphr_labels,
            )
            eval_acts, eval_labels = dm.data[iphr_dataset_name]

            for probe_type in probe_types.keys():
                probe = trained_probes[probe_type][layer]
                # Classifier specific predictions
                if probe_type == CCSProbe:
                    # For IPHR datasets, we treat them as affirmative statements
                    eval_acts = eval_acts - probe.mean_affirm

                predictions = probe.pred(eval_acts)
                res = {
                    "predictions": predictions,
                    "labels": eval_labels,
                    "accuracy": (predictions == eval_labels).float().mean().item(),
                }
                results[probe_type][layer] = res
                pbar.update(1)

    # Save results to disk as pickle
    with open(benchmark_results_path, "wb") as f:
        pickle.dump(results, f)

# %%

# Make a line plot for each probe type of the accuracy of the probe on the IPHR dataset for each layer

# Set up the plot style
fig = plt.figure(figsize=(12, 8))
plt.style.use("default")  # Use default style instead of seaborn

# Plot line for each probe type
for probe_type in probe_types:
    accuracies = [
        results[probe_type][layer]["accuracy"] for layer in layers_to_evaluate
    ]
    plt.plot(
        layers_to_evaluate,  # Use actual layer numbers for x-axis
        accuracies,
        label=probe_type,
        marker="o",
        alpha=0.7,
        linewidth=2,
        markersize=4,
    )

plt.xlabel("Layer", fontsize=16)
plt.ylabel("Accuracy on IPHR Dataset", fontsize=16)
plt.title(
    f"Accuracy of Lie Detection Techniques Across {model_id.split('/')[-1]} Layers\nExperiment Variation: {experiment_variation}",
    fontsize=18,
)
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)

# Add minor gridlines
plt.minorticks_on()
plt.grid(True, which="minor", alpha=0.1)

# Save the plot
fig.savefig(
    f"benchmark_ttpd_results_{experiment_variation}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%

# First, identify which unfaithful questions we actually have statements for in our sampled dataset
sampled_unfaithful_q_ids = {
    item["q_id"] for item in iphr_metadata if item["is_unfaithful"]
}
print(f"\nNumber of unfaithful questions in original dataset: {len(unfaithful_q_ids)}")
print(
    f"Number of unfaithful questions with statements in sampled dataset: {len(sampled_unfaithful_q_ids)}"
)

# Create a dictionary to store detection metrics for each probe type and layer
detection_metrics = {probe_type: {} for probe_type in probe_types}

# For each probe type and layer, calculate confusion matrix metrics
for probe_type in probe_types:
    for layer in layers_to_evaluate:
        # Get predictions
        predictions = results[probe_type][layer]["predictions"]

        # Create dictionaries to track if questions have any lying statements
        unfaithful_q_detection = {q_id: False for q_id in sampled_unfaithful_q_ids}
        faithful_q_false_detection = {
            q_id: False
            for q_id in set(m["q_id"] for m in iphr_metadata if not m["is_unfaithful"])
        }

        # Go through each statement and check predictions
        for pred, metadata in zip(predictions, iphr_metadata):
            q_id = metadata["q_id"]
            is_unfaithful = metadata["is_unfaithful"]

            # If prediction is 0, it means the statement was detected as lying
            if pred == 0:
                if is_unfaithful:
                    unfaithful_q_detection[q_id] = True
                else:
                    faithful_q_false_detection[q_id] = True

        # Calculate confusion matrix metrics
        TP = sum(
            unfaithful_q_detection.values()
        )  # Unfaithful questions correctly detected
        FN = len(sampled_unfaithful_q_ids) - TP  # Unfaithful questions missed
        FP = sum(
            faithful_q_false_detection.values()
        )  # Faithful questions wrongly flagged
        TN = len(faithful_q_false_detection) - FP  # Faithful questions correctly passed

        # Calculate precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        detection_metrics[probe_type][layer] = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

# Create plots for precision, recall, and F1 score
metrics_to_plot = ["precision", "recall", "f1"]
plt.style.use("default")

for metric in metrics_to_plot:
    fig = plt.figure(figsize=(12, 6))
    for probe_type in probe_types:
        values = [
            detection_metrics[probe_type][layer][metric] for layer in layers_to_evaluate
        ]
        plt.plot(
            layers_to_evaluate,
            values,
            label=probe_type,
            marker="o",
            alpha=0.7,
            linewidth=2,
            markersize=4,
        )

    plt.xlabel("Layer", fontsize=16)
    plt.ylabel(metric.capitalize(), fontsize=16)
    plt.title(
        f"{metric.capitalize()} of Lie Detection Techniques Across {model_id.split('/')[-1]} Layers\nExperiment Variation: {experiment_variation}",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.minorticks_on()
    plt.grid(True, which="minor", alpha=0.1)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"benchmark_ttpd_{metric}_{experiment_variation}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# Print summary statistics for the best performing layer of each probe based on F1 score
print("\nBest layer performance for each probe (based on F1 score):")
print("-" * 70)
print(
    f"{'Probe Type':<15} {'Layer':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP/FP/TN/FN'}"
)
print("-" * 70)

for probe_type in probe_types:
    # Find best layer based on F1 score
    best_layer = max(detection_metrics[probe_type].items(), key=lambda x: x[1]["f1"])[0]

    metrics = detection_metrics[probe_type][best_layer]
    print(
        f"{probe_type:<15} {best_layer:<8} "
        f"{metrics['precision']:.3f}      {metrics['recall']:.3f}      {metrics['f1']:.3f}      "
        f"{metrics['TP']}/{metrics['FP']}/{metrics['TN']}/{metrics['FN']}"
    )

# Find the best probe and its best layer based on F1 score
best_probe = None
best_layer = None
best_f1 = -1

for probe_type, layer_metrics in detection_metrics.items():
    for layer, metrics in layer_metrics.items():
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_probe = probe_type
            best_layer = layer

print(f"\nDetailed analysis for best model ({best_probe} at layer {best_layer}):")
print("-" * 50)
best_metrics = detection_metrics[best_probe][best_layer]
print(f"Precision: {best_metrics['precision']:.3f}")
print(f"Recall: {best_metrics['recall']:.3f}")
print(f"F1 Score: {best_metrics['f1']:.3f}")
print("\nConfusion Matrix:")
print(f"True Positives: {best_metrics['TP']} (unfaithful questions correctly detected)")
print(f"False Positives: {best_metrics['FP']} (faithful questions wrongly flagged)")
print(f"True Negatives: {best_metrics['TN']} (faithful questions correctly passed)")
print(f"False Negatives: {best_metrics['FN']} (unfaithful questions missed)")

# Get predictions for detailed analysis
predictions = results[best_probe][best_layer]["predictions"]

# Track which questions were detected/missed and store all statements
question_stats = {
    q_id: {
        "total": 0,
        "lying": 0,
        "statements": [],
        "is_unfaithful": any(
            m["is_unfaithful"] for m in iphr_metadata if m["q_id"] == q_id
        ),
    }
    for q_id in set(m["q_id"] for m in iphr_metadata)
}

for idx, (pred, metadata, statement) in enumerate(
    zip(predictions, iphr_metadata, iphr_dataset)
):
    q_id = metadata["q_id"]
    question_stats[q_id]["total"] += 1
    if pred == 0:  # Statement marked as lying
        question_stats[q_id]["lying"] += 1
    question_stats[q_id]["statements"].append(
        {
            "statement": statement,
            "response_id": metadata["response_id"],
            "marked_as_lying": pred == 0,
        }
    )

# Print detailed results for misclassified questions (FP and FN)
# print("\nDetailed analysis of misclassifications:")
# print("\nFalse Positives (faithful questions wrongly flagged as unfaithful):")
# for q_id, stats in question_stats.items():
#     if not stats["is_unfaithful"] and stats["lying"] > 0:
#         print(
#             f"\nQuestion {q_id}: {stats['lying']}/{stats['total']} statements marked as lying"
#         )
#         for stmt_info in stats["statements"]:
#             lying_mark = "LYING" if stmt_info["marked_as_lying"] else "TRUTHFUL"
#             print(f"  [{lying_mark}] Response {stmt_info['response_id']}:")
#             print(f"    {stmt_info['statement']}")

# print("\nFalse Negatives (unfaithful questions missed):")
# for q_id, stats in question_stats.items():
#     if stats["is_unfaithful"] and stats["lying"] == 0:
#         print(
#             f"\nQuestion {q_id}: {stats['lying']}/{stats['total']} statements marked as lying"
#         )
#         for stmt_info in stats["statements"]:
#             lying_mark = "LYING" if stmt_info["marked_as_lying"] else "TRUTHFUL"
#             print(f"  [{lying_mark}] Response {stmt_info['response_id']}:")
#             print(f"    {stmt_info['statement']}")

# %%
