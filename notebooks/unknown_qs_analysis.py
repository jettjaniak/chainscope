#!/usr/bin/env python3

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from chainscope.typing import *

# %%
# Caches for loaded data
faithfulness_yamls_cache = {}
evals_cache = {}

# %%
# Load the data
df = pd.read_pickle(DATA_DIR / "df-wm.pkl")
df = df[df["mode"] == "cot"]

# Filter for Sonnet 1k and 64k
sonnet_1k = df[df["model_id"] == "anthropic/claude-3.7-sonnet_1k"]
sonnet_64k = df[df["model_id"] == "anthropic/claude-3.7-sonnet_64k"]

assert len(sonnet_1k) > 0, f"Sonnet 1k must have at least one row, available models: {df['model_id'].unique()}"
assert len(sonnet_64k) > 0, f"Sonnet 64k must have at least one row, available models: {df['model_id'].unique()}"
assert len(sonnet_1k) == len(sonnet_64k), f"Sonnet 1k and 64k must have the same number of rows, {len(sonnet_1k)} != {len(sonnet_64k)}"

# %%
def load_eval(row) -> dict[str, CotEvalResult]:
    q_id = row["qid"]
    model_id = row["model_id"]
    instr_id = row["instr_id"]

    # Create cache key
    eval_key = (
        model_id,
        instr_id,
        row["prop_id"],
        row["comparison"],
        row["dataset_id"],
    )

    # Try to get from cache first
    if eval_key in evals_cache:
        return evals_cache[eval_key][q_id]

    dataset_params = DatasetParams(
        prop_id=row["prop_id"],
        comparison=row["comparison"],
        answer=row["answer"],
        max_comparisons=1,
        uuid=row["dataset_id"].split("_")[-1],
    )

    sampling_params = SamplingParams(
        temperature=float(row["temperature"]),
        top_p=float(row["top_p"]),
        max_new_tokens=int(row["max_new_tokens"]),
    )

    # Load evaluations if not in cache
    print(f"Loading COT eval for {row['dataset_id']}")
    cot_eval = dataset_params.load_cot_eval(
        row["instr_id"],
        row["model_id"],
        sampling_params,
    )
    evals_cache[eval_key] = cot_eval.results_by_qid

    return evals_cache[eval_key][q_id]

# %%
# Process responses and count unknowns using CotEvalResult
def count_answers_by_type(eval_dict: dict[str, CotEvalResult], answer_type: Literal["UNKNOWN", "REFUSED"]) -> int:
    return sum(1 for result in eval_dict.values() 
              if result.final_answer == answer_type)

unknown_1k_count = 0
unknown_64k_count = 0
refused_1k_count = 0
refused_64k_count = 0
resolved_in_64k_count = 0
resolved_in_1k_count = 0
refused_resolved_in_64k_count = 0
refused_resolved_in_1k_count = 0

# %%
# Load faithfulness data for 64k
model_id = "anthropic/claude-3.7-sonnet_64k"
model_dir_name = model_id.split("/")[-1]
faith_dir = DATA_DIR / "faithfulness" / model_dir_name

# Check if we've already cached this model's data
if model_id in faithfulness_yamls_cache:
    merged_faith_data = faithfulness_yamls_cache[model_id]
else:
    # Get all YAML files in the directory
    yaml_files = list(faith_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No faithfulness YAML files found in directory for {model_id}: {faith_dir}")
        merged_faith_data = {}
    else:
        # Load and merge all YAML files
        print(f"Loading faithfulness data for {model_id} from {len(yaml_files)} files in {faith_dir}")
        merged_faith_data = {}
        for yaml_file in tqdm(yaml_files, desc="Loading faithfulness data"):
            try:
                with open(yaml_file) as f:
                    faith_data = yaml.safe_load(f)
                    if faith_data:
                        merged_faith_data.update(faith_data)
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")
        
        # Cache the merged data
        faithfulness_yamls_cache[model_id] = merged_faith_data

# Count unfaithful pairs
unfaithful_64k_total = len(merged_faith_data.keys())

# %%
# Process each question pair
unfaithful_64k_unknown_1k = 0
unfaithful_64k_unknown_64k = 0
unfaithful_64k_refused_1k = 0
unfaithful_64k_refused_64k = 0

# Lists to collect differences
unknown_diffs = []
refused_diffs = []

# Get all unique QIDs
sonnet_1k_qids = set(sonnet_1k["qid"])
sonnet_64k_qids = set(sonnet_64k["qid"])
assert sonnet_1k_qids == sonnet_64k_qids, f"Sonnet 1k and 64k must have the same QIDs, {sonnet_1k_qids} != {sonnet_64k_qids}"
qids = sonnet_1k_qids

for qid in tqdm(qids, desc="Processing questions"):
    row_1k = sonnet_1k[sonnet_1k["qid"] == qid].iloc[0]
    row_64k = sonnet_64k[sonnet_64k["qid"] == qid].iloc[0]
    
    eval_1k = load_eval(row_1k)
    eval_64k = load_eval(row_64k)
    
    # Count unknowns and refused in each context
    unknowns_1k = count_answers_by_type(eval_1k, "UNKNOWN")
    unknowns_64k = count_answers_by_type(eval_64k, "UNKNOWN")
    refused_1k = count_answers_by_type(eval_1k, "REFUSED")
    refused_64k = count_answers_by_type(eval_64k, "REFUSED")
    
    # Calculate differences (1k - 64k)
    unknown_diffs.append(unknowns_1k - unknowns_64k)
    refused_diffs.append(refused_1k - refused_64k)
    
    unknown_1k_count += unknowns_1k
    unknown_64k_count += unknowns_64k
    refused_1k_count += refused_1k
    refused_64k_count += refused_64k
    
    # Count questions resolved in each context
    if unknowns_1k > 0 and unknowns_64k == 0:
        resolved_in_64k_count += 1
    if unknowns_1k == 0 and unknowns_64k > 0:
        resolved_in_1k_count += 1
        
    # Count refused questions resolved in each context
    if refused_1k > 0 and refused_64k == 0:
        refused_resolved_in_64k_count += 1
    if refused_1k == 0 and refused_64k > 0:
        refused_resolved_in_1k_count += 1
    
    # For unfaithful pairs in 64k, check unknown and refused status
    if qid in merged_faith_data:
        unfaithful_64k_unknown_1k += unknowns_1k > 0
        unfaithful_64k_unknown_64k += unknowns_64k > 0
        unfaithful_64k_refused_1k += refused_1k > 0
        unfaithful_64k_refused_64k += refused_64k > 0

# %%
print(f"\n## Unknown answers statistics:")
print(f"Sonnet 3.7 1k context: {unknown_1k_count} questions with at least one unknown answer (out of {len(sonnet_1k)} questions)")
print(f"Sonnet 3.7 64k context: {unknown_64k_count} questions with at least one unknown answer (out of {len(sonnet_64k)} questions)")

print(f"\n## Refused answers statistics:")
print(f"Sonnet 3.7 1k context: {refused_1k_count} questions with at least one refused answer (out of {len(sonnet_1k)} questions)")
print(f"Sonnet 3.7 64k context: {refused_64k_count} questions with at least one refused answer (out of {len(sonnet_64k)} questions)")

print(f"\n## Questions resolved in one over the other:")
print(f"Unknown answers:")
print(f"  - {resolved_in_64k_count} questions had at least one unknown answer in 1k but not in 64k")
print(f"  - {resolved_in_1k_count} questions had at least one unknown answer in 64k but not in 1k")
print(f"Refused answers:")
print(f"  - {refused_resolved_in_64k_count} questions had at least one refused answer in 1k but not in 64k")
print(f"  - {refused_resolved_in_1k_count} questions had at least one refused answer in 64k but not in 1k")

print(f"\n## Unfaithful pairs in 64k analysis:")
print(f"Total unfaithful pairs in 64k: {unfaithful_64k_total}")
print(f"Of these:")
print(f"  - {unfaithful_64k_unknown_1k} had at least one unknown answer in 1k context")
print(f"  - {unfaithful_64k_unknown_64k} had at least one unknown answer in 64k context")
print(f"  - {unfaithful_64k_refused_1k} had at least one refused answer in 1k context")
print(f"  - {unfaithful_64k_refused_64k} had at least one refused answer in 64k context")

# Print difference statistics
print("\n## Difference statistics (1k - 64k):")
print("\nUnknown answers:")
print(f"  - Mean difference: {np.mean(unknown_diffs):.2f}")
print(f"  - Median difference: {np.median(unknown_diffs):.2f}")
print(f"  - Standard deviation: {np.std(unknown_diffs):.2f}")
print(f"  - Range: [{min(unknown_diffs)}, {max(unknown_diffs)}]")

print("\nRefused answers:")
print(f"  - Mean difference: {np.mean(refused_diffs):.2f}")
print(f"  - Median difference: {np.median(refused_diffs):.2f}")
print(f"  - Standard deviation: {np.std(refused_diffs):.2f}")
print(f"  - Range: [{min(refused_diffs)}, {max(refused_diffs)}]")

# %%
def plot_diff_histogram(diffs: list[int], title: str, filename: str):
    fig = plt.figure(figsize=(10, 6))
    bins = list(np.arange(-10.5, 11.5, 1))  # Creates bins from -10 to 10
    plt.hist(diffs, bins=bins, align='mid', rwidth=0.8)
    plt.title(title)
    plt.xlabel("Difference (1k - 64k)")
    plt.ylabel("Number of questions")
    plt.grid(True, alpha=0.3)
    
    # Add counts as text above each bar
    counts, edges = np.histogram(diffs, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    for count, center in zip(counts, centers):
        if count > 0:  # Only add text for non-zero bars
            plt.text(center, count, str(count), 
                    horizontalalignment='center',
                    verticalalignment='bottom')
    
    fig.savefig(filename)
    plt.show()
    plt.close()

# Plot unknown differences
plot_diff_histogram(
    unknown_diffs,
    "Difference in Unknown Answers per question (1k - 64k)",
    "unknown_diffs_histogram.png"
)

# Plot refused differences
plot_diff_histogram(
    refused_diffs,
    "Difference in Refused Answers per question (1k - 64k)",
    "refused_diffs_histogram.png"
)
# %%
