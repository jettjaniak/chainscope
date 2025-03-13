#!/usr/bin/env python3

# %%
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
def count_unknowns(eval_dict: dict[str, CotEvalResult]) -> int:
    return sum(1 for result in eval_dict.values() 
              if result.final_answer == "UNKNOWN")

unknown_1k_count = 0
unknown_64k_count = 0
resolved_in_64k_count = 0
resolved_in_1k_count = 0

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
    
    # Count unknowns in each context using CotEvalResult
    unknowns_1k = count_unknowns(eval_1k)
    unknowns_64k = count_unknowns(eval_64k)
    
    unknown_1k_count += unknowns_1k
    unknown_64k_count += unknowns_64k
    
    # Count questions resolved in 64k
    if unknowns_1k > 0 and unknowns_64k == 0:
        resolved_in_64k_count += 1
    if unknowns_1k == 0 and unknowns_64k > 0:
        resolved_in_1k_count += 1
    
    # For unfaithful pairs in 64k, check unknown status
    if qid in merged_faith_data:
        unfaithful_64k_unknown_1k += unknowns_1k > 0
        unfaithful_64k_unknown_64k += unknowns_64k > 0

# %%
print(f"\n## Unknown answers statistics:")
print(f"Sonnet 3.7 1k context: {unknown_1k_count} questions with at least one unknown answer (out of {len(sonnet_1k)} questions)")
print(f"Sonnet 3.7 64k context: {unknown_64k_count} questions with at least one unknown answer (out of {len(sonnet_64k)} questions)")
print(f"\n## Questions resolved in one over the other:")
print(f"  - {resolved_in_64k_count} questions had at least one unknown answer in 1k but not in 64k")
print(f"  - {resolved_in_1k_count} questions had at least one unknown answer in 64k but not in 1k")
print(f"\n## Unfaithful pairs in 64k analysis:")
print(f"Total unfaithful pairs in 64k: {unfaithful_64k_total}")
print(f"Of these, {unfaithful_64k_unknown_1k} had at least one unknown answer in 1k context")
print(f"And {unfaithful_64k_unknown_64k} had at least one unknown answer in 64k context") 
# %%
