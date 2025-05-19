# %%
import json

import pandas as pd

from chainscope.typing import *

# %%

# Cache for responses and evaluations
_responses_cache: dict[Path, CotResponses] = {}
_evals_cache: dict[tuple[str, str, str, str], CotEval] = {}
_faithfulness_cache: dict[str, UnfaithfulnessPairsDataset] = {}
_qs_dataset_cache: dict[Path, QsDataset] = {}

# %%

model_id = "meta-llama/Llama-3.3-70B-Instruct"

# %%

# Load the data
df = pd.read_pickle(DATA_DIR / "df-wm-non-ambiguous-hard-2.pkl")
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, dataset_suffix, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

df = df[df["mode"] == "cot"]

# Filter by model
df = df[df["model_id"] == model_id]

assert (
    len(df) > 0
), f"No data found, models are: {pd.read_pickle(DATA_DIR / 'df-wm.pkl')['model_id'].unique()}"

prop_ids_with_suffix = set()
for row in df.itertuples():
    prop_id = row.prop_id
    if row.dataset_suffix is not None:
        prop_id = f"{prop_id}_{row.dataset_suffix}"
    prop_ids_with_suffix.add(prop_id)

print(f"Number of unique prop_ids with suffix: {len(prop_ids_with_suffix)}")

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

model_file_name = model_id.split("/")[-1]
faithfulness_dir = DATA_DIR / "faithfulness" / model_file_name
faithfulness_files = list(faithfulness_dir.glob("*.yaml"))

unfaithful_q_ids = []
p_correct_by_qid = {}
reversed_q_ids = {}

for file in faithfulness_files:
    prop_id_with_suffix = file.stem
    if prop_id_with_suffix not in prop_ids_with_suffix:
        print(f"Skipping {prop_id_with_suffix} because it is not in the df")
        continue
    
    if prop_id_with_suffix not in _faithfulness_cache:
        try:
            _faithfulness_cache[prop_id_with_suffix] = UnfaithfulnessPairsDataset.load_from_path(file)
        except Exception as e:
            print(f"Error loading unfaithfulness dataset in file {file}: {e}")
            raise e
    
    dataset = _faithfulness_cache[prop_id_with_suffix]
    for qid, question in dataset.questions_by_qid.items():
        if question.metadata is None:
            continue
            
        unfaithful_q_ids.append(qid)
        unfaithful_q_ids.append(question.metadata.reversed_q_id)
        
        p_correct_by_qid[qid] = question.metadata.p_correct
        p_correct_by_qid[question.metadata.reversed_q_id] = question.metadata.reversed_q_p_correct
        
        reversed_q_ids[qid] = question.metadata.reversed_q_id
        reversed_q_ids[question.metadata.reversed_q_id] = qid

# %%

# Load unfaithfulness pattern evaluation data
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    max_new_tokens=8000,
)

# Load pattern evaluations for each prop_id_with_suffix
pattern_evals: dict[str, UnfaithfulnessPatternEval] = {}
for prop_id_with_suffix in prop_ids_with_suffix:
    try:
        eval_path = DATA_DIR / "unfaithfulness_pattern_eval" / sampling_params.id / prop_id_with_suffix / f"{model_id.split('/')[-1]}.yaml"
        if eval_path.exists():
            pattern_evals[prop_id_with_suffix] = UnfaithfulnessPatternEval.load(eval_path)
            print(f"Loaded pattern evaluation for {prop_id_with_suffix}")
        else:
            print(f"No pattern evaluation found for {prop_id_with_suffix}")
    except Exception as e:
        print(f"Error loading pattern evaluation for {prop_id_with_suffix}: {e}")

print(f"Loaded {len(pattern_evals)} pattern evaluations")

# %%
assert len(unfaithful_q_ids) > 0, "No unfaithful data found"

print(f"Number of unfaithful q_ids: {len(unfaithful_q_ids)}")

# Example of unfaithful q_id:
print(f" ## Example of unfaithful q_id: {unfaithful_q_ids[0]}.")
print(f"P_correct: {p_correct_by_qid[unfaithful_q_ids[0]]}")
print(f"Reversed q_id: {reversed_q_ids[unfaithful_q_ids[0]]}")
print(f"P_correct (reversed): {p_correct_by_qid[reversed_q_ids[unfaithful_q_ids[0]]]}")

# %%

filter_unfaithfulness_pattern: Literal["fact-manipulation", "argument-switching", "answer-flipping", "other", "none"] | None = None

# Find the pair of unfaithful qs with largest difference in p_correct

# Create sorted pairs of questions
unfaithful_qids_pairs = []
seen_pairs = set()
for qid in unfaithful_q_ids:
    rev_q_id = reversed_q_ids[qid]
    if (
        rev_q_id in p_correct_by_qid
        and qid not in seen_pairs
        and rev_q_id not in seen_pairs
    ):
        # Get dataset info to find pattern analysis
        dataset_id = df.loc[df["qid"] == qid, "dataset_id"].values[0]
        prop_id = df.loc[df["qid"] == qid, "prop_id"].values[0]
        dataset_suffix = df.loc[df["qid"] == qid, "dataset_suffix"].values[0]
        
        # Construct prop_id_with_suffix
        prop_id_with_suffix = f"{prop_id}"
        if dataset_suffix is not None:
            prop_id_with_suffix = f"{prop_id_with_suffix}_{dataset_suffix}"
        
        # Check if this pair matches the filter pattern
        matches_filter = True
        if filter_unfaithfulness_pattern is not None:
            matches_filter = False
            if prop_id_with_suffix in pattern_evals:
                pattern_eval = pattern_evals[prop_id_with_suffix]
                if qid in pattern_eval.pattern_analysis_by_qid:
                    pattern_analysis = pattern_eval.pattern_analysis_by_qid[qid]
                    if pattern_analysis.categorization_for_pair:
                        if filter_unfaithfulness_pattern in pattern_analysis.categorization_for_pair:
                            matches_filter = True
        
        if matches_filter:
            pair = tuple(sorted([qid, rev_q_id]))  # Sort to ensure consistent ordering
            diff = abs(p_correct_by_qid[qid] - p_correct_by_qid[rev_q_id])
            unfaithful_qids_pairs.append((pair, diff))
            seen_pairs.add(qid)
            seen_pairs.add(rev_q_id)

# Sort by difference in p_correct
unfaithful_qids_pairs.sort(key=lambda x: x[1], reverse=True)

# Print summary of filtered results
if filter_unfaithfulness_pattern:
    print(f"\nFiltering for pairs showing '{filter_unfaithfulness_pattern}' pattern")
print(f"Number of unfaithful qids pairs: {len(unfaithful_qids_pairs)}")

# %%

# Load the instruction-wm prompt
wm_template = Instructions.load("instr-wm").cot

# %%

# Print the top K pairs with largest difference
K = 3
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
    prop_id = df.loc[df["qid"] == qid1, "prop_id"].values[0]
    dataset_suffix = df.loc[df["qid"] == qid1, "dataset_suffix"].values[0]
    x_name = df.loc[df["qid"] == qid1, "x_name"].values[0]
    y_name = df.loc[df["qid"] == qid1, "y_name"].values[0]

    # Get prop_id_with_suffix from dataset_id
    prop_id_with_suffix = f"{prop_id}"
    if dataset_suffix is not None:
        prop_id_with_suffix = f"{prop_id_with_suffix}_{dataset_suffix}"

    # Get pattern analysis if available
    pattern_analysis = None
    if prop_id_with_suffix in pattern_evals:
        pattern_eval = pattern_evals[prop_id_with_suffix]
        if qid1 in pattern_eval.pattern_analysis_by_qid:
            pattern_analysis = pattern_eval.pattern_analysis_by_qid[qid1]

    print(f"Dataset id: {dataset_id}")
    print(f"First entity: {x_name}")
    print(f"Second entity: {y_name}")
    
    # Print pattern analysis if available
    if pattern_analysis:
        if pattern_analysis.categorization_for_pair:
            print(f"\nUnfaithfulness patterns: {', '.join(pattern_analysis.categorization_for_pair)}")
    print()

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
