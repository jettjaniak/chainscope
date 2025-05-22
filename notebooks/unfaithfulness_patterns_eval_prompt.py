# %%
import logging

from chainscope.typing import *
from scripts.iphr.unfaithfulness_patterns_eval import \
    build_unfaithfulness_prompt

# %%
model_id = "meta-llama/Llama-3.3-70B-Instruct"
dataset_suffix = "non-ambiguous-hard-2"
q1_qid = "8a3b4a1f4071ac9cccf3628a1cf7d0e5c81d4e9dfc711feb8d03040a94410d62"
q2_qid = "8e8a7bed46c0943ef9027236f5b1f61b5106142699539bcf33acebc70713439c"

# %%
# Load the faithfulness dataset
model_file_name = model_id.split("/")[-1]
faithfulness_dir = DATA_DIR / "faithfulness" / model_file_name
faithfulness_files = list(faithfulness_dir.glob(f"*{dataset_suffix}.yaml"))

# Find the file containing our questions
dataset = None
main_qid = None
for file in faithfulness_files:
    try:
        temp_dataset = UnfaithfulnessPairsDataset.load_from_path(file)
        if q1_qid in temp_dataset.questions_by_qid:
            dataset = temp_dataset
            main_qid = q1_qid
            break
        if q2_qid in temp_dataset.questions_by_qid:
            dataset = temp_dataset
            main_qid = q2_qid
            break
    except Exception as e:
        logging.warning(f"Error loading unfaithfulness dataset in file {file}: {e}")
        continue

if dataset is None or main_qid is None:
    raise ValueError(f"Could not find dataset containing question {q1_qid} or {q2_qid}")

# Get the question data
q1_data = dataset.questions_by_qid[main_qid]
if q1_data.metadata is None:
    raise ValueError(f"No metadata found for question {main_qid}")

# Get all responses for both questions
q1_all_responses = q1_data.metadata.q1_all_responses
q2_all_responses = q1_data.metadata.q2_all_responses

# Build the prompt
prompt, q1_response_mapping, q2_response_mapping = build_unfaithfulness_prompt(
    q1_str=q1_data.metadata.q_str,
    q1_all_responses=q1_all_responses,
    q1_answer=q1_data.metadata.answer,
    q2_str=q1_data.metadata.reversed_q_str,
    q2_all_responses=q2_all_responses,
    q2_answer="NO" if q1_data.metadata.answer == "YES" else "YES",
)

# Print the prompt
print("Unfaithfulness Evaluation Prompt:")
print("-" * 80)
print(prompt)
print("-" * 80)

# Print some metadata about the questions
print("\nQuestion Metadata:")
print(f"Q1 ID: {main_qid}")
print(f"Q1: {q1_data.metadata.q_str}")
print(f"Q1 Answer: {q1_data.metadata.answer}")
print(f"Q2: {q1_data.metadata.reversed_q_str}")
print(f"Q2 Answer: {'NO' if q1_data.metadata.answer == 'YES' else 'YES'}")
print(f"Number of Q1 responses: {len(q1_all_responses)}")
print(f"Number of Q2 responses: {len(q2_all_responses)}")



# %%
