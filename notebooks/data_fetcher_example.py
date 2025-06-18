# %%
import random

from chainscope.data_fetcher import DataFetcher

df = DataFetcher(model_id="meta-llama/Llama-3.3-70B-Instruct")

# %%

print(df.get_available_prop_ids())

# %%

# Load faithful question pairs

faithful_pairs = df.get_faithful_question_pairs()
print(f"Found {len(faithful_pairs)} faithful question pairs")
faithful_example = faithful_pairs[0]

# %%

print(f"Question 1: `{faithful_example.q1.q_str}`")
print(f"Question ID: `{faithful_example.q1.qid}`")
print(f"Dataset ID: `{faithful_example.q1.dataset_id}`")
print(f"Expected answer: `{faithful_example.q1.correct_answer}`")
print(f"Model accuracy: {faithful_example.q1.accuracy}")
print()

q1_random_response_id = random.choice(list(faithful_example.q1.responses.keys()))
print(f"Prompt: `{faithful_example.q1.responses[q1_random_response_id].prompt_str}`")
print(f"Example response: `{faithful_example.q1.responses[q1_random_response_id].response_str}`")
print(f"Parsed final answer: `{faithful_example.q1.responses[q1_random_response_id].model_answer}` (correct: {faithful_example.q1.responses[q1_random_response_id].is_correct})")

# %%

print(f"Question 2: `{faithful_example.q2.q_str}`")
print(f"Question ID: `{faithful_example.q2.qid}`")
print(f"Dataset ID: `{faithful_example.q2.dataset_id}`")
print(f"Expected answer: `{faithful_example.q2.correct_answer}`")
print(f"Model accuracy: {faithful_example.q2.accuracy}")
print()

q2_random_response_id = random.choice(list(faithful_example.q2.responses.keys()))
print(f"Prompt: `{faithful_example.q2.responses[q2_random_response_id].prompt_str}`")
print(f"Example response: `{faithful_example.q2.responses[q2_random_response_id].response_str}`")
print(f"Parsed final answer: `{faithful_example.q2.responses[q2_random_response_id].model_answer}` (correct: {faithful_example.q2.responses[q2_random_response_id].is_correct})")

# %%

# Load unfaithful question pairs

unfaithful_pairs = df.get_unfaithful_question_pairs()
print(f"Found {len(unfaithful_pairs)} unfaithful question pairs")
unfaithful_example = unfaithful_pairs[0]

# %%

print(f"Unfaithfulness patterns: {unfaithful_example.unfaithfulness_patterns}")

print(f"Question 1: `{unfaithful_example.q1.q_str}`")
print(f"Question ID: `{unfaithful_example.q1.qid}`")
print(f"Dataset ID: `{unfaithful_example.q1.dataset_id}`")
print(f"Expected answer: `{unfaithful_example.q1.correct_answer}`")
print(f"Model accuracy: {unfaithful_example.q1.accuracy}")
print()

q1_random_response_id = random.choice(list(unfaithful_example.q1.responses.keys()))
print(f"Prompt: `{unfaithful_example.q1.responses[q1_random_response_id].prompt_str}`")
print(f"Example response: `{unfaithful_example.q1.responses[q1_random_response_id].response_str}`")
print(f"Parsed final answer: `{unfaithful_example.q1.responses[q1_random_response_id].model_answer}` (correct: {unfaithful_example.q1.responses[q1_random_response_id].is_correct})")

# %%

print(f"Question 2: `{unfaithful_example.q2.q_str}`")
print(f"Question ID: `{unfaithful_example.q2.qid}`")
print(f"Dataset ID: `{unfaithful_example.q2.dataset_id}`")
print(f"Expected answer: `{unfaithful_example.q2.correct_answer}`")
print(f"Model accuracy: {unfaithful_example.q2.accuracy}")
print()

q2_random_response_id = random.choice(list(unfaithful_example.q2.responses.keys()))
print(f"Prompt: `{unfaithful_example.q2.responses[q2_random_response_id].prompt_str}`")
print(f"Example response: `{unfaithful_example.q2.responses[q2_random_response_id].response_str}`")
print(f"Parsed final answer: `{unfaithful_example.q2.responses[q2_random_response_id].model_answer}` (correct: {unfaithful_example.q2.responses[q2_random_response_id].is_correct})")

# %%