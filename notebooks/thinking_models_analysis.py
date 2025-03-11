# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chainscope.typing import *

# Cache for responses and evaluations
_responses_cache: dict[Path, CotResponses] = {}
_evals_cache: dict[tuple[str, str, str, str], CotEval] = {}

# %%

model_id = "deepseek/deepseek-r1"

# %%
# Load the data
df = pd.read_pickle(DATA_DIR / "df-wm.pkl")
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
    expected_answer = dataset_params.answer

    return (
        responses.responses_by_qid[q_id],
        cot_eval.results_by_qid[q_id],
        expected_answer,
    )


# %%
# Collect response lengths and correctness
lengths = []
correct = []

for _, row in df.iterrows():
    responses, eval, expected_answer = load_responses_and_eval(row)
    assert type(responses) is dict, "Responses are not a dictionary"

    for response_id, response in responses.items():
        if response is not None:
            assert type(response) is str, "Response is not a string"
            lengths.append(len(response))
            correct.append(eval[response_id].final_answer == expected_answer)

# %%
# Analysis parameters
NUM_BUCKETS = 20
MIN_SAMPLES_PER_BUCKET = 20  # Filter out buckets with fewer samples than this

# Create bins for response lengths
bins = np.linspace(min(lengths), max(lengths), NUM_BUCKETS)
digitized = np.digitize(lengths, bins)

# Calculate percentage correct per bin
correct_by_bin = defaultdict(list)
for length_bin, is_correct in zip(digitized, correct):
    correct_by_bin[length_bin].append(is_correct)

percentages = []
bin_centers = []
counts = []
filtered_bins = []

for bin_idx in range(1, len(bins)):
    if bin_idx in correct_by_bin:
        bin_results = correct_by_bin[bin_idx]
        if len(bin_results) >= MIN_SAMPLES_PER_BUCKET:
            percentages.append(sum(bin_results) / len(bin_results) * 100)
            bin_centers.append((bins[bin_idx - 1] + bins[bin_idx]) / 2)
            counts.append(len(bin_results))
            filtered_bins.append((bins[bin_idx - 1], bins[bin_idx]))

# Create new bins array for histogram that only includes the filtered ranges
filtered_bin_edges = [b[0] for b in filtered_bins] + [filtered_bins[-1][1]]

# Create the plot with two subplots, sharing x axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1], sharex=True)

# Top subplot: Percentage correct bars
bars = ax1.bar(
    bin_centers, percentages, width=(filtered_bins[0][1] - filtered_bins[0][0]) * 0.8
)

# Add count labels on top of each bar
for bar, count, percentage in zip(bars, counts, percentages):
    height = percentage
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{percentage:.1f}%",
        ha="center",
        va="bottom",
    )

ax1.axvline(x=6868.5, color="red", linestyle="--", alpha=0.7)
ax1.set_xlabel("")  # Remove x-label from top plot
ax1.set_ylabel("Percentage Correct")
ax1.set_title(f"Response Correctness by Length for {model_id}")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Bottom subplot: Distribution of response lengths
ax2.hist(lengths, bins=filtered_bin_edges, alpha=0.7)
ax2.axvline(x=6868.5, color="red", linestyle="--", alpha=0.7)
ax2.set_xlabel("Response Length (characters)")
ax2.set_ylabel("Count")
ax2.grid(True, alpha=0.3)

# Set x-ticks at every bar position with formatted labels (only need to do this for ax2 now)
ax2.set_xticks(bin_centers)
ax2.set_xticklabels([f"{int(edge):,}" for edge in bin_centers], rotation=45)

# Ensure the x-axis limits are exactly the same
xlim = (
    min(bin_centers) - (filtered_bins[0][1] - filtered_bins[0][0]) / 2,
    max(bin_centers) + (filtered_bins[0][1] - filtered_bins[0][0]) / 2,
)
ax1.set_xlim(xlim)
ax2.set_xlim(xlim)

plt.tight_layout()
plt.show()

# %%
