# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix

from chainscope.typing import *
from chainscope.utils import get_model_display_name

# %%

df = pd.read_pickle(DATA_DIR / "df.pkl")
filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
df = df[~df.prop_id.isin(filter_prop_ids)]
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

# %%

# Load all answer flipping eval files
answer_flipping_eval_files = list(DATA_DIR.glob("answer_flipping_eval/**/*.yaml"))

results: dict[
    str, dict[str, dict[str, dict[str, int]]]
] = {}  # model_id -> comparison -> prop_id -> {label: count}
response_uuids_with_flipping: set[str] = set()
response_uuids_without_flipping: set[str] = set()
for eval_file in answer_flipping_eval_files:
    eval_data = AnswerFlippingEval.load(eval_file)
    model_id = eval_data.model_id
    ds_params = eval_data.ds_params

    comparison = ds_params.comparison
    prop_id = ds_params.prop_id
    if prop_id in filter_prop_ids:
        continue

    if model_id not in results:
        results[model_id] = {}
    if comparison not in results[model_id]:
        results[model_id][comparison] = {}
    if prop_id not in results[model_id][comparison]:
        results[model_id][comparison][prop_id] = {}

    for qid, qid_labels in eval_data.label_by_qid.items():
        for response_uuid, label in qid_labels.items():
            if label not in results[model_id][comparison][prop_id]:
                results[model_id][comparison][prop_id][label] = 0
            results[model_id][comparison][prop_id][label] += 1

            if label == "YES":
                response_uuids_with_flipping.add(response_uuid)
            if label == "NO":
                response_uuids_without_flipping.add(response_uuid)

# %%

# Load faithfulness dataset
faithfulness_files = list(DATA_DIR.glob("faithfulness/*.yaml"))
faithfulness_data: dict[str, dict[str, dict]] = {}  # model_name -> qid -> data
for file in faithfulness_files:
    model_id = file.stem
    with open(file) as f:
        data = yaml.safe_load(f)
        if len(data) == 0:
            continue
        faithfulness_data[model_id] = data

# %%
# Create a DataFrame with answer flipping data for easier analysis
flipping_rows = []
for model_id, model_data in results.items():
    for comparison, comp_data in model_data.items():
        for prop_id, label_counts in comp_data.items():
            total = sum(label_counts.values())
            if total == 0:
                continue
            yes_percentage = (label_counts.get("YES", 0) / total) * 100
            flipping_rows.append(
                {
                    "model_id": model_id,
                    "model_name": get_model_display_name(model_id),
                    "comparison": comparison,
                    "prop_id": prop_id,
                    "flip_rate": yes_percentage,
                    "total_responses": total,
                    "yes_count": label_counts.get("YES", 0),
                    "no_count": label_counts.get("NO", 0),
                }
            )

flipping_df = pd.DataFrame(flipping_rows)

# Add bias information from original df
bias_by_group = df.groupby(["prop_id", "comparison"])["p_yes"].mean().reset_index()
bias_by_group["bias"] = bias_by_group["p_yes"] - 0.5
flipping_df = flipping_df.merge(bias_by_group, on=["prop_id", "comparison"])

# %%
# 1. Correlation between answer flipping and group bias
plt.figure(figsize=(10, 6))
plt.scatter(flipping_df["bias"].abs(), flipping_df["flip_rate"])
plt.xlabel("Absolute Group Bias (|p_yes - 0.5|)")
plt.ylabel("Answer Flipping Rate (%)")
plt.title("Answer Flipping Rate vs Group Bias")

correlation = flipping_df["bias"].abs().corr(flipping_df["flip_rate"])
plt.text(0.05, 0.95, f"Correlation: {correlation:.3f}", transform=plt.gca().transAxes)
plt.show()

# %%
# 2. Correlation between answer flipping and CoT accuracy difference
# First, create a DataFrame with accuracy differences for question pairs
pair_rows = []
for model_name, model_questions in faithfulness_data.items():
    for qid, qdata in model_questions.items():
        if "metadata" not in qdata:
            continue
        metadata = qdata["metadata"]

        # Get flipping data for this question
        matching_flip_data = flipping_df[
            (flipping_df["model_name"] == model_name)
            & (flipping_df["prop_id"] == metadata["prop_id"])
            & (flipping_df["comparison"] == metadata["comparison"])
        ]

        # Skip if no matching flipping data found
        if len(matching_flip_data) == 0:
            print(f"No matching flipping data found for {model_id} {qid}")
            continue

        flip_data = matching_flip_data.iloc[0]

        pair_rows.append(
            {
                "model_id": model_id,
                "qid": qid,
                "prop_id": metadata["prop_id"],
                "comparison": metadata["comparison"],
                "accuracy_diff": metadata["accuracy_diff"],
                "flip_rate": flip_data["flip_rate"],
                "is_unfaithful": len(qdata["unfaithful_responses"]) > 0,
                "p_correct": metadata["p_correct"],
                "reversed_p_correct": metadata["reversed_q_p_correct"],
            }
        )

if len(pair_rows) == 0:
    print("No pair rows found")
    raise ValueError("No pair rows found")

pair_df = pd.DataFrame(pair_rows)

# Print some basic stats about the accuracy differences
print("\nAccuracy Difference Statistics:")
print(pair_df["accuracy_diff"].describe())
print("\nNumber of pairs:", len(pair_df))

plt.figure(figsize=(10, 6))
plt.scatter(pair_df["accuracy_diff"], pair_df["flip_rate"])
plt.xlabel("Absolute Accuracy Difference")
plt.ylabel("Answer Flipping Rate (%)")
plt.title("Answer Flipping Rate vs Accuracy Difference")

correlation = pair_df["accuracy_diff"].corr(pair_df["flip_rate"])
plt.text(0.05, 0.95, f"Correlation: {correlation:.3f}", transform=plt.gca().transAxes)
plt.show()

# %%
# 3 & 4. Answer flipping rates for faithful vs unfaithful pairs
unfaithful_stats = pair_df[pair_df["is_unfaithful"]]["flip_rate"].describe()

print("Answer Flipping Stats for Unfaithful Pairs:")
print(unfaithful_stats)

# Visualize distribution
plt.figure(figsize=(10, 6))
plt.hist(
    [
        pair_df[pair_df["is_unfaithful"]]["flip_rate"],
    ],
    label=["Unfaithful pairs of Qs"],
    bins=20,
    alpha=0.5,
)
plt.xlabel("Answer Flipping Rate (%)")
plt.ylabel("Count")
plt.title("Distribution of Answer Flipping Rates in Unfaithful Pairs")
plt.legend()
plt.show()

# %%
# 5. Answer flipping with correct answers
correct_flip_rows = []
for model_name, model_questions in faithfulness_data.items():
    for qid, qdata in model_questions.items():
        if "metadata" not in qdata:
            continue
        metadata = qdata["metadata"]

        # Count flipped correct answers
        correct_responses = qdata["faithful_responses"]
        total_correct = len(correct_responses)

        if total_correct > 0:
            # Get flipping data for this question
            matching_flip_data = flipping_df[
                (flipping_df["model_name"] == model_name)
                & (flipping_df["prop_id"] == metadata["prop_id"])
                & (flipping_df["comparison"] == metadata["comparison"])
            ]

            # Skip if no matching flipping data found
            if len(matching_flip_data) == 0:
                continue

            flip_data = matching_flip_data.iloc[0]

            correct_flip_rows.append(
                {
                    "model_id": model_id,
                    "qid": qid,
                    "total_correct": total_correct,
                    "flip_rate": flip_data["flip_rate"],
                }
            )

correct_flip_df = pd.DataFrame(correct_flip_rows)

plt.figure(figsize=(10, 6))
plt.hist(correct_flip_df["flip_rate"], bins=20)
plt.xlabel("Answer Flipping Rate (%)")
plt.ylabel("Count")
plt.title("Answer Flipping Rates for Questions with Correct Answers")
plt.show()

print("\nAnswer Flipping Stats for Questions with Correct Answers:")
print(correct_flip_df["flip_rate"].describe())

# %%
# 6. Confusion matrix between answer flipping and faithfulness

# Create sets of response UUIDs for faithful/unfaithful responses
faithful_responses: set[str] = set()
unfaithful_responses: set[str] = set()

for model_name, model_questions in faithfulness_data.items():
    for qid, qdata in model_questions.items():
        if "metadata" not in qdata:
            continue
        faithful_responses |= qdata.get("faithful_responses", {}).keys()
        unfaithful_responses |= qdata.get("unfaithful_responses", {}).keys()

# Create true and predicted labels for confusion matrix
all_responses = (response_uuids_with_flipping | response_uuids_without_flipping) & (
    faithful_responses | unfaithful_responses
)

y_true = []
y_pred = []

for response_uuid in all_responses:
    # Swap order to match conventional confusion matrix layout
    y_true.append(1 if response_uuid in unfaithful_responses else 0)
    y_pred.append(1 if response_uuid in response_uuids_with_flipping else 0)

# Create and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["No Flipping", "Flipping"],
    yticklabels=["Faithful", "Unfaithful"],
    cmap="Blues",
)
plt.title("Confusion Matrix of Responses: Faithfulness vs Answer Flipping")
plt.ylabel("Faithfulness status")
plt.xlabel("Answer flipping status")
plt.show()

# Print some statistics
total = len(all_responses)
print(f"\nTotal responses analyzed: {total}")
print(f"Faithful responses: {len(faithful_responses & all_responses)}")
print(f"Unfaithful responses: {len(unfaithful_responses & all_responses)}")
print(f"Responses with flipping: {len(response_uuids_with_flipping & all_responses)}")
print(
    f"Responses without flipping: {len(response_uuids_without_flipping & all_responses)}"
)
