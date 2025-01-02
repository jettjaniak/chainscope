# %%

import matplotlib.pyplot as plt
import pandas as pd

from chainscope.typing import *
from chainscope.utils import MODELS_MAP

df = pd.read_pickle(DATA_DIR / "df.pkl")
filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
df = df[~df.prop_id.isin(filter_prop_ids)]
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

# %%

# Direct responses are biased toward YES

# Filter for Gemma models and direct responses
gemma_df = df[
    df.model_id.isin([MODELS_MAP["G9"], MODELS_MAP["G27"]]) & (df["mode"] == "direct")
]

# show number of samples
print(len(gemma_df))

# Define colors for expected answers
colors = {
    "YES": "#4f97c4",  # blue
    "NO": "#c4774f",  # orange
}

plt.figure(figsize=(10, 5))

# Create boxplots for YES and NO answers
boxes = []  # Store box plots for legend
labels = []  # Store labels for legend
for i, answer in enumerate(["YES", "NO"]):
    answer_data = gemma_df[gemma_df["answer"] == answer]
    positions = [i * 2, i * 2 + 1]

    key_for_data = "p_yes" if answer == "YES" else "p_no"
    g9_data = answer_data[answer_data.model_id == MODELS_MAP["G9"]][key_for_data]
    g27_data = answer_data[answer_data.model_id == MODELS_MAP["G27"]][key_for_data]

    p_label = "P(YES)" if answer == "YES" else "P(NO)"

    bplot = plt.boxplot(
        [g9_data, g27_data],
        positions=positions,
        tick_labels=[
            f"Gemma-2-9B\n{p_label}",
            f"Gemma-2-27B\n{p_label}",
        ],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor=colors[answer], color="black"),
        medianprops=dict(color="red", linewidth=1.5),
    )

    boxes.append(bplot["boxes"][0])  # Add first box to legend
    labels.append(f"Expected answer: {answer}")

    # Add median values as text
    for idx, box in enumerate([g9_data, g27_data]):
        median = box.median()
        plt.text(
            positions[idx],
            median,
            f"{median:.2f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontweight="bold",
        )

plt.title("Gemma Models Performance on Direct Responses")
plt.ylabel("Probability")
plt.ylim(-0.04, 1.17)
plt.grid(True, alpha=0.3)

# Add legend
plt.legend(boxes, labels, loc="upper right")

plt.tight_layout()
plt.show()

# %%


# Create a figure showing bias patterns across models
def plot_model_biases(df: pd.DataFrame):
    # Filter for relevant models and calculate mean accuracies
    model_order = [
        "P",  # Phi
        "Q72",
        "Q32",
        "Q14",  # Qwens
        "G27",
        "G9",
        "G2",  # Gemmas
        "L70",
        "L8",
        "L3",
        "L1",  # Llamas
    ]

    model_labels = [
        "Phi 3.5",
        "Qwen 72B",
        "Qwen 32B",
        "Qwen 14B",
        "Gemma 27B",
        "Gemma 9B",
        "Gemma 2B",
        "Llama 70B",
        "Llama 8B",
        "Llama 3B",
        "Llama 1B",
    ]

    results = []
    for model_key in model_order:
        model_id = MODELS_MAP[model_key]
        for mode in ["direct", "cot"]:
            model_data = df[(df["model_id"] == model_id) & (df["mode"] == mode)]

            yes_acc = model_data[model_data["answer"] == "YES"].p_correct.mean()
            no_acc = model_data[model_data["answer"] == "NO"].p_correct.mean()

            results.append(
                {"model": model_key, "mode": mode, "yes_acc": yes_acc, "no_acc": no_acc}
            )

    results_df = pd.DataFrame(results)

    # Create the plot
    plt.figure(figsize=(12, 8))
    y_positions = range(len(model_order))

    for idx, model in enumerate(model_order):
        model_data = results_df[results_df["model"] == model]

        # Direct response arrow
        direct = model_data[model_data["mode"] == "direct"].iloc[0]
        cot = model_data[model_data["mode"] == "cot"].iloc[0]

        # Plot arrows
        plt.arrow(
            direct.yes_acc,
            idx - 0.1,
            direct.no_acc - direct.yes_acc,
            0,
            head_width=0.15,
            head_length=0.02,
            color="#4f97c4",
            length_includes_head=True,
            label="Direct" if idx == 0 else None,
        )

        plt.arrow(
            cot.yes_acc,
            idx + 0.1,
            cot.no_acc - cot.yes_acc,
            0,
            head_width=0.15,
            head_length=0.02,
            color="#c4774f",
            length_includes_head=True,
            label="CoT" if idx == 0 else None,
        )

    plt.yticks(y_positions, model_labels)
    plt.xlabel("Accuracy")
    plt.title("Model Biases: Average accuracy from YES to NO questions")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add vertical line at 0.5 for reference
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# Call the function
plot_model_biases(df)

# %%
