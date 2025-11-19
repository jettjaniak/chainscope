#!/usr/bin/env python3

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chainscope.typing import DATA_DIR

# %%

# Data from the pattern analysis
data = [
    {
        "model": "claude-3.5-haiku",
        "total_unfaithful_pairs": 363,
        "fact-manipulation": 67.2,
        "argument-switching": 44.6,
        "answer-flipping": 67.5,
        "other": 5.2,
    },
    {
        "model": "claude-3.6-sonnet",
        "total_unfaithful_pairs": 22,
        "fact-manipulation": 91.7,
        "argument-switching": 8.3,
        "answer-flipping": 25.0,
        "other": 8.3,
    },
    {
        "model": "claude-3.7-sonnet",
        "total_unfaithful_pairs": 90,
        "fact-manipulation": 14.3,
        "argument-switching": 7.9,
        "answer-flipping": 93.7,
        "other": 27.0,
    },
    {
        "model": "claude-3.7-sonnet_1k",
        "total_unfaithful_pairs": 2,
        "fact-manipulation": 100.0,
        "argument-switching": 0.0,
        "answer-flipping": 0.0,
        "other": 0.0,
    },
    {
        "model": "claude-3.7-sonnet_64k",
        "total_unfaithful_pairs": 12,
        "fact-manipulation": 77.8,
        "argument-switching": 11.1,
        "answer-flipping": 55.6,
        "other": 0.0,
    },
    {
        "model": "deepseek-chat",
        "total_unfaithful_pairs": 60,
        "fact-manipulation": 68.3,
        "argument-switching": 21.7,
        "answer-flipping": 31.7,
        "other": 1.7,
    },
    {
        "model": "deepseek-r1",
        "total_unfaithful_pairs": 18,
        "fact-manipulation": 100.0,
        "argument-switching": 15.4,
        "answer-flipping": 7.7,
        "other": 0.0,
    },
    {
        "model": "gpt-4o-mini",
        "total_unfaithful_pairs": 660,
        "fact-manipulation": 51.2,
        "argument-switching": 78.3,
        "answer-flipping": 68.8,
        "other": 1.8,
    },
    {
        "model": "gpt-4o-2024-08-06",
        "total_unfaithful_pairs": 18,
        "fact-manipulation": 92.3,
        "argument-switching": 23.1,
        "answer-flipping": 76.9,
        "other": 0.0,
    },
    {
        "model": "chatgpt-4o-latest",
        "total_unfaithful_pairs": 24,
        "fact-manipulation": 100.0,
        "argument-switching": 0.0,
        "answer-flipping": 6.7,
        "other": 0.0,
    },
    {
        "model": "gemini-pro-1.5",
        "total_unfaithful_pairs": 320,
        "fact-manipulation": 76.2,
        "argument-switching": 28.1,
        "answer-flipping": 45.6,
        "other": 7.5,
    },
    {
        "model": "gemini-2.5-flash-preview",
        "total_unfaithful_pairs": 106,
        "fact-manipulation": 35.8,
        "argument-switching": 76.4,
        "answer-flipping": 58.5,
        "other": 0.9,
    },
    {
        "model": "gemini-2.5-pro-preview",
        "total_unfaithful_pairs": 7,
        "fact-manipulation": 100.0,
        "argument-switching": 0.0,
        "answer-flipping": 0.0,
        "other": 0.0,
    },
    {
        "model": "Llama-3.1-70B",
        "total_unfaithful_pairs": 159,
        "fact-manipulation": 73.0,
        "argument-switching": 49.7,
        "answer-flipping": 50.3,
        "other": 5.0,
    },
    {
        "model": "Llama-3.3-70B-Instruct",
        "total_unfaithful_pairs": 102,
        "fact-manipulation": 88.2,
        "argument-switching": 41.2,
        "answer-flipping": 17.6,
        "other": 1.0,
    },
    {
        "model": "qwq-32b",
        "total_unfaithful_pairs": 220,
        "fact-manipulation": 99.1,
        "argument-switching": 19.1,
        "answer-flipping": 25.9,
        "other": 0.5,
    },
]

# Compute flattened data from the data list
flattened_data = {
    "model": [entry["model"] for entry in data],
    "total_unfaithful_pairs": [entry["total_unfaithful_pairs"] for entry in data],
    "fact-manipulation": [entry["fact-manipulation"] for entry in data],
    "argument-switching": [entry["argument-switching"] for entry in data],
    "answer-flipping": [entry["answer-flipping"] for entry in data],
    "other": [entry["other"] for entry in data],
}

# Create DataFrame
df = pd.DataFrame(flattened_data)

N_BOOTSTRAPS = 2000
pattern_columns = [
    "fact-manipulation",
    "argument-switching",
    "answer-flipping",
    "other",
]

rng = np.random.default_rng(0)
pattern_ci_bounds: dict[str, dict[str, list[float]]] = {
    pattern_key: {"low": [], "high": []} for pattern_key in pattern_columns
}


def _pattern_yerr(pattern_name: str) -> np.ndarray:
    values = df[pattern_name].to_numpy()
    lows = np.array(pattern_ci_bounds[pattern_name]["low"])
    highs = np.array(pattern_ci_bounds[pattern_name]["high"])
    return np.vstack([values - lows, highs - values])


for _, row in df.iterrows():
    total_pairs = int(row["total_unfaithful_pairs"])
    assert total_pairs > 0, "Each model must have at least one unfaithful pair."
    for pattern_key in pattern_columns:
        success_prob = np.clip(row[pattern_key] / 100.0, 0.0, 1.0)
        bootstrap_counts = rng.binomial(total_pairs, success_prob, size=N_BOOTSTRAPS)
        bootstrap_rates = bootstrap_counts / total_pairs
        ci_low, ci_high = np.percentile(bootstrap_rates, [2.5, 97.5]) * 100
        pattern_ci_bounds[pattern_key]["low"].append(ci_low)
        pattern_ci_bounds[pattern_key]["high"].append(ci_high)

pattern_yerr = {
    pattern_key: _pattern_yerr(pattern_key) for pattern_key in pattern_columns
}

print("\nConfidence intervals for unfaithful pattern rates (percent).")
for pattern_key in pattern_columns:
    print(f"\n{pattern_key.title()}:")
    print(f"{'Model':<30} | {'Rate':<8} | {'95% CI Low':<10} | {'95% CI High':<10}")
    print("-" * 80)
    for idx, model in enumerate(df["model"]):
        rate = df[pattern_key].iloc[idx]
        ci_low = pattern_ci_bounds[pattern_key]["low"][idx]
        ci_high = pattern_ci_bounds[pattern_key]["high"][idx]
        print(f"{model:<30} | {rate:>6.2f}% | {ci_low:>10.2f}% | {ci_high:>10.2f}%")

# Set up the plot
plt.figure(figsize=(16, 6))
sns.set_style("whitegrid")

# Create x positions with increased spacing between model groups
group_spacing = 1.5  # Increase this value to add more space between model groups
x = np.arange(len(df["model"])) * group_spacing
width = 0.3  # Keep width the same

# Plot bars
plt.bar(
    x - 1.5 * width,
    df["fact-manipulation"],
    width,
    label="Fact Manipulation",
    color="#2ecc71",
    yerr=pattern_yerr["fact-manipulation"],
    capsize=3,
)
plt.bar(
    x - 0.5 * width,
    df["argument-switching"],
    width,
    label="Argument Switching",
    color="#e74c3c",
    yerr=pattern_yerr["argument-switching"],
    capsize=3,
)
plt.bar(
    x + 0.5 * width,
    df["answer-flipping"],
    width,
    label="Answer Flipping",
    color="#3498db",
    yerr=pattern_yerr["answer-flipping"],
    capsize=3,
)
plt.bar(
    x + 1.5 * width,
    df["other"],
    width,
    label="Other",
    color="#f1c40f",
    yerr=pattern_yerr["other"],
    capsize=3,
)

# Add floating labels with card-like backgrounds
for i, (model, total) in enumerate(zip(df["model"], df["total_unfaithful_pairs"])):
    # Find the maximum height of bars + error bars for this model
    fact_high = (
        df["fact-manipulation"].iloc[i]
        + pattern_ci_bounds["fact-manipulation"]["high"][i]
        - df["fact-manipulation"].iloc[i]
    )
    arg_high = (
        df["argument-switching"].iloc[i]
        + pattern_ci_bounds["argument-switching"]["high"][i]
        - df["argument-switching"].iloc[i]
    )
    flip_high = (
        df["answer-flipping"].iloc[i]
        + pattern_ci_bounds["answer-flipping"]["high"][i]
        - df["answer-flipping"].iloc[i]
    )
    other_high = (
        df["other"].iloc[i]
        + pattern_ci_bounds["other"]["high"][i]
        - df["other"].iloc[i]
    )

    # We want the visual top, which is the CI high bound
    max_visual_height = max(
        pattern_ci_bounds["fact-manipulation"]["high"][i],
        pattern_ci_bounds["argument-switching"]["high"][i],
        pattern_ci_bounds["answer-flipping"]["high"][i],
        pattern_ci_bounds["other"]["high"][i],
    )

    # Create card-like background
    card_height = 6  # Height of the card in percentage points
    card_width = 2 * width  # Width of the card (reduced from 2.5)
    card_x = x[i] - card_width / 2

    # Position slightly above the highest error bar
    card_y = max_visual_height + 2

    # Ensure bubbles are not cut off
    if card_y + card_height > 118:
        card_y = 118 - card_height - 1

    # Add rectangle with rounded corners
    rect = patches.FancyBboxPatch(
        (card_x, card_y),
        card_width,
        card_height,
        boxstyle=patches.BoxStyle("Round"),
        facecolor="white",
        edgecolor="#404040",
        alpha=0.9,
        zorder=3,
    )
    plt.gca().add_patch(rect)

    # Add text
    plt.text(
        x[i],
        card_y + card_height / 2,
        f"n={total}",
        ha="center",
        va="center",
        fontsize=14,
        zorder=4,
    )

# Set y-axis limit to accommodate the higher labels
plt.ylim(0, 113)

# Set x-axis limits to remove padding
plt.xlim(x[0] - width * 3, x[-1] + width * 3)

# Add separators between groups
for i in range(len(x) - 1):
    midpoint = (x[i] + x[i + 1]) / 2
    plt.axvline(midpoint, color="#404040", linestyle="--", alpha=0.7)

# Mapping from internal model names to display names (as in the provided figure)
MODEL_DISPLAY_NAMES = {
    "claude-3.5-haiku": "Haiku 3.5",
    "claude-3.6-sonnet": "Sonnet 3.5 v2",
    "claude-3.7-sonnet": "Sonnet 3.7",
    "claude-3.7-sonnet_1k": "Sonnet 3.7 (1k)",
    "claude-3.7-sonnet_64k": "Sonnet 3.7 (64k)",
    "deepseek-chat": "DeepSeek V3",
    "deepseek-r1": "DeepSeek R1",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o-2024-08-06": "GPT-4o Aug '24",
    "chatgpt-4o-latest": "ChatGPT-4o",
    "gemini-pro-1.5": "Gemini 1.5 Pro",
    "gemini-2.5-flash-preview": "Gemini 2.5 Flash",
    "gemini-2.5-pro-preview": "Gemini 2.5 Pro",
    "Llama-3.1-70B": "Llama-3.1-70B",
    "Llama-3.3-70B-Instruct": "Llama 3.3 70B It",
    "qwq-32b": "QwQ 32B",
}

# Customize the plot
plt.xlabel("Model", fontsize=24)
plt.ylabel("Frequency of Patterns\nin Unfaithful Pairs (%)", fontsize=20)
# plt.title('Distribution of Unfaithful Patterns Across Models', fontsize=16)
# Use display names for x-axis labels
xtick_labels = [
    MODEL_DISPLAY_NAMES[m] if m in MODEL_DISPLAY_NAMES else m
    for m in df["model"].tolist()
]
plt.xticks(x, xtick_labels, rotation=45, ha="right", fontsize=16)
plt.yticks(np.arange(0, 101, 20), fontsize=16)
plt.legend(
    bbox_to_anchor=(0.5, 1),
    loc="lower center",
    fontsize=16,
    frameon=True,
    ncol=4,
    columnspacing=1.0,
)
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.grid(False, axis="x")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_dir = DATA_DIR / ".." / ".." / "plots" / "all_models"
plt.savefig(
    output_dir / "unfaithful_patterns_distribution.pdf", bbox_inches="tight", dpi=300
)

plt.show()
plt.close()

# %%

# Create a new plot focusing only on answer flipping
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# Create x positions
x = np.arange(len(df["model"]))
width = 0.6

# Plot bars for answer flipping only
bars = plt.bar(
    x, df["answer-flipping"], width, label="Answer Flipping", color="#3498db"
)

# Add text labels on top of each bar
for i, (model, percentage, total) in enumerate(
    zip(df["model"], df["answer-flipping"], df["total_unfaithful_pairs"])
):
    plt.text(
        x[i],
        percentage + 1,  # Position slightly above the bar
        f"{percentage}%\nn={total}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Customize the plot
plt.xlabel("Model", fontsize=24)
plt.ylabel("Answer Flipping Frequency\nin Unfaithful Pairs (%)", fontsize=20)
plt.title("Answer Flipping Patterns Across Models", fontsize=24, pad=20)

# Use display names for x-axis labels
xtick_labels = [
    MODEL_DISPLAY_NAMES[m] if m in MODEL_DISPLAY_NAMES else m
    for m in df["model"].tolist()
]
plt.xticks(x, xtick_labels, rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)

# Set y-axis to go from 0 to 100 with some padding for the text labels
plt.ylim(0, 110)

plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.grid(False, axis="x")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(output_dir / "answer_flipping_patterns.pdf", bbox_inches="tight", dpi=300)

plt.show()

# %%
