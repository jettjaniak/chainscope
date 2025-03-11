# %%

import matplotlib.pyplot as plt

# Updated data
models = [
    "Qwen 72B IT",
    "QwQ 32B",
    "Gemini Exp 1206",
    "Gemini 2.0 Flash Thinking Exp 1219",
    "DeepSeek V3",
    "DeepSeek R1",
]

# Calculate normalized percentages (2x = 200%, 1x = 100%)
correct_responses = [41, 115, 99, 154, 81, 172]  # Out of 215

# Define the unfaithful counts for each category
other_counts = {
    "Qwen 72B IT": 2,
    "QwQ 32B": 1,
    "Gemini Exp 1206": 0,
    "Gemini 2.0 Flash Thinking Exp 1219": 1,
    "DeepSeek V3": 0,
    "DeepSeek R1": 2,
}

sec_counts = {
    "Qwen 72B IT": 0,
    "QwQ 32B": 0,
    "Gemini Exp 1206": 2,
    "Gemini 2.0 Flash Thinking Exp 1219": 1,
    "DeepSeek V3": 0,
    "DeepSeek R1": 0,
}

# Calculate percentages normalized by correct responses
other_percentages = []
sec_percentages = []
for model in models:
    other_percentage = (
        other_counts[model] / correct_responses[models.index(model)]
    ) * 100
    sec_percentage = (sec_counts[model] / correct_responses[models.index(model)]) * 100
    other_percentages.append(other_percentage)
    sec_percentages.append(sec_percentage)

# Define color mapping based on model vendors
vendor_colors = {
    "Other": "#4e79a7",  # Standard blue
    "SEC": "#f28e2b",  # Standard orange
}

# Create figure
fig = plt.figure(figsize=(8, 6))

# Set font sizes
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

# Create stacked bars
ax = plt.gca()
bottom_bars = plt.bar(
    models, other_percentages, label="Other", color=vendor_colors["Other"]
)
top_bars = plt.bar(
    models,
    sec_percentages,
    bottom=other_percentages,
    label="Restoration Error",
    color=vendor_colors["SEC"],
)

# Rotate x-axis labels
plt.xticks(rotation=45, ha="right")
ax.set_xticklabels(
    [
        "Qwen 72B IT",
        "QwQ 32B",
        "Gemini Exp 1206",
        "Gemini 2.0 Flash\nThinking Exp 1219",
        "DeepSeek V3",
        "DeepSeek R1",
    ]
)

# Add legend
plt.legend()

# Add percentage labels on top of bars
for i in range(len(models)):
    total = other_percentages[i] + sec_percentages[i]
    if total > 0:
        ax.text(
            i,  # x position
            total + 0.2,  # y position (slightly above bar)
            f"{total:.2f}%",  # text to display
            ha="center",  # horizontal alignment
            va="bottom",  # vertical alignment
            fontsize=11,  # font size
        )

# Customize the plot
plt.ylabel("Unfaithful Responses (%)")
plt.ylim(0, 6)  # Increased upper limit slightly to fit labels
plt.yticks([0, 1, 2, 3, 4, 5, 6])

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()

# Save the plot
fig.savefig("putnam_unfaithfulness.pdf", dpi=300, bbox_inches="tight")
plt.close()
