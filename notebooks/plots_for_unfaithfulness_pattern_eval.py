#!/usr/bin/env python3

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%

# Data from the pattern analysis
data = [
    {
        'model': 'claude-3.5-haiku',
        'total_unfaithful_pairs': 363,
        'fact-manipulation': 64.7,
        'argument-switching': 44.9,
        'answer-flipping': 71.3,
        'other': 14.0
    },
    {
        'model': 'claude-3.6-sonnet',
        'total_unfaithful_pairs': 22,
        'fact-manipulation': 91.7,
        'argument-switching': 0.0,
        'answer-flipping': 33.3,
        'other': 8.3
    },
    {
        'model': 'claude-3.7-sonnet',
        'total_unfaithful_pairs': 90,
        'fact-manipulation': 14.3,
        'argument-switching': 11.1,
        'answer-flipping': 95.2,
        'other': 52.4
    },
    {
        'model': 'claude-3.7-sonnet_1k',
        'total_unfaithful_pairs': 2,
        'fact-manipulation': 100.0,
        'argument-switching': 0.0,
        'answer-flipping': 0.0,
        'other': 0.0
    },
    {
        'model': 'claude-3.7-sonnet_64k',
        'total_unfaithful_pairs': 12,
        'fact-manipulation': 77.8,
        'argument-switching': 22.2,
        'answer-flipping': 66.7,
        'other': 11.1
    },
    {
        'model': 'deepseek-chat',
        'total_unfaithful_pairs': 60,
        'fact-manipulation': 73.3,
        'argument-switching': 21.7,
        'answer-flipping': 38.3,
        'other': 11.7
    },
    {
        'model': 'deepseek-r1',
        'total_unfaithful_pairs': 18,
        'fact-manipulation': 100.0,
        'argument-switching': 7.7,
        'answer-flipping': 15.4,
        'other': 0.0
    },
    {
        'model': 'gpt-4o-mini',
        'total_unfaithful_pairs': 660,
        'fact-manipulation': 48.0,
        'argument-switching': 76.4,
        'answer-flipping': 76.4,
        'other': 7.1
    },
    {
        'model': 'gpt-4o-2024-08-06',
        'total_unfaithful_pairs': 18,
        'fact-manipulation': 84.6,
        'argument-switching': 15.4,
        'answer-flipping': 61.5,
        'other': 7.7
    },
    {
        'model': 'chatgpt-4o-latest',
        'total_unfaithful_pairs': 24,
        'fact-manipulation': 100.0,
        'argument-switching': 6.7,
        'answer-flipping': 20.0,
        'other': 0.0
    },
    {
        'model': 'gemini-pro-1.5',
        'total_unfaithful_pairs': 320,
        'fact-manipulation': 75.3,
        'argument-switching': 27.5,
        'answer-flipping': 46.6,
        'other': 14.1
    },
    {
        'model': 'gemini-2.5-flash-preview',
        'total_unfaithful_pairs': 106,
        'fact-manipulation': 35.8,
        'argument-switching': 73.6,
        'answer-flipping': 62.3,
        'other': 3.8
    },
    {
        'model': 'gemini-2.5-pro-preview',
        'total_unfaithful_pairs': 7,
        'fact-manipulation': 100.0,
        'argument-switching': 0.0,
        'answer-flipping': 0.0,
        'other': 0.0
    },
    {
        'model': 'Llama-3.1-70B',
        'total_unfaithful_pairs': 159,
        'fact-manipulation': 69.2,
        'argument-switching': 46.5,
        'answer-flipping': 58.5,
        'other': 8.8
    },
    {
        'model': 'Llama-3.3-70B-Instruct',
        'total_unfaithful_pairs': 102,
        'fact-manipulation': 80.4,
        'argument-switching': 40.2,
        'answer-flipping': 69.6,
        'other': 8.8
    },
    {
        'model': 'qwq-32b',
        'total_unfaithful_pairs': 220,
        'fact-manipulation': 99.5,
        'argument-switching': 14.5,
        'answer-flipping': 36.4,
        'other': 0.0
    }
]

# Compute flattened data from the data list
flattened_data = {
    'model': [entry['model'] for entry in data],
    'total_unfaithful_pairs': [entry['total_unfaithful_pairs'] for entry in data],
    'fact-manipulation': [entry['fact-manipulation'] for entry in data],
    'argument-switching': [entry['argument-switching'] for entry in data],
    'answer-flipping': [entry['answer-flipping'] for entry in data],
    'other': [entry['other'] for entry in data]
}

# Create DataFrame
df = pd.DataFrame(flattened_data)

# Set up the plot
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# Create x positions with increased spacing between model groups
group_spacing = 1.5  # Increase this value to add more space between model groups
x = np.arange(len(df['model'])) * group_spacing
width = 0.2  # Keep width the same

# Plot bars
plt.bar(x - 1.5*width, df['fact-manipulation'], width, label='Fact Manipulation', color='#2ecc71')
plt.bar(x - 0.5*width, df['argument-switching'], width, label='Argument Switching', color='#e74c3c')
plt.bar(x + 0.5*width, df['answer-flipping'], width, label='Answer Flipping', color='#3498db')
plt.bar(x + 1.5*width, df['other'], width, label='Other', color='#f1c40f')

# Add floating labels with card-like backgrounds
for i, (model, total) in enumerate(zip(df['model'], df['total_unfaithful_pairs'])):
    # Find the maximum height of bars for this model
    max_height = max(
        df['fact-manipulation'].iloc[i],
        df['argument-switching'].iloc[i],
        df['answer-flipping'].iloc[i],
        df['other'].iloc[i]
    )
    
    # Create card-like background
    card_height = 6  # Height of the card in percentage points
    card_width = 3 * width  # Width of the card (reduced from 2.5)
    card_x = x[i] - card_width/2
    card_y = max_height + 2  # Position above the highest bar with some padding
    
    # Add rectangle with rounded corners
    rect = patches.FancyBboxPatch(
        (card_x, card_y),
        card_width,
        card_height,
        boxstyle=patches.BoxStyle("Round"),
        facecolor='white',
        edgecolor='#404040',
        alpha=0.9,
        zorder=3
    )
    plt.gca().add_patch(rect)
    
    # Add text
    plt.text(
        x[i],
        card_y + card_height/2,
        f'n={total}',
        ha='center',
        va='center',
        fontsize=14,
        zorder=4
    )

# Add vertical separator lines between models, centered between groups
# Add separator for the first group
first_separator = x[0] - (x[1] - x[0])/2
plt.axvline(first_separator, color='#404040', linestyle='--', alpha=0.7)

# Add separators between groups
for i in range(len(x) - 1):
    midpoint = (x[i] + x[i+1]) / 2
    plt.axvline(midpoint, color='#404040', linestyle='--', alpha=0.7)

# Add separator for the last group
last_separator = x[-1] + (x[-1] - x[-2])/2
plt.axvline(last_separator, color='#404040', linestyle='--', alpha=0.7)

# Mapping from internal model names to display names (as in the provided figure)
MODEL_DISPLAY_NAMES = {
    'claude-3.5-haiku': 'Haiku 3.5',
    'claude-3.6-sonnet': 'Sonnet 3.5 v2',
    'claude-3.7-sonnet': 'Sonnet 3.7',
    'claude-3.7-sonnet_1k': 'Sonnet 3.7 (1k)',
    'claude-3.7-sonnet_64k': 'Sonnet 3.7 (64k)',
    'deepseek-chat': 'DeepSeek V3',
    'deepseek-r1': 'DeepSeek R1',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4o-2024-08-06': "GPT-4o Aug '24",
    'chatgpt-4o-latest': 'ChatGPT-4o',
    'gemini-pro-1.5': 'Gemini 1.5 Pro',
    'gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'Llama-3.1-70B': 'Llama-3.1-70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B It',
    'qwq-32b': 'QwQ 32B',
}

# Customize the plot
plt.xlabel('Model', fontsize=16)
plt.ylabel('Frequency of Patterns in Unfaithful Question Pairs (%)', fontsize=16)
plt.title('Distribution of Unfaithful Patterns Across Models', fontsize=16)
# Use display names for x-axis labels
xtick_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in df['model'].tolist()]
plt.xticks(x, xtick_labels, rotation=45, ha='right', fontsize=16)
plt.ylim(0, 150)
plt.yticks(np.arange(0, 110, 10), fontsize=16)
plt.legend(loc='upper right', fontsize=16, frameon=True)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.grid(False, axis='x')

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()

# Save the plot
plt.savefig('unfaithful_patterns_distribution.pdf', bbox_inches='tight', dpi=300)
plt.close()

# %%