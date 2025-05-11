#!/usr/bin/env python3

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import to_rgb

from chainscope.typing import *

# %%

# Data from the pattern analysis
data = {
    'model': [
        'Llama-3.1-70B', 'Llama-3.3-70B-Instruct', 'chatgpt-4o-latest',
        'claude-3.5-haiku', 'claude-3.6-sonnet', 'claude-3.7-sonnet',
        'claude-3.7-sonnet_1k', 'deepseek-chat', 'deepseek-r1',
        'gemini-2.5-flash-preview', 'gemini-pro-1.5', 'gpt-4o-2024-08-06',
        'gpt-4o-mini', 'qwq-32b'
    ],
    'total_unfaithful_pairs': [159, 102, 15, 363, 12, 63, 2, 60, 13, 106, 320, 13, 660, 220],
    'fact-manipulation': [69.2, 80.4, 100.0, 64.7, 91.7, 14.3, 100.0, 73.3, 100.0, 35.8, 75.3, 84.6, 48.0, 99.5],
    'argument-switching': [46.5, 40.2, 6.7, 44.9, 0.0, 11.1, 0.0, 21.7, 7.7, 73.6, 27.5, 15.4, 76.4, 14.5],
    'answer-flipping': [58.5, 69.6, 20.0, 71.3, 33.3, 95.2, 0.0, 38.3, 15.4, 62.3, 46.6, 61.5, 76.4, 36.4],
    'other': [8.8, 8.8, 0.0, 14.0, 8.3, 52.4, 0.0, 11.7, 0.0, 3.8, 14.1, 7.7, 7.1, 0.0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set up the plot
plt.figure(figsize=(15, 8))
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
    card_height = 4  # Height of the card in percentage points
    card_width = 0.5 * width  # Width of the card (reduced from 2.5)
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
        fontsize=9,
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

# Customize the plot
plt.xlabel('Model')
plt.ylabel('Percentage of Question Pairs (%)')
plt.title('Distribution of Unfaithful Patterns Across Models')
plt.xticks(x, df['model'].tolist(), rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.grid(False, axis='x')

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()

# Save the plot
plt.savefig('unfaithful_patterns_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

# %%