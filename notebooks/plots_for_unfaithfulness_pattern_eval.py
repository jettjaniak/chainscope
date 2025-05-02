# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from chainscope.typing import *

# %%

# Data for the first dataset (non-ambiguous-hard)
non_ambiguous_hard_qp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [75.3, 45.0, 67.1, 8.9],
    'Dataset': ['non-ambiguous-hard'] * 4
}

non_ambiguous_hard_resp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [54.9, 22.8, 21.2, 1.1],
    'Dataset': ['non-ambiguous-hard'] * 4
}

# Data for the second dataset (non-ambiguous-obscure-or-close-call-2)
non_ambiguous_obscure_or_close_call_2_qp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [32.7, 23.1, 92.3, 7.7],
    'Dataset': ['non-ambiguous-obscure-or-close-call-2'] * 4
}

non_ambiguous_obscure_or_close_call_2_resp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [23.3, 16.5, 58.8, 1.3],
    'Dataset': ['non-ambiguous-obscure-or-close-call-2'] * 4
}

# Data for the third dataset (wm-ambiguous)
wm_ambiguous_qp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [93.5, 48.4, 49.5, 9.6],
    'Dataset': ['wm-ambiguous'] * 4
}

# Calculate percentages for responses
total_responses = 14559 + 3649 + 1770 + 181
wm_ambiguous_resp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [14559/total_responses*100, 3649/total_responses*100, 1770/total_responses*100, 181/total_responses*100],
    'Dataset': ['wm-ambiguous'] * 4
}

# %%

# Create DataFrames
qp_df = pd.concat([
    pd.DataFrame(non_ambiguous_hard_qp_data), 
    pd.DataFrame(non_ambiguous_obscure_or_close_call_2_qp_data),
    pd.DataFrame(wm_ambiguous_qp_data)
], ignore_index=True)

resp_df = pd.concat([
    pd.DataFrame(non_ambiguous_hard_resp_data), 
    pd.DataFrame(non_ambiguous_obscure_or_close_call_2_resp_data),
    pd.DataFrame(wm_ambiguous_resp_data)
], ignore_index=True)

# %%

# Set style
plt.style.use("seaborn-v0_8-white")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot for question pairs
sns.barplot(data=qp_df, x='Pattern', y='Percentage', hue='Dataset', ax=ax1)
ax1.set_title('Unfaithfulness Patterns in Question Pairs Marked as Unfaithful')
ax1.set_ylabel('Percentage (%)')
ax1.set_yticks(np.arange(0, 101, 10))
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, color='grey', linestyle='-', alpha=0.2)
ax1.legend().remove()


# Plot for responses
sns.barplot(data=resp_df, x='Pattern', y='Percentage', hue='Dataset', ax=ax2)
ax2.set_title('Unfaithfulness Patterns in Responses from Question Pairs Marked as Unfaithful')
ax2.set_ylabel('Percentage (%)')
ax2.set_yticks(np.arange(0, 101, 10))
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, color='grey', linestyle='-', alpha=0.2)
ax2.legend(title='Dataset')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# %%
