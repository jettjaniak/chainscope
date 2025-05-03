# %%

from pathlib import Path

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
    'Percentage': [34.2, 14.2, 13.2, 0.7],
    'Dataset': ['non-ambiguous-hard'] * 4
}

# Data for non-ambiguous-hard-2 dataset
non_ambiguous_hard_2_qp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [63.6, 46.0, 70.7, 13.6],
    'Dataset': ['non-ambiguous-hard-2'] * 4
}

non_ambiguous_hard_2_resp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [26.1, 16.0, 14.4, 3.9],
    'Dataset': ['non-ambiguous-hard-2'] * 4
}

# Data for the second dataset (non-ambiguous-obscure-or-close-call-2)
non_ambiguous_obscure_or_close_call_2_qp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [32.7, 23.1, 92.3, 7.7],
    'Dataset': ['non-ambiguous-obscure-or-close-call-2'] * 4
}

non_ambiguous_obscure_or_close_call_2_resp_data = {
    'Pattern': ['fact-manipulation', 'argument-switching', 'answer-flipping', 'other'],
    'Percentage': [10.2, 7.2, 25.7, 0.6],
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
    'Percentage': [58.7, 14.7, 7.1, 0.7],
    'Dataset': ['wm-ambiguous'] * 4
}

# %%

# Create DataFrames
qp_df = pd.concat([
    pd.DataFrame(non_ambiguous_hard_qp_data), 
    pd.DataFrame(non_ambiguous_hard_2_qp_data),
    pd.DataFrame(non_ambiguous_obscure_or_close_call_2_qp_data),
    pd.DataFrame(wm_ambiguous_qp_data)
], ignore_index=True)

resp_df = pd.concat([
    pd.DataFrame(non_ambiguous_hard_resp_data), 
    pd.DataFrame(non_ambiguous_hard_2_resp_data),
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

# dataset_suffix = "non-ambiguous-hard-2"
dataset_suffix = None
model_name = "claude-3.5-haiku"

# %%

# Function to extract confidence scores from unfaithfulness pattern evaluation files
def extract_confidence_scores_old_scheme(base_dir, dataset_suffix, model_name):
    # Initialize data collection
    all_confidence_scores = []
    pattern_confidence_scores = {
        'fact-manipulation': [],
        'argument-switching': [],
        'answer-flipping': [],
        'other': [],
        'none': []
    }
    
    # Get all pattern evaluation directories for the specified dataset suffix
    if dataset_suffix is None:
        eval_files = list(Path(base_dir).glob(f"*/*.yaml"))
    else:
        eval_files = list(Path(base_dir).glob(f"*_{dataset_suffix}/{model_name}.yaml"))
    
    def process_responses(responses):
        for resp_id, resp_analysis in responses.items():
            # Add to overall confidence scores
            all_confidence_scores.append(resp_analysis["confidence"])
            
            # Add to pattern-specific confidence scores
            patterns_in_resp = set(resp_analysis["unfaithfulness_patterns"])

            assert len(resp_analysis["unfaithfulness_patterns"]) == len(patterns_in_resp), f"Duplicate patterns in response {resp_id}: {resp_analysis['unfaithfulness_patterns']}"
            assert "none" not in patterns_in_resp or len(patterns_in_resp) == 1, f"Response {resp_id} has none pattern mixed with other patterns: {resp_analysis['unfaithfulness_patterns']}"

            for pattern in patterns_in_resp:
                if pattern in pattern_confidence_scores:
                    pattern_confidence_scores[pattern].append(resp_analysis["confidence"])

    for eval_file in eval_files:
        if dataset_suffix is None and "_" in eval_file.parent.name:
            logging.warning(f"Skipping {eval_file} because dataset suffix is not provided and {eval_file.parent.name} contains _")
            continue

        logging.warning(f"Processing {eval_file}")

        # Load the evaluation file using yaml
        with open(eval_file, 'r') as f:
            eval_data = yaml.safe_load(f)
        
        # Extract confidence scores from all responses
        for qid, analysis in eval_data["pattern_analysis_by_qid"].items():
            if analysis["q1_analysis"] and analysis["q1_analysis"]["responses"]:
                process_responses(analysis["q1_analysis"]["responses"])
                            
            if analysis["q2_analysis"] and analysis["q2_analysis"]["responses"]:
                process_responses(analysis["q2_analysis"]["responses"])
    
    return all_confidence_scores, pattern_confidence_scores

def extract_confidence_scores_new_scheme(base_dir, dataset_suffix, model_name):
    # Initialize data collection
    all_confidence_scores = []
    pattern_confidence_scores = {
        'fact-manipulation': [],
        'argument-switching': [],
        'answer-flipping': [],
        'other': [],
        'none': []
    }
    
    # Get all pattern evaluation directories for the specified dataset suffix
    if dataset_suffix is None:
        eval_files = list(Path(base_dir).glob(f"*/*.yaml"))
    else:
        eval_files = list(Path(base_dir).glob(f"*_{dataset_suffix}/{model_name}.yaml"))
    
    def process_responses(responses):
        for resp_id, resp_analysis in responses.items():
            # Add to overall confidence scores
            all_confidence_scores.append(resp_analysis["confidence"])
            
            # Add to pattern-specific confidence scores
            patterns_in_resp = set(resp_analysis["evidence_of_unfaithfulness"])

            assert len(resp_analysis["evidence_of_unfaithfulness"]) == len(patterns_in_resp), f"Duplicate patterns in response {resp_id}: {resp_analysis['evidence_of_unfaithfulness']}"
            assert "none" not in patterns_in_resp or len(patterns_in_resp) == 1, f"Response {resp_id} has none pattern mixed with other patterns: {resp_analysis['evidence_of_unfaithfulness']}"

            for pattern in patterns_in_resp:
                if pattern in pattern_confidence_scores:
                    pattern_confidence_scores[pattern].append(resp_analysis["confidence"])

    for eval_file in eval_files:
        if dataset_suffix is None and "_" in eval_file.parent.name:
            logging.warning(f"Skipping {eval_file} because dataset suffix is not provided and {eval_file.parent.name} contains _")
            continue

        logging.warning(f"Processing {eval_file}")

        # Load the evaluation file using yaml
        with open(eval_file, 'r') as f:
            eval_data = yaml.safe_load(f)
        
        # Extract confidence scores from all responses
        for qid, analysis in eval_data["pattern_analysis_by_qid"].items():
            if analysis["q1_analysis"] and analysis["q1_analysis"]["responses"]:
                process_responses(analysis["q1_analysis"]["responses"])
                            
            if analysis["q2_analysis"] and analysis["q2_analysis"]["responses"]:
                process_responses(analysis["q2_analysis"]["responses"])
    
    return all_confidence_scores, pattern_confidence_scores


# Extract confidence scores
base_dir = Path(DATA_DIR) / "unfaithfulness_pattern_eval" / "T0.0_P0.9_M8000"
if dataset_suffix == "non-ambiguous-hard-2":
    all_scores, pattern_scores = extract_confidence_scores_new_scheme(base_dir, dataset_suffix, model_name)
else:
    all_scores, pattern_scores = extract_confidence_scores_old_scheme(base_dir, dataset_suffix, model_name)
del pattern_scores["none"]

# %%

# Create dataframe for plotting
confidence_df = pd.DataFrame({
    'Confidence': all_scores,
    'Pattern': ['All'] * len(all_scores)
})

# Add pattern-specific data
for pattern, scores in pattern_scores.items():
    if scores:  # Only include patterns with data
        pattern_df = pd.DataFrame({
            'Confidence': scores,
            'Pattern': [pattern.capitalize()] * len(scores)
        })
        confidence_df = pd.concat([confidence_df, pattern_df], ignore_index=True)

# %%

# Create a separate violin plot to better visualize the distribution by pattern
plt.figure(figsize=(12, 6))

# Set a specific order for the patterns with "All" first
pattern_order = ['All'] + sorted([p for p in confidence_df['Pattern'].unique() if p != 'All'])

# Calculate percentage of total responses for each pattern
total_responses = len(all_scores)
print(f"Total responses: {total_responses}")

pattern_percentages = {}
pattern_percentages['All'] = 100.0  # All is 100% as specified

for pattern in pattern_order:
    if pattern != 'All':
        pattern_count = len(confidence_df[confidence_df['Pattern'] == pattern])
        pattern_percentages[pattern] = (pattern_count / total_responses) * 100

# Create new x-tick labels with percentages
x_tick_labels = [f"{pattern}\n({pattern_percentages[pattern]:.1f}%)" for pattern in pattern_order]

# Create the violin plot
sns.violinplot(data=confidence_df, 
               x='Pattern', y='Confidence', palette='viridis',
               inner='quartile', order=pattern_order)

# Set the modified x-tick labels
plt.xticks(range(len(pattern_order)), x_tick_labels)

if dataset_suffix is not None:
    dataset_name = dataset_suffix
else:
    dataset_name = "wm-ambiguous"
plt.title(f'Confidence Score Distributions by Unfaithfulness Pattern for responses of {model_name} on {dataset_name}')
plt.xlabel('Unfaithfulness Pattern')
plt.ylabel('Confidence Score (1-10)')
plt.ylim(0, 11)
plt.yticks(range(0, 11, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %%