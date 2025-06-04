#!/usr/bin/env python3

# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chainscope.typing import *

# %%
# Set plot style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.2)

# Dataset suffix to filter by
dataset_suffix = "non-ambiguous-hard-2"

# %%
def collect_confidence_scores():
    """Collect confidence scores from all unfaithfulness pattern evaluations."""
    # Find all unfaithfulness pattern eval files
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_new_tokens=8000,
    )
    
    # Create directory pattern to search
    pattern_dir = DATA_DIR / "unfaithfulness_pattern_eval" / sampling_params.id
    
    # Get all pattern evaluation files (for all models)
    all_files = list(pattern_dir.glob("**/*.yaml"))
    print(f"Found {len(all_files)} total pattern evaluation files")
    
    # Filter files by dataset_suffix
    model_files = []
    for file_path in all_files:
        prop_dir = file_path.parent.name
        if dataset_suffix and prop_dir.endswith(f"_{dataset_suffix}"):
            model_files.append(file_path)
        elif not dataset_suffix and "_" not in prop_dir:
            model_files.append(file_path)
    
    print(f"After filtering for dataset_suffix '{dataset_suffix}': {len(model_files)} files")
    
    # Collect all confidence scores for answer flipping
    confidence_yes = []
    confidence_no = []
    
    # Process each file
    for file_path in model_files:
        try:
            # Load the pattern evaluation
            pattern_eval = UnfaithfulnessPatternEval.load_from_path(file_path)
            
            # Process each question analysis
            for qid, analysis in pattern_eval.pattern_analysis_by_qid.items():
                # Process Q1 responses
                if analysis.q1_analysis:
                    for resp_id, resp_analysis in analysis.q1_analysis.responses.items():
                        if resp_analysis.confidence is not None:
                            if hasattr(resp_analysis, 'answer_flipping_classification'):
                                # Add to appropriate collection based on classification
                                classification = resp_analysis.answer_flipping_classification
                                if classification == "YES":
                                    confidence_yes.append(resp_analysis.confidence)
                                elif classification == "NO":
                                    confidence_no.append(resp_analysis.confidence)
                
                # Process Q2 responses
                if analysis.q2_analysis:
                    for resp_id, resp_analysis in analysis.q2_analysis.responses.items():
                        if resp_analysis.confidence is not None:
                            if hasattr(resp_analysis, 'answer_flipping_classification'):
                                # Add to appropriate collection based on classification
                                classification = resp_analysis.answer_flipping_classification
                                if classification == "YES":
                                    confidence_yes.append(resp_analysis.confidence)
                                elif classification == "NO":
                                    confidence_no.append(resp_analysis.confidence)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Print statistics
    yes_percent = len(confidence_yes) / (len(confidence_yes) + len(confidence_no)) * 100 if (len(confidence_yes) + len(confidence_no)) > 0 else 0
    no_percent = len(confidence_no) / (len(confidence_yes) + len(confidence_no)) * 100 if (len(confidence_yes) + len(confidence_no)) > 0 else 0
    print(f"\nAnswer flipping statistics:")
    print(f"  YES: {len(confidence_yes)} scores ({yes_percent:.1f}%)")
    print(f"  NO: {len(confidence_no)} scores ({no_percent:.1f}%)")
    
    return confidence_yes, confidence_no

# %%
def create_violin_plot(confidence_yes, confidence_no):
    """Create a single figure with two violin plots side by side, combining data from all models."""
    # Convert to DataFrame for seaborn
    df_yes = pd.DataFrame({
        'Confidence Score (1-10)': confidence_yes,
        'Answer Flipping': 'YES'
    })
    
    df_no = pd.DataFrame({
        'Confidence Score (1-10)': confidence_no,
        'Answer Flipping': 'NO'
    })
    
    df = pd.concat([df_yes, df_no])
    
    # Calculate percentages
    total = len(df)
    yes_percent = len(df_yes) / total * 100
    no_percent = len(df_no) / total * 100
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot violin plots
    ax = sns.violinplot(x='Answer Flipping', y='Confidence Score (1-10)', 
                    data=df, inner='quartile', palette='viridis')
    
    # Set plot title and labels
    plt.title(f'Confidence Score Distributions by Answer Flipping Classification\nfor responses on {dataset_suffix}', fontsize=14)
    plt.xlabel('Answer Flipping Classification', fontsize=12)
    plt.ylabel('Confidence Score (1-10)', fontsize=12)
    
    # Add percentage labels under each category
    plt.text(0, 0, f"({yes_percent:.1f}%)", ha='center', va='top', fontsize=10)
    plt.text(1, 0, f"({no_percent:.1f}%)", ha='center', va='top', fontsize=10)
    
    # Add count labels
    plt.text(0, 10.5, f"n={len(confidence_yes)}", ha='center', fontsize=10)
    plt.text(1, 10.5, f"n={len(confidence_no)}", ha='center', fontsize=10)
    
    # Set y-axis range
    plt.ylim(0, 11)
    
    # Save the plot
    plt.tight_layout()
    path = DATA_DIR / "plots" / f"confidence_scores_by_answer_flipping_combined_{dataset_suffix}.pdf"
    plt.savefig(path, dpi=300)
    plt.show()

# %%
# Run the analysis
confidence_yes, confidence_no = collect_confidence_scores()

# %%
# Create the violin plot for answer flipping YES vs NO
create_violin_plot(confidence_yes, confidence_no)

