# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from chainscope.typing import *
from chainscope.utils import (
    get_model_family,
    sort_models,
)

# %% Load all ambiguity eval YAMLs
yaml_pattern = "ambiguity_eval/T0.7_P0.9_M1000/*/*.yaml"
yaml_paths = list(DATA_DIR.glob(yaml_pattern))
print(f"Found {len(yaml_paths)} YAML files")

# %% Process each YAML and collect stats
dataset_stats = []

# After loading the ambiguity eval YAMLs, store the CLEAR questions in a set
clear_questions = set()
all_questions = set()
for path in yaml_paths:
    try:
        eval_data = AmbiguityEval.load(path)
        clear_questions.update(
            qid
            for qid, status in eval_data.final_ambiguity_by_qid.items()
            if status == "CLEAR"
        )
        all_questions.update(qid for qid in eval_data.final_ambiguity_by_qid.keys())
        # Count CLEAR questions
        total_qs = len(eval_data.final_ambiguity_by_qid)
        clear_qs = sum(
            1
            for status in eval_data.final_ambiguity_by_qid.values()
            if status == "CLEAR"
        )

        # Calculate percentage
        clear_percentage = (clear_qs / total_qs) * 100 if total_qs > 0 else 0

        # Get dataset identifier from path
        dataset_id = path.parts[-1]

        dataset_stats.append(
            {
                "dataset_id": dataset_id,
                "total_questions": total_qs,
                "clear_questions": clear_qs,
                "clear_percentage": clear_percentage,
            }
        )
    except Exception as e:
        print(f"Error processing {path}: {e}")
        break

print(
    f"Found {len(clear_questions)} CLEAR questions out of {len(all_questions)} total questions"
)

# %% Create DataFrame and sort by clear percentage
df = pd.DataFrame(dataset_stats)
df_sorted = df.sort_values("clear_percentage", ascending=False)

# %% Display results
pd.set_option("display.max_rows", None)
print("\nDatasets sorted by percentage of CLEAR questions:")
print(df_sorted.to_string(index=False, float_format=lambda x: "%.1f" % x))

# %%

df = pd.read_pickle(DATA_DIR / "df-wm.pkl")
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

faithfulness_yamls_cache = {}

# %%


def save_iphr_plot(df: pd.DataFrame, save_dir: Path) -> None:
    """Save a bar plot showing the percentage of unfaithful question pairs for each model."""
    # Define vendor colors
    vendor_colors = {
        "anthropic": "#d4a27f",  # Orange/red/rust color for Sonnet
        "deepseek": "#6B4CF6",  # Purple for DeepSeek
        "google": "#2D87F3",  # Sky Blue for Gemini
        "meta-llama": "#1E3D8C",  # Darker Blue for Llama
        "openai": "#00A67E",  # Green for GPT
    }

    # Load faithfulness data for each model
    results = []
    for model_id in sort_models(df["model_id"].unique().tolist()):
        # Skip models we don't want to include
        if "0.5B" in model_id:
            continue

        # How many prop_ids do we have in df for this model?
        num_prop_ids = len(df[df["model_id"] == model_id]["prop_id"].unique())
        if num_prop_ids <= 3:
            print(f"Skipping {model_id} because it has only {num_prop_ids} prop_ids")
            continue

        # Convert model ID to filename format
        model_dir_name = model_id.split("/")[-1]
        faith_dir = DATA_DIR / "faithfulness" / model_dir_name

        # Check if directory exists
        if not faith_dir.exists() or not faith_dir.is_dir():
            print(
                f"Faithfulness directory not found for {model_id}. Expected at {faith_dir}"
            )
            continue

        # Initialize merged data dictionary
        merged_faith_data = {}

        # Check if we've already cached this model's data
        if model_id in faithfulness_yamls_cache:
            merged_faith_data = faithfulness_yamls_cache[model_id]
        else:
            # Get all YAML files in the directory
            yaml_files = list(faith_dir.glob("*.yaml"))
            if not yaml_files:
                print(
                    f"No faithfulness YAML files found in directory for {model_id}: {faith_dir}"
                )
                continue

            # Load and merge all YAML files
            print(
                f"Loading faithfulness data for {model_id} from {len(yaml_files)} files in {faith_dir}"
            )
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file) as f:
                        faith_data = yaml.safe_load(f)
                        if faith_data:
                            merged_faith_data.update(faith_data)
                except Exception as e:
                    print(f"Error loading {yaml_file}: {e}")

            # Cache the merged data
            faithfulness_yamls_cache[model_id] = merged_faith_data

        # After loading merged_faith_data but before counting unfaithful pairs
        # Filter out non-CLEAR questions
        merged_faith_data = {
            qid: data
            for qid, data in merged_faith_data.items()
            if qid in clear_questions
        }

        # Count unfaithful pairs
        unfaithful_count = len(merged_faith_data.keys())

        # Calculate percentage (adjust total number to only count CLEAR questions)
        percentage = float((unfaithful_count / len(clear_questions)) * 100)

        # Correct special case for Sonnet 3.5
        if model_id == "claude-3-5-sonnet-20241022":
            model_id = "anthropic/claude-3.5-sonnet"

        # Get model vendor and name
        vendor = get_model_family(model_id)

        # Map model names consistently
        model_name = model_id.split("/")[-1]
        if "claude" in model_id.lower():
            if "sonnet" in model_id.lower():
                model_name = "Sonnet " + model_id.split("-")[1]
                if "_" in model_id:
                    model_name = model_name + f" ({model_id.split('_')[1]})"
                elif "3.5" in model_id:
                    model_name = model_name + " v2"
            elif "haiku" in model_id.lower():
                model_name = "Haiku " + model_id.split("-")[1]
            else:
                model_name = model_id
        elif "deepseek-r1" in model_id.lower():
            model_name = "DeepSeek R1"
        elif "deepseek-chat" in model_id.lower():
            model_name = "DeepSeek V3"
        elif "gemini-pro-1.5" in model_id.lower():
            model_name = "Gemini Pro 1.5"
        elif "llama-3.3-70b-instruct" in model_id.lower():
            model_name = "Llama 3.3 70B It"
        elif "gpt-4" in model_id.lower():
            if "latest" in model_id.lower():
                model_name = "ChatGPT-4o"
            else:
                model_name = "GPT-4o Aug '24"

        results.append(
            {
                "model": model_id,
                "xtick": model_name,
                "vendor": vendor,
                "unfaithful_count": unfaithful_count,
                "percentage": percentage,
            }
        )

    if not results:
        print("No faithfulness data found")
        return

    # Create DataFrame and ensure numeric types
    plot_data = pd.DataFrame(results)
    plot_data["percentage"] = pd.to_numeric(plot_data["percentage"])

    # Use white background
    plt.style.use("seaborn-v0_8-white")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create empty lists to store legend handles
    legend_handles = []
    seen_vendors = set()

    separation = 0.01
    width = 0.05

    # Calculate x positions for bars
    x_positions = [i * width + i * separation for i in range(len(plot_data))]

    for i, row in enumerate(plot_data.itertuples()):
        color = vendor_colors[row.vendor]
        bar = ax.bar(
            x_positions[i],
            row.percentage,
            width,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Add to legend only if we haven't seen this vendor yet
        if row.vendor not in seen_vendors:
            # Map vendor names to display names
            vendor_display_names = {
                "anthropic": "Anthropic",
                "deepseek": "DeepSeek",
                "google": "Google",
                "meta-llama": "Meta",
                "openai": "OpenAI",
            }
            legend_handles.append((bar, vendor_display_names[row.vendor]))
            seen_vendors.add(row.vendor)

    # Add legend
    ax.legend(
        [h[0] for h in legend_handles],
        [h[1] for h in legend_handles],
        loc="upper right",
        frameon=True,
        fontsize=16,
    )

    # Add percentage labels on top of bars
    def add_labels(position, value):
        ax.text(
            position,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    for i, row in enumerate(plot_data.itertuples()):
        add_labels(x_positions[i], row.percentage)

    # Set xticks at the bar positions
    plt.xticks(x_positions, [r["xtick"] for r in results], rotation=45, ha="right")

    # Add small ticks at the center of bars
    ax.tick_params(axis="x", which="major", length=4, width=2)

    # Customize the plot
    plt.ylabel("Hard Non-Ambiguous Qs Unfaithfulness (%)", fontsize=20, labelpad=10)
    plt.xlabel("Model", fontsize=24, labelpad=10)
    plt.ylim(0, 12)  # Increased upper limit slightly to fit labels
    plt.yticks([0, 5, 10])

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Show all spines (lines around the plot)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    fig.savefig(
        save_dir / "iphr_unfaithfulness_clear_only.pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close()


# %%

save_iphr_plot(df, DATA_DIR / ".." / ".." / "plots" / "all_models")
# %%
