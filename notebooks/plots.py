# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chainscope.typing import *
from chainscope.utils import MODELS_MAP

df = pd.read_pickle(DATA_DIR / "df.pkl")


# %%
def plot_comparison(df: pd.DataFrame, model_id: str, comparison: str, answer: str):
    condition_df = df[
        (df.model_id == model_id)
        & (df.comparison == comparison)
        & (df.answer == answer)
    ]

    # Difference plot
    condition_df_pivot = condition_df.pivot(
        index=["prop_id", "qid"], columns="mode", values="p_correct"
    )
    condition_df_pivot["difference"] = (
        condition_df_pivot["cot"] - condition_df_pivot["direct"]
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=condition_df_pivot.reset_index(),
        y="prop_id",
        x="difference",
        orient="h",
    )
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.title(
        f"Difference in P(Correct) (CoT - Direct) for {model_id}\n({comparison}_{answer})"
    )
    plt.xlabel("Difference in P(Correct)")
    plt.ylabel("Property")
    plt.tight_layout()
    plt.show()


# %%
def plot_all_models_differences(df: pd.DataFrame, model_keys: list[str]):
    # List to store median differences for each condition and model
    all_medians = []

    for model_key in model_keys:
        model_id = MODELS_MAP[model_key]
        for comparison in ["gt", "lt"]:
            for answer in ["NO", "YES"]:
                condition_df = df[
                    (df.model_id == model_id)
                    & (df.comparison == comparison)
                    & (df.answer == answer)
                ]

                try:
                    # Calculate differences for this condition
                    condition_df_pivot = condition_df.pivot(
                        index=["prop_id", "qid"], columns="mode", values="p_correct"
                    )
                    differences = (
                        condition_df_pivot["cot"] - condition_df_pivot["direct"]
                    )

                    # Get median for each prop_id
                    prop_medians = differences.groupby(level="prop_id").median()

                    # Store all medians with their condition
                    for median in prop_medians:
                        all_medians.append(
                            {
                                "median_diff": median,
                                "condition": f"{comparison}_{answer}",
                                "model_key": model_key,
                                "model_id": model_id,
                            }
                        )
                except KeyError:
                    continue

    # Create dataframe of all medians
    medians_df = pd.DataFrame(all_medians)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Distribution of Median Differences (CoT - Direct) by Model and Condition"
    )

    # Plot each condition in a separate subplot
    for idx, (comparison, answer) in enumerate(
        [(c, a) for c in ["gt", "lt"] for a in ["NO", "YES"]]
    ):
        ax = axes[idx // 2, idx % 2]
        condition = f"{comparison}_{answer}"
        condition_data = medians_df[medians_df.condition == condition]

        sns.boxplot(
            data=condition_data,
            y="model_key",
            x="median_diff",
            orient="h",
            ax=ax,
        )
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(condition)
        ax.set_xlabel("Median Difference in P(Correct)")
        ax.set_ylabel("Model")

    plt.tight_layout()
    plt.show()


# %%
# Plot differences for selected models
plot_all_models_differences(df, ["G2", "L1", "L3", "Q1.5", "Q3", "P"])

# %%
