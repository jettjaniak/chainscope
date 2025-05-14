# %%

import numpy as np
import pandas as pd

from chainscope.typing import *
from chainscope.utils import get_model_display_name

# %%

df = pd.read_pickle(DATA_DIR / "df-wm-non-ambiguous-hard-2.pkl")
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

# df = df[df.unknown_rate < 0.5]
# %%


def calculate_group_variance_ratio(data: pd.DataFrame) -> float:
    """Calculate the ratio of between-group to within-group variance."""
    # Calculate between-group variance
    group_means = data.groupby(["prop_id", "comparison"])["p_yes"].mean()
    overall_mean = data["p_yes"].mean()
    between_var = sum((group_means - overall_mean) ** 2) / (len(group_means) - 1)

    # Calculate within-group variance
    group_vars = data.groupby(["prop_id", "comparison"])["p_yes"].var()
    within_var = group_vars.mean()

    return between_var / within_var if within_var != 0 else float("inf")


def run_permutation_test(
    df: pd.DataFrame, n_permutations: int, random_seed: int | None = None
) -> tuple[float, float, float]:
    """
    Run a permutation test using variance ratio as the test statistic.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    observed_ratio = calculate_group_variance_ratio(df)

    permuted_ratios = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted_df = df.copy()
        permuted_df["p_yes"] = np.random.permutation(df["p_yes"].values)  # type: ignore
        permuted_ratios[i] = calculate_group_variance_ratio(permuted_df)

    p_value = float(np.sum(permuted_ratios >= observed_ratio) + 1) / (
        n_permutations + 1
    )
    std_err = np.sqrt(p_value * (1 - p_value) / (n_permutations + 1))

    return observed_ratio, p_value, std_err


# %%
N_PERMUTATIONS = 10_000
RANDOM_SEED = 42
# Run permutation test for each model separately
results = []
for model_id in df.model_id.unique():
    model_df = df[df.model_id == model_id]
    observed_ratio, p_value, std_err = run_permutation_test(
        model_df, N_PERMUTATIONS, RANDOM_SEED
    )
    results.append(
        {
            "model": get_model_display_name(model_id),
            "observed_ratio": observed_ratio,
            "p_value": p_value,
            "std_err": std_err,
        }
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("p_value")

# %%
results_df["pval_99ci_lower"] = results_df["p_value"] - results_df["std_err"] * 2.576
results_df["pval_99ci_upper"] = results_df["p_value"] + results_df["std_err"] * 2.576
print("Permutation Test Results:")
print(results_df.to_string(index=False))
# %%
