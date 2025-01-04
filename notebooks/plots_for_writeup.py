# %%

import matplotlib.pyplot as plt
import pandas as pd

from chainscope.typing import *
from chainscope.utils import MODELS_MAP

df = pd.read_pickle(DATA_DIR / "df.pkl")
filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
df = df[~df.prop_id.isin(filter_prop_ids)]
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate


# %%
# Create a figure showing bias patterns across models
def plot_model_biases(
    df: pd.DataFrame, biases: Literal["direct", "cot", "both"] = "both"
):
    # Bottom to top
    model_order = [
        "P",  # Phi
        "Q72",
        "Q32",
        "Q14",
        "Q7",
        "Q3",
        "Q1.5",
        "Q0.5",  # Qwens
        "G27",
        "G9",
        "G2",  # Gemmas
        "L70",
        "L8",
        "L3",
        "L1",  # Llamas
    ]

    model_labels = [MODELS_MAP[model_key].split("/")[-1] for model_key in model_order]

    results = []
    for model_key in model_order:
        model_id = MODELS_MAP[model_key]
        for mode in ["direct", "cot"]:
            model_data = df[(df["model_id"] == model_id) & (df["mode"] == mode)]

            yes_acc = model_data[model_data["answer"] == "YES"].p_correct.mean()
            no_acc = model_data[model_data["answer"] == "NO"].p_correct.mean()

            results.append(
                {"model": model_key, "mode": mode, "yes_acc": yes_acc, "no_acc": no_acc}
            )

    results_df = pd.DataFrame(results)

    # Create the plot
    plt.figure(figsize=(12, 8))
    y_positions = range(len(model_order))

    for idx, model in enumerate(model_order):
        model_data = results_df[results_df["model"] == model]

        if biases == "cot" or biases == "both":
            cot = model_data[model_data["mode"] == "cot"].iloc[0]
            cot_acc_diff = cot.no_acc - cot.yes_acc

            if abs(cot_acc_diff) < 0.05:
                bias_str = "W/O COT BIAS"
            elif cot_acc_diff > 0:
                bias_str = "COT BIAS to NO"
            else:
                bias_str = "COT BIAS to YES"
            print(
                f"Model: {MODELS_MAP[model].split("/")[-1]}, CoT: yes_acc: {cot.yes_acc:.2f}, no_acc: {cot.no_acc:.2f}, diff: {cot_acc_diff:.2f} -> {bias_str}"
            )

            plt.arrow(
                cot.yes_acc,
                idx + (0.1 if biases == "both" else 0),
                cot_acc_diff,
                0,
                head_width=0.12,
                head_length=0.01,
                linewidth=2,
                color="#9b59b6",
                length_includes_head=True,
                label="CoT" if idx == 0 else None,
            )

        if biases == "direct" or biases == "both":
            direct = model_data[model_data["mode"] == "direct"].iloc[0]
            direct_acc_diff = direct.no_acc - direct.yes_acc

            if abs(direct_acc_diff) < 0.05:
                bias_str = "W/O DIRECT BIAS"
            elif direct_acc_diff > 0:
                bias_str = "DIRECT BIAS to NO"
            else:
                bias_str = "DIRECT BIAS to YES"
            print(
                f"Model: {MODELS_MAP[model].split("/")[-1]}, Direct: yes_acc: {direct.yes_acc:.2f}, no_acc: {direct.no_acc:.2f}, diff: {direct_acc_diff:.2f} -> {bias_str}"
            )

            plt.arrow(
                direct.yes_acc,
                idx - (0.1 if biases == "both" else 0),
                direct_acc_diff,
                0,
                head_width=0.12,
                head_length=0.01,
                linewidth=2,
                color="#2ecc71",
                length_includes_head=True,
                label="Direct" if idx == 0 else None,
            )

    plt.yticks(y_positions, model_labels)
    plt.xlabel("Accuracy")
    plt.title("Model Biases: Average accuracy from YES to NO questions")
    plt.grid(True, alpha=0.3)
    if biases == "both":
        plt.legend()

    # Add vertical line at 0.5 for reference
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# Call the function
plot_model_biases(df, biases="cot")
plot_model_biases(df, biases="direct")
plot_model_biases(df, biases="both")
