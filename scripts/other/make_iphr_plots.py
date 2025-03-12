#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from chainscope.typing import *
from chainscope.utils import get_model_display_name, sort_models


def create_dual_diverging_barplots(
    df: pd.DataFrame,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create dual diverging barplots showing frequency of YES responses for greater/less than comparisons."""
    # Aggregate the data with both mean and standard error
    plot_data = (
        df.groupby(["prop_id", "comparison"])
        .agg({"p_yes": ["mean", "count", "std"]})
        .reset_index()
    )

    # Flatten column names
    plot_data.columns = ["prop_id", "comparison", "mean", "count", "std"]
    # Calculate standard error
    plot_data["stderr"] = plot_data["std"] / np.sqrt(plot_data["count"])

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, dpi=300)
    model_name = df.model_id.unique()[0].split("/")[-1]
    fig.suptitle(model_name)
    prop_ids = sorted(df.prop_id.unique().tolist())

    # Process each subplot
    for ax, comp, title in zip([ax1, ax2], ["gt", "lt"], ["Greater Than", "Less Than"]):
        # Filter data
        ax.set_title(title)
        comp_data = plot_data[plot_data["comparison"] == comp].copy()
        for prop_id in prop_ids:
            prop_data = comp_data[comp_data["prop_id"] == prop_id]
            if len(prop_data) != 1:
                continue
            # Center values around 0.5
            centered_value = prop_data["mean"].iloc[0] - 0.5

            # Create horizontal bars
            y_pos = prop_ids.index(prop_id)
            color = "red" if centered_value < 0 else "green"

            # Plot bars with error bars
            ax.barh(y_pos, centered_value, align="center", color=color, zorder=3)
            ax.errorbar(
                centered_value,
                y_pos,
                xerr=prop_data["stderr"].iloc[0],
                fmt="none",
                color="black",
                zorder=4,
            )

    for ax in [ax1, ax2]:
        ax.set_yticks(list(range(len(prop_ids))))
        short_prop_ids = [
            x.replace("wm-", "").replace("population", "popu") for x in prop_ids
        ]
        ax.set_yticklabels(short_prop_ids, fontsize=8)
        # Add light grey dashed horizontal grid lines
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="grey", zorder=1)
        # Rest of the customization remains the same
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3, zorder=2)
        x_ticks = np.linspace(-0.5, 0.5, 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x + 0.5:.1f}" for x in x_ticks])
        ax.set_xlabel("freq. of YES")

    plt.tight_layout()
    return fig, (ax1, ax2)


def create_bias_strength_plot(df: pd.DataFrame) -> Figure:
    """Create a plot showing bias strength percentage for each model."""
    filter_prop_ids = ["animals-speed", "sea-depths", "sound-speeds", "train-speeds"]
    df = df[~df.prop_id.isin(filter_prop_ids)]
    df = df[df["mode"] == "cot"]

    # First calculate mean p_yes for each group
    plot_data = (
        df.groupby(["model_id", "prop_id", "comparison"])["p_yes"].mean().reset_index()
    )
    plot_data["model_vendor"] = plot_data["model_id"].map(lambda x: x.split("/")[0])
    plot_data["model_display"] = plot_data["model_id"].map(lambda x: x.split("/")[-1])
    plot_data = plot_data[plot_data.model_id != "Qwen/Qwen2.5-0.5B-Instruct"]
    xlabel = "bias strength (%)"
    # Then calculate the absolute deviation from 0.5 for these means
    plot_data[xlabel] = abs(0.5 - plot_data["p_yes"]) * 100

    # Create the plot
    fig = plt.figure(dpi=300)
    model_ids = list(plot_data["model_id"].unique())
    sns.barplot(
        plot_data,
        y="model_display",  # Use display name for y-axis
        x=xlabel,
        order=[
            get_model_display_name(m) for m in sort_models(model_ids)
        ],  # Sort using full names but display short names
        hue="model_vendor",
        errorbar="se",
    )
    plt.legend().set_visible(False)
    return fig


def main() -> None:
    """Main function to create and save plots."""
    # Load data
    df = pd.read_pickle(DATA_DIR / "df-wm.pkl")

    # Create plots directory
    plots_dir = Path("plots/iphr")
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create and save bias strength plot
    bias_fig = create_bias_strength_plot(df)
    bias_fig.savefig(plots_dir / "bias_strength.png", bbox_inches="tight")
    plt.close(bias_fig)

    per_cat_dir = plots_dir / "per_cat"
    per_cat_dir.mkdir(exist_ok=True, parents=True)

    # Create and save dual diverging barplots for each model
    df = df[df["mode"] == "cot"]
    for model in df.model_id.unique():
        model_data = df[df.model_id == model]
        if len(model_data) == 0:
            print(f"No data for {model}")
            continue
        model_str = model.split("/")[-1]
        fig, _ = create_dual_diverging_barplots(model_data)
        fig.savefig(per_cat_dir / f"{model_str}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
