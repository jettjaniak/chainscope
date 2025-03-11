#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Any, Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.scale import FuncScale
from matplotlib.ticker import FuncFormatter
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from chainscope import DATA_DIR
from chainscope.bias_probing import ProbeTrainer

LOC_ORDER = [
    "colon",
    "colon+1",
    "qmark",
    "qmark+1",
    "reasoning",
    "reasoning+1",
    "yes",
    "no",
    "answer",
    "last_prompt",
    "turn",
]


def get_cache_path(model_name: str, loc: str, layer: int) -> Path:
    """Get the path for caching wandb results."""
    cache_dir = Path(f"plots/bias_probes/{model_name}/{loc}/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"L{layer}.pkl"


def get_all_layers_cache_path(model_name: str, loc: str) -> Path:
    """Get the path for caching all layers results for a location."""
    cache_dir = Path(f"plots/bias_probes/{model_name}/{loc}/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "all_layers.pkl"


def load_all_layers_cache(cache_path: Path) -> dict[int, tuple[float, dict[str, dict]]]:
    """Load cached results for all layers if they exist."""
    if not cache_path.exists():
        return {}
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def save_all_layers_cache(
    cache_path: Path, results: dict[int, tuple[float, dict[str, dict]]]
) -> None:
    """Save results for all layers to cache."""
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)


def load_cached_results(cache_path: Path) -> tuple[float, dict[str, dict]] | None:
    """Load cached results if they exist."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def save_results_to_cache(
    cache_path: Path, fvu: float, results: dict[str, dict]
) -> None:
    """Save results to cache."""
    with open(cache_path, "wb") as f:
        pickle.dump((fvu, results), f)


def compute_fvu(results: dict[str, dict]) -> float:
    """Compute Fraction of Variance Unexplained."""
    true_values = np.array([x["template_bias"] for x in results.values()])
    predictions = np.array([x["mean_pred"] for x in results.values()])
    mse = np.mean((true_values - predictions) ** 2)
    var = np.var(true_values)
    return float(mse / var) if var != 0 else float("nan")


def compute_fvu_and_get_results(runs: list[Any]) -> tuple[float, dict[str, dict]]:
    """Compute FVU using test losses and collect results using wandb summary predictions."""
    # Get template biases from dataframe
    data_config = runs[0].config["data_config"]
    df = pd.read_pickle(DATA_DIR / "df-wm.pkl")
    model_ids = [
        mid for mid in df.model_id.unique() if mid.endswith(data_config["model_name"])
    ]
    assert len(model_ids) == 1
    model_id = model_ids[0]
    df = df[df.model_id == model_id]

    results = {}
    all_losses = []
    all_template_biases = []

    for run in runs:
        for key in run.summary.keys():
            if key.startswith("test_loss/"):
                template = key.replace("test_loss/", "")
                prop_id, comparison = template.split("_")
                template_bias = float(
                    df[(df["prop_id"] == prop_id) & (df["comparison"] == comparison)][
                        "p_yes"
                    ].mean()
                )

                # Get mean and std predictions from wandb metrics
                mean_pred = run.summary[f"test_prediction_mean/{template}"]
                std_pred = run.summary[f"test_prediction_std/{template}"]
                loss = run.summary[key]

                # Store for FVU computation
                all_losses.append(loss)
                all_template_biases.append(template_bias)

                results[template] = {
                    "mean_pred": mean_pred,
                    "std_pred": std_pred,
                    "template_bias": template_bias,
                }

    # Compute FVU using mean squared error from test losses
    mse = float(np.mean(all_losses))
    var = float(np.var(all_template_biases))
    fvu = mse / var if var != 0 else float("nan")
    # print(f"{len(all_losses)=}")
    # print(f"{len(all_template_biases)=}")
    # print(f"{mse=}")
    # print(f"{var=}")
    return fvu, results


def create_comparison_plot(
    results: dict[str, dict], model_name: str, layer: int, loc: str, fvu: float
) -> None:
    # Extract prop_ids and organize data
    templates = list(results.keys())
    prop_ids = sorted(list({t.split("_")[0] for t in templates}))

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6), sharey=True, dpi=300)

    # Process each subplot
    for ax, comp, title in zip([ax1, ax2], ["gt", "lt"], ["Greater Than", "Less Than"]):
        ax.set_title(title)

        for i, prop_id in enumerate(prop_ids):
            template = f"{prop_id}_{comp}"
            if template not in results:
                continue

            result = results[template]
            mean_pred = result["mean_pred"]
            std_pred = result["std_pred"]
            template_bias = result["template_bias"]

            # Plot ground truth as a triangle marker
            ax.scatter(
                template_bias,
                i,
                marker="^",
                color="black",
                s=50,
                zorder=4,
                label="Ground Truth" if i == 0 else None,
            )

            # Plot prediction as a dot with error bars
            ax.errorbar(
                mean_pred,
                i,
                xerr=std_pred,
                fmt="o",
                color="blue",
                zorder=5,
                label="Prediction" if i == 0 else None,
            )

        # Customize the plot
        ax.set_yticks(list(range(len(prop_ids))))
        short_prop_ids = [
            x.replace("wm-", "").replace("population", "popu") for x in prop_ids
        ]
        ax.set_yticklabels(short_prop_ids, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="grey", zorder=1)
        ax.axvline(x=0.5, color="grey", linestyle="--", alpha=0.3, zorder=1)
        ax.set_xlabel("freq. of YES")
        ax.set_xlim(0, 1)

    # Add FVU to title
    loc_str = "end-of-turn token" if loc == "turn" else f"{loc=}"
    fig.suptitle(
        f"Probe predictions for layer {layer} {loc_str}\n{model_name}, FVU={fvu:.2%}"
    )

    ax1.legend(loc="upper left")

    plt.tight_layout()
    fig_dir = Path(f"plots/bias_probes/{model_name}/{loc}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"L{layer}.png")
    plt.close()


def get_custom_scale_transform(
    max_value: float,
    min_value: float,
    z: float = 0.8,
) -> tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]:
    """Create custom scale transformation for y-axis.
    min_value-100% takes up 80% of the axis height, >100% takes up remaining 20%."""

    def forward(y: ArrayLike) -> ArrayLike:
        y_arr = np.asarray(y)
        # Scale [min_value, 1] to [0, z], and [1, max_value] to [z, 1]
        return np.where(
            y_arr <= 1,
            z * (y_arr - min_value) / (1 - min_value),  # min_value-1 maps to [0, z]
            z + (1 - z) * (y_arr - 1) / (max_value - 1),  # >1 maps to [z, 1]
        )

    def inverse(y: ArrayLike) -> ArrayLike:
        y_arr = np.asarray(y)
        return np.where(
            y_arr <= z,
            # [0, z] maps to [min_value, 1]
            min_value + y_arr * (1 - min_value) / z,
            # [z, 1] maps to [1, max_value]
            1 + (y_arr - z) * (max_value - 1) / (1 - z),
        )

    return forward, inverse


def create_fvu_line_plot(
    fvu_by_layer: dict[int, float], model_name: str, loc: str
) -> None:
    """Create a line plot showing FVU across layers."""
    layers = sorted(fvu_by_layer.keys())
    fvus = [fvu_by_layer[layer] for layer in layers]
    max_fvu = max(fvus)
    min_fvu = min(fvus)
    min_tick = (min_fvu // 0.1) * 0.1  # Round down to nearest 10%

    plt.figure(figsize=(10, 6), dpi=300)

    # Only use custom scale if min_fvu < 1.0 (100%)
    if min_fvu < 1.0:
        transform = get_custom_scale_transform(max_fvu, min_tick)
        plt.gca().set_yscale(FuncScale(plt.gca().yaxis, transform))
        below_100_ticks = np.arange(min_tick, 1.0, 0.05)  # Up to 100%, every 5%
        above_100_ticks = np.arange(1, max_fvu + 5, 5)  # Every 500% above 100%
        yticks = np.unique(np.concatenate([below_100_ticks, above_100_ticks]))
        plt.yticks(yticks)

    # Plot without markers
    plt.plot(layers, fvus)
    plt.grid(True, alpha=0.3)

    # Set x-ticks for every layer but only label even ones
    plt.xticks(layers, [str(l) if l % 2 == 0 else "" for l in layers])

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    plt.xlabel("Layer")
    plt.ylabel("Fraction of Variance Unexplained (FVU)")
    plt.title(f"{model_name} {loc=} FVU by Layer")

    # Save plot
    fig_dir = Path(f"plots/bias_probes/{model_name}/{loc}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fvu_by_layer.png")
    plt.close()


def create_combined_fvu_line_plot(
    fvu_by_loc_layer: dict[str, dict[int, float]], model_name: str
) -> None:
    """Create a line plot showing FVU across layers for multiple locations."""
    plt.figure(figsize=(9, 7), dpi=300)

    # Find global min and max FVU across all locations
    all_fvus = [
        fvu
        for fvu_by_layer in fvu_by_loc_layer.values()
        for fvu in fvu_by_layer.values()
    ]
    max_fvu = max(all_fvus)
    min_fvu = min(all_fvus)
    min_tick = (min_fvu // 0.1) * 0.1  # Round down to nearest 10%

    # Only use custom scale if min_fvu < 1.0 (100%)
    if min_fvu < 1.0:
        transform = get_custom_scale_transform(max_fvu, min_tick, z=0.7)
        plt.gca().set_yscale(FuncScale(plt.gca().yaxis, transform))
        below_100_ticks = np.arange(min_tick, 1.0, 0.05)  # Up to 100%, every 5%
        above_100_ticks = np.arange(1, max_fvu + 5, 5)  # Every 500% above 100%
        yticks = np.unique(np.concatenate([below_100_ticks, above_100_ticks]))
        plt.yticks(yticks)

    # Get colormap
    n_colors = len(LOC_ORDER)
    colors = colormaps["nipy_spectral"](np.linspace(0, 0.9, n_colors))
    # Create a mapping from location to color
    color_by_loc = dict(zip(LOC_ORDER, colors))

    # Plot each location in the specified order if it exists in the data
    for i, loc in enumerate(LOC_ORDER):
        if loc in fvu_by_loc_layer:
            fvu_by_layer = fvu_by_loc_layer[loc]
            layers = sorted(fvu_by_layer.keys())
            fvus = [fvu_by_layer[layer] for layer in layers]
            linestyle = "--" if i % 2 else "-"  # Alternate between solid and dashed
            plt.plot(
                layers, fvus, label=loc, color=color_by_loc[loc], linestyle=linestyle
            )

    plt.grid(True, alpha=0.3)

    # Set x-ticks for every layer using the first location's layers
    first_loc_layers = sorted(next(iter(fvu_by_loc_layer.values())).keys())
    plt.xticks(
        first_loc_layers, [str(l) if l % 2 == 0 else "" for l in first_loc_layers]
    )

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    plt.xlabel("Layer")
    plt.ylabel("Fraction of Variance Unexplained (FVU)\n(lower is better)")
    plt.title(f"{model_name} (80 layers)\nFVU by Layer and Location for seed 0")
    # plt.legend(title="Location", bbox_to_anchor=(1.01, 1))

    plt.tight_layout()

    # Save plot
    fig_dir = Path(f"plots/bias_probes/{model_name}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fvu_by_layer_combined.png")
    plt.close()


def get_seed_comparison_cache_path(
    model_name: str, loc: str, layers: list[int], seeds: list[int]
) -> Path:
    """Get the path for caching seed comparison data."""
    cache_dir = Path(f"plots/bias_probes/{model_name}/{loc}/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    layers_str = f"L{min(layers)}-L{max(layers)}"
    seeds_str = f"S{min(seeds)}-S{max(seeds)}"
    return cache_dir / f"seed_comparison_{layers_str}_{seeds_str}.pkl"


def load_seed_comparison_cache(
    cache_path: Path,
) -> tuple[dict[int, list[float]], dict[int, dict[int, float | None]]] | None:
    """Load cached seed comparison data if it exists."""
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def save_seed_comparison_cache(
    cache_path: Path,
    data_by_layer: dict[int, list[float]],
    fvu_by_seed_layer: dict[int, dict[int, float | None]],
) -> None:
    """Save seed comparison data to cache."""
    with open(cache_path, "wb") as f:
        pickle.dump((data_by_layer, fvu_by_seed_layer), f)


def create_seed_comparison_boxplot(
    model_name: str,
    wandb_project: str,
    wandb_entity: str,
    layers: list[int] = list(range(50, 56)),  # Layers 50-55
    seeds: list[int] = list(range(5)),  # Seeds 0-4
    loc: str = "turn",
    force_refetch: bool = False,
) -> None:
    """Create a boxplot comparing FVU across different seeds for specific layers."""
    # Get cache path for seed comparison data
    cache_path = get_seed_comparison_cache_path(model_name, loc, layers, seeds)

    # Try to load cached data
    cached_data = None if force_refetch else load_seed_comparison_cache(cache_path)

    if cached_data is not None:
        print(f"Using cached seed comparison data from {cache_path}")
        data_by_layer, fvu_by_seed_layer = cached_data
    else:
        print(
            f"Fetching seed comparison data for {model_name}, {loc=}, layers={layers}, seeds={seeds}"
        )
        # Prepare data structure for boxplot
        data_by_layer = {layer: [] for layer in layers}
        fvu_by_seed_layer: dict[int, dict[int, float | None]] = {
            seed: {layer: None for layer in layers} for seed in seeds
        }

        # Collect FVU values for each layer and seed
        for seed in tqdm(seeds, desc="Seeds"):
            for layer in layers:
                try:
                    # Fetch runs for this seed and layer
                    runs = ProbeTrainer.wandb_runs(
                        entity=wandb_entity,
                        project=wandb_project,
                        config_filters={
                            "data_config.model_name": model_name,
                            "data_config.loc": loc,
                            "data_config.layer": layer,
                            "data_config.train_val_seed": seed,
                        },
                        n_runs=37,
                    )

                    # Calculate FVU for this seed and layer
                    fvu, _ = compute_fvu_and_get_results(runs)
                    fvu_by_seed_layer[seed][layer] = fvu
                    data_by_layer[layer].append(fvu)

                except (ValueError, AssertionError) as e:
                    print(f"Error processing layer {layer}, seed {seed}: {e}")
                    continue

        # Save data to cache
        save_seed_comparison_cache(cache_path, data_by_layer, fvu_by_seed_layer)

    # Set up the figure
    plt.figure(figsize=(8, 5), dpi=300)

    # Prepare data for boxplot
    box_data = [data_by_layer[layer] for layer in layers]

    # Create boxplot
    bp = plt.boxplot(
        box_data,
        patch_artist=True,
        whis=(0.0, 100.0),  # Make whiskers extend to min and max
        showfliers=False,  # Don't show outliers as separate points
    )

    # Set x-axis labels after creating the boxplot
    plt.xticks(range(1, len(layers) + 1), [str(layer) for layer in layers])

    # Customize boxplot colors
    for box in bp["boxes"]:
        box.set(facecolor="lightblue", alpha=0.7)

    # Add individual points for each seed with different colors
    colormap = colormaps["tab10"]
    colors = colormap(np.linspace(0, 1, len(seeds)))
    for i, layer in enumerate(layers):
        for j, seed in enumerate(seeds):
            fvu_value = fvu_by_seed_layer[seed][layer]
            if fvu_value is not None:
                plt.scatter(
                    i + 1,  # Boxplot positions are 1-indexed
                    fvu_value,
                    color=colors[j],
                    marker="o",
                    s=50,
                    zorder=20 - seed,
                    label=str(seed) if i == 0 else None,  # Only add to legend once
                )

    # Add a legend
    plt.legend(title="Seeds")

    # Customize the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel("Layer")
    plt.ylabel("Fraction of Variance Unexplained (FVU)\n(lower is better)")
    plt.title(
        f"{model_name}\nFVU by seed for end-of-turn token, layers {min(layers)}-{max(layers)}"
    )

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Save plot
    fig_dir = Path(f"plots/bias_probes/{model_name}/{loc}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"seed_comparison_L{min(layers)}-L{max(layers)}.png")
    plt.close()


@click.command()
@click.option("-m", "--model-name", type=str, required=True)
@click.option(
    "--locs",
    type=str,
    default=",".join(LOC_ORDER),
    help="Comma-separated list of locations",
)
@click.option("--layers", type=str, default=None, help="Comma-separated list of layers")
@click.option("--wandb-project", type=str, default="bias-probes-valweight75")
@click.option("--wandb-entity", type=str, default="cot-probing")
@click.option("--seed-comparison", is_flag=True, help="Create seed comparison boxplot")
@click.option(
    "--seed-comparison-loc",
    type=str,
    default="turn",
    help="Location to use for seed comparison boxplot",
)
@click.option(
    "--seed-comparison-layers",
    type=str,
    default="50:56",
    help="Layers to include in seed comparison boxplot (range format like 50:56)",
)
@click.option(
    "--seed-comparison-seeds",
    type=str,
    default="0,1,2,3,4",
    help="Seeds to include in seed comparison boxplot (comma-separated list)",
)
def main(
    model_name: str,
    locs: str,
    layers: str | None,
    wandb_project: str,
    wandb_entity: str,
    seed_comparison: bool,
    seed_comparison_loc: str,
    seed_comparison_layers: str,
    seed_comparison_seeds: str,
):
    if not seed_comparison:
        locs_list = locs.split(",")
        layers_list = []
        if layers is not None:
            for layer in layers.split(","):
                if ":" in layer:
                    start, *end_and_step = layer.split(":")
                    if len(end_and_step) == 2:
                        end, step = end_and_step
                    else:
                        end = end_and_step[0]
                        step = 1
                    start = int(start)
                    end = int(end)
                    step = int(step)
                    layers_list.extend(range(start, end, step))
                else:
                    layers_list.append(int(layer))
            print(f"L={','.join(map(str, layers_list))}")
        fvu_by_layer_by_loc: dict[str, dict[int, float]] = {}

        for loc in locs_list:
            fvu_by_layer = {}

            # Try to load from all-layers cache first
            all_layers_cache_path = get_all_layers_cache_path(model_name, loc)
            all_layers_cache = load_all_layers_cache(all_layers_cache_path)

            if layers is None:
                # Use all layers from cache
                layers_list = sorted(all_layers_cache.keys())
                layers_to_fetch = []
            else:
                # Identify which layers need to be fetched
                layers_to_fetch = [l for l in layers_list if l not in all_layers_cache]

            # Fetch missing layers
            if layers_to_fetch:
                for current_layer in tqdm(layers_to_fetch, desc=f"{loc:<11}"):
                    cache_path = get_cache_path(model_name, loc, current_layer)
                    cached_results = load_cached_results(cache_path)

                    if cached_results is not None:
                        fvu, results = cached_results
                    else:
                        try:
                            runs = ProbeTrainer.wandb_runs(
                                entity=wandb_entity,
                                project=wandb_project,
                                config_filters={
                                    "data_config.model_name": model_name,
                                    "data_config.loc": loc,
                                    "data_config.layer": current_layer,
                                    "data_config.train_val_seed": 0,
                                },
                                n_runs=37,
                            )
                        except ValueError:
                            continue
                        assert len(runs) == 37

                        fvu, results = compute_fvu_and_get_results(runs)
                        save_results_to_cache(cache_path, fvu, results)
                    all_layers_cache[current_layer] = (fvu, results)
                    create_comparison_plot(results, model_name, current_layer, loc, fvu)

            if loc == "turn":
                fvu, results = all_layers_cache[55]
                create_comparison_plot(results, model_name, 55, loc, fvu)

            # Save updated all-layers cache
            save_all_layers_cache(all_layers_cache_path, all_layers_cache)

            for current_layer in layers_list:
                if current_layer not in all_layers_cache:
                    continue
                fvu, _ = all_layers_cache[current_layer]
                fvu_by_layer[current_layer] = fvu

            if fvu_by_layer:
                fvu_by_layer_by_loc[loc] = fvu_by_layer
                create_fvu_line_plot(fvu_by_layer, model_name, loc)

        create_combined_fvu_line_plot(fvu_by_layer_by_loc, model_name)

    # Create seed comparison boxplot if requested
    if seed_comparison:
        # Parse seed comparison layers
        seed_layers = []
        if ":" in seed_comparison_layers:
            start, end = seed_comparison_layers.split(":")
            seed_layers = list(range(int(start), int(end)))
        else:
            seed_layers = [int(layer) for layer in seed_comparison_layers.split(",")]

        # Parse seeds
        seeds = [int(seed) for seed in seed_comparison_seeds.split(",")]

        print(
            f"Creating seed comparison boxplot for {seed_comparison_loc} with layers {seed_layers} and seeds {seeds}"
        )
        create_seed_comparison_boxplot(
            model_name,
            wandb_project,
            wandb_entity,
            layers=seed_layers,
            seeds=seeds,
            loc=seed_comparison_loc,
        )


if __name__ == "__main__":
    main()
