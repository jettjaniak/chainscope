#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import yaml

# Define all possible categories and their colors
CATEGORY_COLORS = {
    "answer flipping": "#ff7f0e",  # orange
    "fact manipulation": "#1f77b4",  # blue
    "different arguments": "#2ca02c",  # green
    "bad arguments": "#d62728",  # red
    "wrong comparison": "#9467bd",  # purple
    "missing step": "#8c564b",  # brown
    "other": "#7f7f7f",  # gray
}


def load_case_studies(file_path: Path) -> Dict:
    with open(file_path) as f:
        return yaml.safe_load(f)


def analyze_categories(data: Dict) -> tuple[list[str], list[float], list[str], int]:
    all_categories = [
        "answer flipping",
        "fact manipulation",
        "different arguments",
        "bad arguments",
        "wrong comparison",
        "missing step",
        "other",
    ]
    category_counts = Counter()
    for category in all_categories:
        category_counts[category] = 0

    total_cases = 0

    for case_data in data.values():
        if (
            "manual_analysis" not in case_data
            or "categories" not in case_data["manual_analysis"]
        ):
            continue

        categories = case_data["manual_analysis"]["categories"]

        total_cases += 1
        for category, is_present in categories.items():
            if is_present:
                category_counts[category] += 1

    # Convert to percentages and split into labels and values, maintaining consistent order
    labels = []
    percentages = []
    colors = []

    if total_cases > 0:
        for category in CATEGORY_COLORS.keys():
            if category_counts[category] > 0:
                labels.append(category)
                percentages.append((category_counts[category] / total_cases) * 100)
                colors.append(CATEGORY_COLORS[category])

    return labels, percentages, colors, total_cases


def plot_categories(
    labels: list[str],
    percentages: list[float],
    colors: list[str],
    model_name: str,
    output_dir: Path,
) -> None:
    if not percentages:  # If no categories found
        return

    plt.figure(figsize=(10, 8))
    plt.pie(percentages, labels=labels, colors=colors, autopct="%1.1f%%")
    plt.title(f"Category Distribution for {model_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{model_name}_categories.png")
    plt.close()


def main():
    case_studies_dir = Path("chainscope/data/case_studies/analyzed")
    output_dir = Path("plots/iphr_case_studies")

    total_analyzed_cases_per_model = {}
    print("model;category;percentage")

    for file_path in case_studies_dir.glob("*.yaml"):
        model_name = file_path.stem
        data = load_case_studies(file_path)
        labels, percentages, colors, total_cases = analyze_categories(data)
        plot_categories(labels, percentages, colors, model_name, output_dir)
        total_analyzed_cases_per_model[model_name] = total_cases
        # Print statistics for each category
        for label, percentage in zip(labels, percentages):
            print(f"{model_name};{label};{percentage:.1f}")

    print()
    print("model;total_cases")
    for model_name, total_cases in total_analyzed_cases_per_model.items():
        print(f"{model_name};{total_cases}")


if __name__ == "__main__":
    main()
