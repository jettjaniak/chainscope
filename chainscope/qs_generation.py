import logging
from dataclasses import dataclass
from typing import Literal

from chainscope.values import load_values


@dataclass
class Question:
    q_str: str
    expected_answer: Literal["yes", "no"]
    category: str
    x: str
    y: str
    x_value: int | float
    y_value: int | float


def gen_qs(
    expected_answer: Literal["yes", "no"],
    template: Literal["gt", "lt"],
    pick_count: int,
    min_value_diff: int,
    max_value_diff: int,
    verbose: bool,
) -> list[Question]:
    qs = []
    values = load_values()
    for category, category_values in values.items():
        if verbose:
            logging.info(f"Processing values for {category}")

        # Sort values by value
        if expected_answer == "yes":
            sorted_values = sorted(
                category_values.values.items(), key=lambda x: x[1], reverse=True
            )
        else:
            sorted_values = sorted(category_values.values.items(), key=lambda x: x[1])

        if verbose:
            logging.info(f"Sorted values: {sorted_values}")

        for x_idx, (x, x_value) in enumerate(sorted_values):
            picked_count = 0
            for y, y_value in sorted_values[x_idx + 1 :]:
                if picked_count >= pick_count:
                    break

                diff = abs(x_value - y_value)
                if diff < min_value_diff or diff > max_value_diff:
                    if verbose:
                        logging.info(f"Skipping {x} and {y} because diff is {diff}")
                    continue

                if template == "gt":
                    q_str = category_values.gt_template.format(x=x, y=y)
                else:
                    q_str = category_values.lt_template.format(x=x, y=y)

                qs.append(
                    Question(
                        q_str=q_str,
                        expected_answer=expected_answer,
                        category=category,
                        x=x,
                        y=y,
                        x_value=x_value,
                        y_value=y_value,
                    )
                )
                picked_count += 1

    return qs