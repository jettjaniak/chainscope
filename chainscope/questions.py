import hashlib
import logging
from typing import Literal

from chainscope.typing import DatasetParams, Properties, QsDataset, Question


def gen_qs(
    answer: Literal["YES", "NO"],
    comparison: Literal["gt", "lt"],
    max_comparisons: int,
    prop_id: str,
) -> QsDataset:
    properties = Properties.load(prop_id)
    sorted_values = sorted(properties.value_by_name.items(), key=lambda x: x[1])
    if (answer == "YES" and comparison == "gt") or (
        answer == "NO" and comparison == "lt"
    ):
        sorted_values = sorted_values[::-1]

    logging.info(f"Sorted values: {sorted_values}")

    question_by_qid = {}
    for x_idx, (x_name, x_value) in enumerate(sorted_values):
        picked_count = 0
        for y_name, y_value in sorted_values[x_idx + 1 :]:
            if picked_count >= max_comparisons:
                break

            if x_value == y_value:
                logging.info(f"Skipping {x_name} and {y_name} because values are equal")
                continue

            if comparison == "gt":
                question_template = properties.gt_question
                open_ended_question_template = properties.gt_open_ended_question
            else:
                question_template = properties.lt_question
                open_ended_question_template = properties.lt_open_ended_question
            q_str = question_template.format(x=x_name, y=y_name)
            q_str_open_ended = open_ended_question_template.format(x=x_name, y=y_name)

            qid = hashlib.sha256(q_str.encode()).hexdigest()
            question_by_qid[qid] = Question(
                q_str=q_str,
                q_str_open_ended=q_str_open_ended,
                x_name=x_name,
                y_name=y_name,
                x_value=x_value,
                y_value=y_value,
            )
            picked_count += 1

    return QsDataset(
        question_by_qid=question_by_qid,
        params=DatasetParams(
            prop_id=prop_id,
            comparison=comparison,
            answer=answer,
            max_comparisons=max_comparisons,
        ),
    )
