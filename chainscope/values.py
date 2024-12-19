from dataclasses import dataclass

import yaml

from chainscope import DATA_DIR


@dataclass
class CategoryValues:
    gt_template: str
    lt_template: str
    values: dict[str, int | float]


def load_values() -> dict[str, CategoryValues]:
    result = {}
    values_dir = DATA_DIR / "values"
    values_files = values_dir.glob("*.yaml")
    for values_file in values_files:
        values = yaml.safe_load(values_file.read_text())
        result[values_file.stem] = CategoryValues(
            gt_template=values["gt_template"],
            lt_template=values["lt_template"],
            values=values["values"],
        )
    return result
