import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dataclass_wizard import YAMLWizard, fromdict

from chainscope import DATA_DIR


@dataclass
class Properties(YAMLWizard):
    gt_question: str
    lt_question: str
    value_by_name: dict[str, int | float]

    @classmethod
    def load(cls, prop_id: str) -> "Properties":
        properties = cls.from_yaml_file(DATA_DIR / "properties" / f"{prop_id}.yaml")
        assert isinstance(properties, cls)  # could've been a list
        return properties


@dataclass
class Instructions(YAMLWizard):
    cot: str
    direct: str

    @classmethod
    def load(cls, instr_id: str) -> "Instructions":
        with open(DATA_DIR / "instructions.yaml", "r") as f:
            instr_dict = yaml.safe_load(f)[instr_id]
        return fromdict(cls, instr_dict)


@dataclass
class Question(YAMLWizard):
    q_str: str
    x_name: str
    y_name: str
    x_value: int | float
    y_value: int | float


@dataclass
class QsDataset(YAMLWizard):
    question_by_qid: dict[str, Question]
    prop_id: str
    comparison: Literal["gt", "lt"]
    answer: Literal["YES", "NO"]
    max_comparisons: int

    def save(self) -> str:
        ds_uuid = uuid.uuid4().hex[:8]
        name = f"{self.prop_id}_{self.comparison}_{self.answer}_{self.max_comparisons}_{ds_uuid}.yaml"
        output_path = DATA_DIR / "questions" / name
        self.to_yaml_file(output_path)
        return name

    @classmethod
    def load(cls, name: str) -> "QsDataset":
        qsds = cls.from_yaml_file(DATA_DIR / "questions" / f"{name}.yaml")
        assert isinstance(qsds, cls)
        return qsds


@dataclass
class DirectEvalProbs(YAMLWizard):
    p_yes: float
    p_no: float


@dataclass
class DirectEval(YAMLWizard):
    probs_by_qid: dict[str, DirectEvalProbs]
    model_id: str
    instr_id: str

    def save(self, dataset_id: str) -> Path:
        directory = DATA_DIR / "direct_eval" / self.instr_id / dataset_id
        directory.mkdir(exist_ok=True, parents=True)
        model_id = self.model_id.replace("/", "__")
        path = directory / f"{model_id}.yaml"
        self.to_yaml_file(path)
        return path


@dataclass
class SamplingParams(YAMLWizard):
    temperature: float
    top_p: float
    max_new_tokens: int

    def get_identifier(self) -> str:
        return f"T{self.temperature}_P{self.top_p}_M{self.max_new_tokens}"


@dataclass
class CotResponses(YAMLWizard):
    responses_by_qid: dict[str, dict[str, str]]  # qid -> {uuid -> response_str}
    model_id: str
    instr_id: str
    sampling_params: SamplingParams

    def save(self, dataset_id: str) -> Path:
        sp_id = self.sampling_params.get_identifier()
        directory = DATA_DIR / "cot_responses" / self.instr_id / sp_id / dataset_id
        directory.mkdir(exist_ok=True, parents=True)
        model_id = self.model_id.replace("/", "__")
        path = directory / f"{model_id}.yaml"
        self.to_yaml_file(path)
        return path
