import uuid
from dataclasses import dataclass, field
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
class DatasetParams(YAMLWizard):
    prop_id: str
    comparison: Literal["gt", "lt"]
    answer: Literal["YES", "NO"]
    max_comparisons: int
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def pre_id(self) -> str:
        return f"{self.comparison}_{self.answer}_{self.max_comparisons}"

    @property
    def id(self) -> str:
        return f"{self.prop_id}_{self.pre_id}_{self.uuid}"

    @property
    def path(self) -> Path:
        return DATA_DIR / "questions" / self.pre_id / f"{self.id}.yaml"

    @classmethod
    def from_id(cls, dataset_id: str) -> "DatasetParams":
        prop_id, comparison, answer, max_comparisons, uuid = dataset_id.split("_")
        return cls(
            prop_id=prop_id,
            comparison=comparison,  # type: ignore
            answer=answer,  # type: ignore
            max_comparisons=int(max_comparisons),
            uuid=uuid,
        )

    def load_dataset(self) -> "QsDataset":
        qsds = QsDataset.from_yaml_file(self.path)
        assert isinstance(qsds, QsDataset)
        return qsds


@dataclass
class QsDataset(YAMLWizard):
    question_by_qid: dict[str, Question]
    params: DatasetParams

    @classmethod
    def load(cls, dataset_id: str) -> "QsDataset":
        params = DatasetParams.from_id(dataset_id)
        qs_dataset = params.load_dataset()
        assert qs_dataset.params == params
        return qs_dataset

    def save(self) -> Path:
        self.to_yaml_file(self.params.path)
        return self.params.path


@dataclass
class DirectEvalProbs(YAMLWizard):
    p_yes: float
    p_no: float


def get_path(directory: Path, model_id: str) -> Path:
    directory.mkdir(exist_ok=True, parents=True)
    model_id = model_id.replace("/", "__")
    path = directory / f"{model_id}.yaml"
    return path


@dataclass
class DirectEval(YAMLWizard):
    probs_by_qid: dict[str, DirectEvalProbs]
    ds_params: DatasetParams
    model_id: str
    instr_id: str

    def save(self) -> Path:
        directory = (
            DATA_DIR
            / "direct_eval"
            / self.instr_id
            / self.ds_params.pre_id
            / self.ds_params.id
        )
        path = get_path(directory, self.model_id)
        self.to_yaml_file(path)
        return path


@dataclass
class SamplingParams(YAMLWizard):
    temperature: float
    top_p: float
    max_new_tokens: int

    @property
    def id(self) -> str:
        return f"T{self.temperature}_P{self.top_p}_M{self.max_new_tokens}"


@dataclass
class CotResponses(YAMLWizard):
    responses_by_qid: dict[str, dict[str, str]]  # qid -> {uuid -> response_str}
    model_id: str
    instr_id: str
    ds_params: DatasetParams
    sampling_params: SamplingParams

    def save(self) -> Path:
        directory = (
            DATA_DIR
            / "cot_responses"
            / self.instr_id
            / self.sampling_params.id
            / self.ds_params.pre_id
            / self.ds_params.id
        )
        path = get_path(directory, self.model_id)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "CotResponses":
        cot_responses = cls.from_yaml_file(path)
        assert isinstance(cot_responses, cls)
        return cot_responses


@dataclass
class CotEval(YAMLWizard):
    results_by_qid: dict[
        str, dict[str, Literal["YES", "NO", "UNKNOWN"]]
    ]  # qid -> {response_uuid -> result}
    model_id: str
    instr_id: str
    ds_params: DatasetParams
    sampling_params: SamplingParams

    def save(self) -> Path:
        directory = (
            DATA_DIR
            / "cot_eval"
            / self.instr_id
            / self.sampling_params.id
            / self.ds_params.pre_id
            / self.ds_params.id
        )
        path = get_path(directory, self.model_id)
        self.to_yaml_file(path)
        return path
