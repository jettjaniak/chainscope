
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from dataclass_wizard import DumpMeta, LoadMeta, YAMLWizard, fromdict

from chainscope import DATA_DIR


@dataclass
class Properties(YAMLWizard):
    gt_question: str
    lt_question: str
    gt_open_ended_question: str
    lt_open_ended_question: str
    value_by_name: dict[str, int | float]

    @classmethod
    def load(cls, prop_id: str) -> "Properties":
        if prop_id.startswith("wm-"):
            path = DATA_DIR / "properties" / f"{prop_id}.yaml"
        else:
            path = DATA_DIR / "properties" / "old" / f"{prop_id}.yaml"
        properties = cls.from_yaml_file(path)
        assert isinstance(properties, cls)
        return properties

    @classmethod
    def load_from_path(cls, path: Path) -> "Properties":
        properties = cls.from_yaml_file(path)
        assert isinstance(properties, cls)
        return properties


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(Properties)
DumpMeta(key_transform="SNAKE").bind_to(Properties)


@dataclass
class Instructions(YAMLWizard):
    cot: str
    direct: str
    open_ended_cot: str

    @classmethod
    def load(cls, instr_id: str) -> "Instructions":
        with open(DATA_DIR / "instructions.yaml", "r") as f:
            instr_dict = yaml.safe_load(f)[instr_id]
        return fromdict(cls, instr_dict)


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(Instructions)
DumpMeta(key_transform="SNAKE").bind_to(Instructions)

@dataclass
class Question(YAMLWizard):
    q_str: str
    q_str_open_ended: str
    x_name: str
    y_name: str
    x_value: int | float
    y_value: int | float


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(Question)
DumpMeta(key_transform="SNAKE").bind_to(Question)


@dataclass
class MathQuestion(YAMLWizard):
    name: str
    problem: str
    solution: str


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathQuestion)
DumpMeta(key_transform="SNAKE").bind_to(MathQuestion)


@dataclass
class StepFaithfulness(YAMLWizard):
    step_str: str

    reasoning: str
    unfaithfulness: str

    # We also generate o1 responses to check the steps initially flagged:
    reasoning_check: str | None = None
    unfaithfulness_check: (
        Literal["LATENT_ERROR_CORRECTION", "ILLOGICAL", "OTHER"] | None
    ) = None
    # TODO(arthur): Add this to normal eval too?
    severity_check: Literal["TRIVIAL", "MINOR", "MAJOR", "CRITICAL"] | None = None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(StepFaithfulness)
DumpMeta(key_transform="SNAKE").bind_to(StepFaithfulness)


@dataclass
class MathResponse(MathQuestion):
    # list[str] if split into COT steps
    # list[StepFaithfulness] if split into COT steps,
    # and using the faithfulness eval
    model_answer: list[str] | list[StepFaithfulness]
    model_thinking: str | None

    # From evaluate_putnam_answers.py:
    correctness_explanation: str | None = None
    correctness_is_correct: bool | None = None
    correctness_classification: (
        Literal["EQUIVALENT", "NOT_EQUIVALENT", "NA_NEITHER", "NA_BOTH"] | None
    ) = None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathResponse)
DumpMeta(key_transform="SNAKE").bind_to(MathResponse)


@dataclass
class SamplingParams(YAMLWizard):
    temperature: float
    top_p: float
    max_new_tokens: int

    @property
    def id(self) -> str:
        return f"T{self.temperature}_P{self.top_p}_M{self.max_new_tokens}"


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(SamplingParams)
DumpMeta(key_transform="SNAKE").bind_to(SamplingParams)


@dataclass
class DatasetParams(YAMLWizard):
    prop_id: str
    comparison: Literal["gt", "lt"]
    answer: Literal["YES", "NO"]
    max_comparisons: int
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    suffix: str | None = None

    @property
    def pre_id(self) -> str:
        return f"{self.comparison}_{self.answer}_{self.max_comparisons}"

    @property
    def id(self) -> str:
        id = f"{self.prop_id}_{self.pre_id}_{self.uuid}"
        if self.suffix is not None:
            id = f"{id}_{self.suffix}"
        return id

    @property
    def qs_dataset_path(self) -> Path:
        return DATA_DIR / "questions" / self.pre_id / f"{self.id}.yaml"

    def direct_eval_path(self, instr_id: str, model_id: str) -> Path:
        return (
            DATA_DIR
            / "direct_eval"
            / instr_id
            / self.pre_id
            / self.id
            / f"{model_id.replace('/', '__')}.yaml"
        )

    def cot_eval_path(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> Path:
        return (
            DATA_DIR
            / "cot_eval"
            / instr_id
            / sampling_params.id
            / self.pre_id
            / self.id
            / f"{model_id.replace('/', '__')}.yaml"
        )

    def cot_responses_path(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> Path:
        return (
            DATA_DIR
            / "cot_responses"
            / instr_id
            / sampling_params.id
            / self.pre_id
            / self.id
            / f"{model_id.replace('/', '__')}.yaml"
        )

    @classmethod
    def from_id(cls, dataset_id: str) -> "DatasetParams":
        if len(dataset_id.split("_")) == 5:
            prop_id, comparison, answer, max_comparisons, uuid = dataset_id.split("_")
            return cls(
                prop_id=prop_id,
                comparison=comparison,  # type: ignore
                answer=answer,  # type: ignore
                max_comparisons=int(max_comparisons),
                uuid=uuid,
            )
        elif len(dataset_id.split("_")) == 6:
            prop_id, comparison, answer, max_comparisons, uuid, suffix = dataset_id.split("_")
            return cls(
                prop_id=prop_id,
                comparison=comparison,  # type: ignore
                answer=answer,  # type: ignore
                max_comparisons=int(max_comparisons),
                uuid=uuid,
                suffix=suffix,
            )
        else:
            raise ValueError(f"Invalid dataset_id: {dataset_id}")

    def load_qs_dataset(self) -> "QsDataset":
        qsds = QsDataset.from_yaml_file(self.qs_dataset_path)
        assert isinstance(qsds, QsDataset)
        return qsds

    def load_direct_eval(self, instr_id: str, model_id: str) -> "DirectEval":
        direct_eval = DirectEval.from_yaml_file(
            self.direct_eval_path(instr_id, model_id)
        )
        assert isinstance(direct_eval, DirectEval)
        return direct_eval

    def load_cot_eval(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> "CotEval":
        cot_eval = CotEval.from_yaml_file(
            self.cot_eval_path(instr_id, model_id, sampling_params)
        )
        assert isinstance(cot_eval, CotEval)
        return cot_eval

    def load_old_cot_eval(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> "OldCotEval":
        cot_eval = OldCotEval.from_yaml_file(
            self.cot_eval_path(instr_id, model_id, sampling_params)
        )
        assert isinstance(cot_eval, OldCotEval)
        return cot_eval


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(DatasetParams)
DumpMeta(key_transform="SNAKE").bind_to(DatasetParams)


@dataclass
class MathDatasetParams(YAMLWizard):
    description: str

    # These create nested directories if used:
    id: str | None
    pre_id: str | None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathDatasetParams)
DumpMeta(key_transform="SNAKE").bind_to(MathDatasetParams)


@dataclass
class MathQsDataset(YAMLWizard):
    questions: list[MathQuestion]
    params: MathDatasetParams

    def dataset_path(self) -> Path:
        dataset_path = DATA_DIR / "math_datasets"
        if self.params.id:
            dataset_path /= self.params.id
        if self.params.pre_id:
            dataset_path /= self.params.pre_id
        dataset_path /= "dataset.yaml"
        return dataset_path

    def save(self, force: bool = False) -> Path:
        self.to_yaml_file(self.dataset_path())
        # mkdir the directories
        self.dataset_path().parent.mkdir(parents=True, exist_ok=True)
        # exist if exists and not force
        if self.dataset_path().exists() and not force:
            raise FileExistsError(f"File already exists: {self.dataset_path()}")
        return self.dataset_path()


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathQsDataset)
DumpMeta(key_transform="SNAKE").bind_to(MathQsDataset)


@dataclass
class QsDataset(YAMLWizard):
    question_by_qid: dict[str, Question]
    params: DatasetParams

    @classmethod
    def load(cls, dataset_id: str) -> "QsDataset":
        params = DatasetParams.from_id(dataset_id)
        qs_dataset = params.load_qs_dataset()
        assert qs_dataset.params == params
        return qs_dataset

    @classmethod
    def load_from_path(cls, path: Path) -> "QsDataset":
        qs_dataset = cls.from_yaml_file(path)
        assert isinstance(qs_dataset, cls)
        return qs_dataset

    def save(self) -> Path:
        self.params.qs_dataset_path.parent.mkdir(exist_ok=True)
        self.to_yaml_file(self.params.qs_dataset_path)
        logging.warning(
            f"Saved {len(self.question_by_qid)} questions to \n"
            f"{self.params.qs_dataset_path.stem}\n"
            f"{self.params.qs_dataset_path}\n"
        )
        return self.params.qs_dataset_path


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(QsDataset)
DumpMeta(key_transform="SNAKE").bind_to(QsDataset)


@dataclass
class DirectEvalProbs(YAMLWizard):
    p_yes: float
    p_no: float


def get_path(directory: Path, model_id: str, suffix: str = "") -> Path:
    directory.mkdir(exist_ok=True, parents=True)
    model_id = model_id.replace("/", "__")
    path = directory / f"{model_id}{suffix}.yaml"
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

    @classmethod
    def load(cls, path: Path) -> "DirectEval":
        direct_eval = cls.from_yaml_file(path)
        assert isinstance(direct_eval, cls)
        return direct_eval


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(DirectEval)
DumpMeta(key_transform="SNAKE").bind_to(DirectEval)


@dataclass
class DefaultSamplingParams:
    id: Literal["default_sampling_params"] = "default_sampling_params"


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(DefaultSamplingParams)
DumpMeta(key_transform="SNAKE").bind_to(DefaultSamplingParams)


@dataclass
class MathCotResponses(YAMLWizard):
    # For normal datasets
    # qid -> {uuid -> response_str}
    #
    # Can also map to a MathResponse not a response_str
    #
    # TODO(arthur): Verify that this still works fine for the rest
    # of the codebase
    responses_by_qid: dict[str, dict[str, MathResponse | str]]
    model_id: str
    instr_id: str
    ds_params: MathDatasetParams
    sampling_params: DefaultSamplingParams

    def get_path(self, suffix: str = "") -> Path:
        directory = (
            DATA_DIR
            / "cot_responses"
            / self.instr_id
            / self.sampling_params.id
            / "filtered_putnambench"
        )
        return get_path(directory, self.model_id, suffix)

    def save(self, suffix: str = "", path: str | Path | None = None) -> Path:
        if path is None:
            path = self.get_path(suffix)
        else:
            path = Path(path)

        # make the parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "MathCotResponses":
        cot_responses = cls.from_yaml_file(path)
        assert isinstance(cot_responses, cls)
        return cot_responses


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathCotResponses)
DumpMeta(key_transform="SNAKE").bind_to(MathCotResponses)


@dataclass
class CotEvalResult(YAMLWizard):
    final_answer: Literal["REFUSED", "YES", "NO", "UNKNOWN", "FAILED_EVAL"]
    equal_values: (
        Literal["TRUE", "FALSE", "FAILED_EVAL"] | None
    )  # Only relevant for NO answers
    explanation_final_answer: str | None
    explanation_equal_values: str | None
    result: Literal["YES", "NO", "UNKNOWN"] | None = None

    def __post_init__(self):
        """Compute the result if not already set."""
        if self.result is None:
            if self.final_answer == "NO":
                if self.equal_values == "FALSE":
                    self.result = "NO"
                elif self.equal_values == "TRUE":
                    self.result = "UNKNOWN"
            elif self.final_answer == "YES":
                self.result = "YES"
            else:
                self.result = "UNKNOWN"


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(CotEvalResult)
DumpMeta(key_transform="SNAKE").bind_to(CotEvalResult)


@dataclass
class CotEval(YAMLWizard):
    results_by_qid: dict[
        str, dict[str, CotEvalResult]
    ]  # qid -> {response_uuid -> result}
    model_id: str
    instr_id: str
    ds_params: DatasetParams
    sampling_params: SamplingParams
    evaluator: str = "heuristic"

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

    @classmethod
    def load(cls, path: Path) -> "CotEval":
        cot_eval = cls.from_yaml_file(path)
        assert isinstance(cot_eval, cls)
        return cot_eval


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(CotEval)
DumpMeta(key_transform="SNAKE").bind_to(CotEval)


@dataclass
class OldCotEval(YAMLWizard):
    results_by_qid: dict[str, dict[str, str]]  # qid -> {response_uuid -> result}
    model_id: str
    instr_id: str
    ds_params: DatasetParams
    sampling_params: SamplingParams
    evaluator: str = "heuristic"

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

    @classmethod
    def load(cls, path: Path) -> "OldCotEval":
        cot_eval = cls.from_yaml_file(path)
        assert isinstance(cot_eval, cls)
        return cot_eval


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(OldCotEval)
DumpMeta(key_transform="SNAKE").bind_to(OldCotEval)


@dataclass
class AnswerFlippingEval(YAMLWizard):
    label_by_qid: dict[
        str, dict[str, Literal["YES", "NO", "UNKNOWN", "NO_REASONING", "FAILED_EVAL"]]
    ]  # qid -> {response_uuid -> result}
    raw_analysis_by_qid: dict[
        str, dict[str, str]
    ]  # qid -> {response_uuid -> raw_analysis}
    model_id: str
    evaluator_model_ids: list[str]
    instr_id: str
    ds_params: DatasetParams
    sampling_params: SamplingParams

    def save(self) -> Path:
        directory = (
            DATA_DIR
            / "answer_flipping_eval"
            / self.instr_id
            / self.sampling_params.id
            / self.ds_params.pre_id
            / self.ds_params.id
        )
        path = get_path(directory, self.model_id)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "AnswerFlippingEval":
        answer_flipping_eval = cls.from_yaml_file(path)
        assert isinstance(answer_flipping_eval, cls)
        return answer_flipping_eval


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AnswerFlippingEval)
DumpMeta(key_transform="SNAKE").bind_to(AnswerFlippingEval)


@dataclass
class GroundTruthEvalData(YAMLWizard):
    gt_by_qid: dict[
        str, dict[str, dict[str, str]]
    ]  # qid -> {response_uuid -> {entity -> extracted_value}}
    open_ended_responses_by_qid: dict[
        str, dict[str, str]
    ]  # qid -> {response_uuid -> response_str}
    model_id: str
    evaluator_model_id: str
    instr_id: str
    ds_params: DatasetParams
    sampling_params: SamplingParams

    def save(self) -> Path:
        directory = (
            DATA_DIR
            / "gt_eval_data"
            / self.instr_id
            / self.sampling_params.id
            / self.ds_params.pre_id
            / self.ds_params.id
        )
        path = get_path(directory, self.model_id)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "GroundTruthEvalData":
        ground_truth_eval_data = cls.from_yaml_file(path)
        assert isinstance(ground_truth_eval_data, cls)
        return ground_truth_eval_data


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(GroundTruthEvalData)
DumpMeta(key_transform="SNAKE").bind_to(GroundTruthEvalData)


@dataclass
class Problem(YAMLWizard):
    q_str: str
    answer: str
    answer_without_reasoning: str | None
    split: str | None
    category: str | None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(Problem)
DumpMeta(key_transform="SNAKE").bind_to(Problem)


@dataclass
class ProblemDataset(YAMLWizard):
    dataset_name: str
    problems_by_qid: dict[str, Problem]

    @classmethod
    def load(cls, dataset_name: str) -> "ProblemDataset":
        problem_dataset = cls.from_yaml_file(
            DATA_DIR / "problems" / f"{dataset_name}.yaml"
        )
        assert isinstance(problem_dataset, cls)
        return problem_dataset

    def save(self) -> Path:
        directory = DATA_DIR / "problems"
        path = get_path(directory, self.dataset_name)
        self.to_yaml_file(path)
        return path


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(ProblemDataset)
DumpMeta(key_transform="SNAKE").bind_to(ProblemDataset)


@dataclass
class CoTPath(YAMLWizard):
    cot_path_by_qid: dict[
        str, dict[str, dict[int, str]]
    ]  # qid -> {response_uuid -> {step_number -> step_str}}
    model_id: str
    problem_dataset_name: str
    sampling_params: SamplingParams

    def save(self) -> Path:
        directory = DATA_DIR / "cot_paths" / self.problem_dataset_name
        path = get_path(directory, self.model_id)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load_from_path(cls, path: Path) -> "CoTPath":
        cot_path = cls.from_yaml_file(path)
        assert isinstance(cot_path, cls)
        return cot_path

    @classmethod
    def load(cls, model_id: str, dataset_id: str) -> "CoTPath":
        path = (
            DATA_DIR / "cot_paths" / dataset_id / f"{model_id.replace('/', '__')}.yaml"
        )
        return cls.load_from_path(path)


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(CoTPath)
DumpMeta(key_transform="SNAKE").bind_to(CoTPath)


@dataclass
class AnswerCorrectnessResult(YAMLWizard):
    problem_description_status: Literal["CLEAR", "INCOMPLETE", "AMBIGUOUS", "UNKNOWN"]
    problem_description_explanation: str | None
    answer_status: Literal["CORRECT", "INCORRECT", "UNCERTAIN", "UNKNOWN"]
    answer_explanation: str | None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AnswerCorrectnessResult)
DumpMeta(key_transform="SNAKE").bind_to(AnswerCorrectnessResult)


@dataclass
class FirstPassNodeStatus(YAMLWizard):
    node_status: Literal["CORRECT", "INCORRECT", "UNCERTAIN", "UNKNOWN"]
    explanation: str


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(FirstPassNodeStatus)
DumpMeta(key_transform="SNAKE").bind_to(FirstPassNodeStatus)


@dataclass
class FirstPassResponseResult(YAMLWizard):
    steps_status: dict[
        int,
        FirstPassNodeStatus,
    ]


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(FirstPassResponseResult)
DumpMeta(key_transform="SNAKE").bind_to(FirstPassResponseResult)


@dataclass
class SecondPassNodeStatus(YAMLWizard):
    node_status: Literal["INCORRECT", "UNUSED", "UNFAITHFUL", "NONE", "UNKNOWN"]
    node_severity: Literal["TRIVIAL", "MINOR", "MAJOR", "CRITICAL", "UNKNOWN"]
    explanation: str


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(SecondPassNodeStatus)
DumpMeta(key_transform="SNAKE").bind_to(SecondPassNodeStatus)


@dataclass
class SecondPassResponseResult(YAMLWizard):
    steps_status: dict[
        int,
        SecondPassNodeStatus,
    ]
    reasoning: str | None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(SecondPassResponseResult)
DumpMeta(key_transform="SNAKE").bind_to(SecondPassResponseResult)


@dataclass
class ThirdPassNodeStatus(YAMLWizard):
    is_unfaithful: bool
    node_severity: Literal["TRIVIAL", "MINOR", "MAJOR", "CRITICAL", "UNKNOWN"]
    explanation: str


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(ThirdPassNodeStatus)
DumpMeta(key_transform="SNAKE").bind_to(ThirdPassNodeStatus)


@dataclass
class ThirdPassResponseResult(YAMLWizard):
    steps_status: dict[
        int,
        ThirdPassNodeStatus,
    ]


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(ThirdPassResponseResult)
DumpMeta(key_transform="SNAKE").bind_to(ThirdPassResponseResult)


@dataclass
class CoTPathEval(YAMLWizard):
    # Zero pass evaluation: whether the full path is correct or not
    answer_correct_by_qid: dict[
        str,
        dict[
            str,
            AnswerCorrectnessResult,
        ],
    ]  # qid -> {response_uuid -> (answer_correct, explanation)}}

    # First pass evaluation: labeling of incorrect/correct nodes in isolation
    # We are not doing any equivalence checking here because we only have one path per problem.
    first_pass_eval_by_qid: dict[
        str,
        dict[str, FirstPassResponseResult],
    ]  # qid -> {response_uuid -> {step_number -> (node_label, explanation)}}

    # Second pass evaluation for incorrect nodes in paths with correct answers
    # - labeling nodes as incorrect, unused, or unfaithful
    # - labeling severity of nodes
    second_pass_eval_by_qid: dict[
        str,
        dict[
            str,
            SecondPassResponseResult,
        ],
    ]  # qid -> {response_uuid -> {step_number -> (node_status, node_severity, explanation)}}

    # Third pass evaluation for unfaithful nodes with severity minor or major
    third_pass_eval_by_qid: dict[
        str,
        dict[str, ThirdPassResponseResult],
    ]  # qid -> {response_uuid -> {step_number -> (node_status, node_severity)}}

    model_id: str
    evaluator_model_id: str
    problem_dataset_name: str
    sampling_params: SamplingParams

    def save(self) -> Path:
        directory = DATA_DIR / "cot_path_eval" / self.problem_dataset_name
        path = get_path(directory, self.model_id)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load_from_path(cls, path: Path) -> "CoTPathEval":
        cot_path_eval = cls.from_yaml_file(path)
        assert isinstance(cot_path_eval, cls)
        return cot_path_eval

    @classmethod
    def load(cls, model_id: str, dataset_id: str) -> "CoTPathEval":
        path = (
            DATA_DIR
            / "cot_path_eval"
            / dataset_id
            / f"{model_id.replace('/', '__')}.yaml"
        )
        return cls.load_from_path(path)


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(CoTPathEval)
DumpMeta(key_transform="SNAKE").bind_to(CoTPathEval)


@dataclass(frozen=True)
class QuestionResponseId:
    """Identifier for a response."""

    qid: str
    uuid: str


@dataclass
class AnthropicBatchInfo(YAMLWizard):
    """Information about a batch submitted to Anthropic's batch API."""

    batch_id: str
    instr_id: str
    ds_params: DatasetParams
    created_at: str
    custom_id_map: dict[str, QuestionResponseId]
    evaluated_model_id: str
    evaluated_sampling_params: SamplingParams
    evaluator_model_id: str | None
    evaluator_sampling_params: SamplingParams | None

    def save(self) -> Path:
        """Save batch info to disk."""
        directory = (
            DATA_DIR
            / "anthropic_batches"
            / self.instr_id
            / self.evaluated_sampling_params.id
            / self.ds_params.pre_id
            / self.ds_params.id
        )
        path = get_path(directory, f"{self.evaluated_model_id}_{self.batch_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)
        return path

    @classmethod
    def load(cls, path: Path) -> "AnthropicBatchInfo":
        batch_info = cls.from_yaml_file(path)
        assert isinstance(batch_info, cls)
        return batch_info


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AnthropicBatchInfo)
DumpMeta(key_transform="SNAKE").bind_to(AnthropicBatchInfo)


@dataclass
class AmbiguityEval(YAMLWizard):
    """Results from evaluating question ambiguity."""

    # Store all evaluations for each question
    ambiguity_by_qid: dict[str, list[Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]]]
    analysis_by_qid: dict[str, list[str | None]]
    # Final determination based on all evaluations
    final_ambiguity_by_qid: dict[str, Literal["CLEAR", "AMBIGUOUS", "FAILED_EVAL"]]
    model_id: str
    instr_id: str
    ds_params: DatasetParams
    sampling_params: SamplingParams

    def save(self) -> Path:
        """Save ambiguity evaluation to disk."""
        directory = (
            DATA_DIR
            / "ambiguity_eval"
            / self.instr_id
            / self.sampling_params.id
            / self.ds_params.pre_id
        )
        path = get_path(directory, self.ds_params.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "AmbiguityEval":
        ambiguity_eval = cls.from_yaml_file(path)
        assert isinstance(ambiguity_eval, cls)
        return ambiguity_eval


LoadMeta(
    v1=False, v1_unsafe_parse_dataclass_in_union=False, key_transform="SNAKE"
).bind_to(AmbiguityEval)
DumpMeta(key_transform="SNAKE").bind_to(AmbiguityEval)


@dataclass
class PropEval(YAMLWizard):
    """Results from evaluating properties."""

    popularity_by_entity_name: dict[str, int]
    model_id: str
    sampling_params: SamplingParams
    prop_id: str

    def save(self) -> Path:
        """Save property evaluation to disk."""
        directory = DATA_DIR / "prop_eval" / self.sampling_params.id
        path = get_path(directory, self.prop_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "PropEval":
        prop_eval = cls.from_yaml_file(path)
        assert isinstance(prop_eval, cls)
        return prop_eval

    @classmethod
    def load_id(cls, prop_id: str) -> "PropEval":
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=1,
        )
        directory = DATA_DIR / "prop_eval" / sampling_params.id
        path = get_path(directory, prop_id)
        return cls.load(path)


LoadMeta(
    v1=False, v1_unsafe_parse_dataclass_in_union=False, key_transform="SNAKE"
).bind_to(PropEval)
DumpMeta(key_transform="SNAKE").bind_to(PropEval)


@dataclass
class OpenAIBatchInfo(YAMLWizard):
    """Information about a batch submitted to OpenAI's batch API."""

    batch_id: str
    instr_id: str
    ds_params: DatasetParams
    created_at: str
    custom_id_map: dict[str, QuestionResponseId]
    evaluated_model_id: str
    evaluated_sampling_params: SamplingParams
    evaluator_model_id: str | None
    evaluator_sampling_params: SamplingParams | None
    metadata: dict[str, Any]

    def save(self) -> Path:
        """Save batch info to disk."""
        directory = (
            DATA_DIR
            / "openai_batches"
            / self.instr_id
            / self.evaluated_sampling_params.id
            / self.ds_params.pre_id
            / self.ds_params.id
        )
        path = get_path(directory, f"{self.evaluated_model_id}_{self.batch_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)
        return path

    @classmethod
    def load(cls, path: Path) -> "OpenAIBatchInfo":
        batch_info = cls.from_yaml_file(path)
        assert isinstance(batch_info, cls)
        return batch_info


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(OpenAIBatchInfo)
DumpMeta(key_transform="SNAKE").bind_to(OpenAIBatchInfo)


@dataclass
class AtCoderSample(YAMLWizard):
    input: str
    output: str
    explanation: str | None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderSample)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderSample)


@dataclass
class AtCoderProblem(YAMLWizard):
    id: str
    problem_letter: str
    url: str
    statement: str | None
    constraints: str | None
    io_format: str | None
    samples: list[AtCoderSample]


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderProblem)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderProblem)


@dataclass
class AtCoderContest(YAMLWizard):
    id: str
    url: str
    start_time: str
    problems: list[AtCoderProblem]


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderContest)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderContest)


@dataclass
class AtCoderQuestion(YAMLWizard):
    """Represents a single AtCoder problem with its solution attempt."""

    name: str  # contest_id_problem_id
    problem: str  # Combined prompt with all problem info
    cpp_solution: str | None = None  # The C++ solution if available


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderQuestion)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderQuestion)


@dataclass
class AtCoderDatasetParams(YAMLWizard):
    """Parameters for an AtCoder dataset."""

    description: str
    id: str | None
    pre_id: str | None
    dataset_type: Literal["atcoder"] = "atcoder"


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderDatasetParams)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderDatasetParams)


@dataclass
class AtCoderDataset(YAMLWizard):
    """Collection of AtCoder problems."""

    questions: list[AtCoderQuestion]
    params: AtCoderDatasetParams


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderDataset)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderDataset)


@dataclass
class AtCoderResponse(AtCoderQuestion):
    """A response to an AtCoder problem."""

    model_answer: list[str] | list[StepFaithfulness] | None = (
        None  # The C++ solution, wrapped in tags
    )
    model_thinking: str | None = None  # The step-by-step reasoning

    correctness_explanation: str | None = None
    correctness_is_correct: bool | None = None
    correctness_classification: (
        Literal["EQUIVALENT", "NOT_EQUIVALENT", "NA_NEITHER", "NA_BOTH"] | None
    ) = None
    cpp_solution: str | None = None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderResponse)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderResponse)


@dataclass
class AtCoderStats(YAMLWizard):
    """Statistics for AtCoder submissions."""

    finding_code_failed: int = -1
    compilation_failed: int = -1
    atcodertools_cmd_failed: int = -1
    solution_failed: int = -1
    solution_passed: int = -1

    @classmethod
    def zeros(cls) -> "AtCoderStats":
        return cls(
            finding_code_failed=0,
            compilation_failed=0,
            atcodertools_cmd_failed=0,
            solution_failed=0,
            solution_passed=0,
        )


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderStats)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderStats)


@dataclass
class CotResponses(YAMLWizard):
    # For normal datasets
    # qid -> {uuid -> response_str or a MathResponse or AtCoderResponse}
    responses_by_qid: dict[str, dict[str, MathResponse | AtCoderResponse | str]]
    model_id: str
    instr_id: str
    ds_params: AtCoderDatasetParams | MathDatasetParams | DatasetParams
    sampling_params: SamplingParams | DefaultSamplingParams

    # Fields for AtCoder submission tracking:
    atcoder_stats: AtCoderStats = field(default_factory=AtCoderStats)

    def get_path(self, suffix: str = "") -> Path:
        if isinstance(self.ds_params, DatasetParams) and isinstance(
            self.sampling_params, SamplingParams
        ):
            return self.ds_params.cot_responses_path(
                instr_id=self.instr_id,
                model_id=self.model_id,
                sampling_params=self.sampling_params,
            )

        if isinstance(self.ds_params, MathDatasetParams):
            end_dir = "filtered_putnambench"
        elif isinstance(self.ds_params, AtCoderDatasetParams):
            end_dir = "atcoder"
        else:
            raise ValueError(f"Unknown dataset type: {type(self.ds_params)}")

        directory = (
            DATA_DIR
            / "cot_responses"
            / self.instr_id
            / self.sampling_params.id
            / end_dir
        )
        return get_path(directory, self.model_id, suffix)

    def save(self, suffix: str = "", path: str | Path | None = None) -> Path:
        if path is None:
            path = self.get_path(suffix)
        else:
            path = Path(path)

        # make the parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "CotResponses":
        cot_responses = cls.from_yaml_file(path)
        assert isinstance(cot_responses, cls)
        return cot_responses


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(CotResponses)
DumpMeta(key_transform="SNAKE").bind_to(CotResponses)


@dataclass
class SplitCotResponses(YAMLWizard):
    # qid -> {uuid -> [step_str]}
    # ...or uses the list[str] for the model answer
    # if using MathResponse
    split_responses_by_qid: dict[str, dict[str, MathResponse | AtCoderResponse]]

    successfully_split_count: int
    failed_to_split_count: int
    instr_id: str
    ds_params: AtCoderDatasetParams | MathDatasetParams
    sampling_params: DefaultSamplingParams
    model_id: str

    def save(self, suffix: str = "", path: str | Path | None = None) -> Path:
        if path is None:
            directory = DATA_DIR / "split_cot_responses" / self.instr_id
            if self.ds_params.pre_id is not None:
                directory /= self.ds_params.pre_id
            if self.ds_params.id is not None:
                directory /= self.ds_params.id
            if self.sampling_params.id is not None:
                directory /= self.sampling_params.id
            path = get_path(directory, self.model_id, suffix="_split" + suffix)
        else:
            path = Path(path)

        # make the parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "SplitCotResponses":
        split_cot_responses = cls.from_yaml_file(path)
        assert isinstance(split_cot_responses, cls)
        return split_cot_responses


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(SplitCotResponses)
DumpMeta(key_transform="SNAKE").bind_to(SplitCotResponses)


@dataclass
class RAGSource:
    url: str
    title: str
    content: str | None
    relevant_snippet: str | None

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(RAGSource)
DumpMeta(key_transform="SNAKE").bind_to(RAGSource)

@dataclass
class RAGValue:
    value: str
    source: RAGSource

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(RAGValue)
DumpMeta(key_transform="SNAKE").bind_to(RAGValue)


@dataclass
class PropRAGEval(YAMLWizard):
    """Results from evaluating properties using RAG."""
    values_by_entity_name: dict[str, list[RAGValue]]
    model_id: str
    sampling_params: SamplingParams
    prop_id: str

    def save(self) -> Path:
        """Save property RAG evaluation to disk."""
        directory = DATA_DIR / "prop_rag_eval" / self.sampling_params.id
        path = get_path(directory, self.prop_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "PropRAGEval":
        prop_rag_eval = cls.from_yaml_file(path)
        assert isinstance(prop_rag_eval, cls)
        return prop_rag_eval

    @classmethod
    def load_id(cls, prop_id: str, sampling_params: SamplingParams) -> "PropRAGEval":
        directory = DATA_DIR / "prop_rag_eval" / sampling_params.id
        path = get_path(directory, prop_id)
        return cls.load(path)

    def get_path(self) -> Path:
        directory = DATA_DIR / "prop_rag_eval" / self.sampling_params.id
        path = get_path(directory, self.prop_id)
        return path
