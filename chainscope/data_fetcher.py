#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from chainscope.typing import (DATA_DIR, CotEval, CotEvalResult, CotResponses,
                               DatasetParams, Instructions, QsDataset,
                               SamplingParams, UnfaithfulnessPairsDataset,
                               UnfaithfulnessPatternEval)
from chainscope.utils import (load_model_and_tokenizer, make_chat_prompt,
                              sort_models)


@dataclass
class ResponseData:
    response_id: str
    response_str: str
    is_correct: bool
    model_answer: Literal["YES", "NO", "UNKNOWN"]
    prompt_str: str
    evidence_of_unfaithfulness: list[Literal["fact-manipulation", "argument-switching", "answer-flipping", "other", "none"]] | None
    answer_flipping_classification: Literal["YES", "NO", "UNCLEAR", "FAILED_EVAL"] | None


@dataclass
class QuestionData:
    qid: str
    q_str: str
    dataset_id: str
    correct_answer: Literal["YES", "NO"]
    responses: dict[str, ResponseData]  # response_id -> ResponseData
    accuracy: float


@dataclass
class QuestionPairData:
    q1: QuestionData
    q2: QuestionData
    is_unfaithful_pair: bool
    unfaithfulness_patterns: list[str]


@dataclass
class DataFetcher:
    def __init__(self, model_id: str, dataset_suffix: str = "non-ambiguous-hard-2", instr_id: str = "instr-wm"):
        self.model_id = model_id
        self.model_id_no_slash = model_id.replace("/", "__")
        self.model_name = model_id.split("/")[-1]

        self.dataset_suffix = dataset_suffix
        self.instr_id = instr_id
        self.inst_template = Instructions.load(self.instr_id).cot

        self.df = self._load_df()
        # Columns: q_str, qid, prop_id, comparison, answer, dataset_id, dataset_suffix, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate, yes_count, no_count, unknown_count, known_count, total_count

        self._unfaithfulness_pairs_datasets: dict[str, UnfaithfulnessPairsDataset] = {}
        self._unfaithfulness_pattern_evals: dict[str, UnfaithfulnessPatternEval] = {}
        self._responses_cache: dict[str, CotResponses] = {}
        self._evals_cache: dict[str, CotEval] = {}
        self._faithfulness_cache: dict[str, UnfaithfulnessPairsDataset] = {}
        self._qs_dataset_cache: dict[str, QsDataset] = {}

    def get_faithful_question_pairs(self, prop_id: str | None = None, comparison: str | None = None) -> list[QuestionPairData]:
        """
        Returns all faithful question pairs.
        """
        faithful_pairs: list[QuestionPairData] = []
        prop_ids_filter = self.get_available_prop_ids()

        if prop_id is not None:
            # Apply the prop_id filter
            assert prop_id in prop_ids_filter, f"Property ID {prop_id} not found in {prop_ids_filter}"
            prop_ids_filter = [prop_id]

        for (prop_id_group, comparison_group), group in self.df.groupby(
            ["prop_id", "comparison"]
        ):
            assert isinstance(prop_id_group, str), f"Expected prop_id to be a string, got {type(prop_id_group)}"
            assert isinstance(comparison_group, str), f"Expected comparison to be a string, got {type(comparison_group)}"

            if prop_id_group not in prop_ids_filter:
                print(f"Skipping prop_id {prop_id_group} because it is not in the filter")
                continue

            if comparison is not None and comparison_group != comparison:
                print(f"Skipping comparison {comparison_group} because it is not {comparison}")
                continue

            # Find pairs of questions with reversed x_name and y_name
            pairs = {}
            for _, row in group.iterrows():
                key = frozenset([row.x_name, row.y_name])
                if key not in pairs:
                    pairs[key] = []
                pairs[key].append(row)
            pairs = {k: v for k, v in pairs.items() if len(v) == 2}

            print(f"Found {len(pairs)} pairs for prop_id {prop_id_group} and comparison {comparison_group}")

            faithfulness_dataset = None
            try:
                faithfulness_dataset = self._load_faithfulness_dataset(prop_id_group)
            except FileNotFoundError:
                # This is fine, means all questions in this prop_id are faithful
                pass

            for pair in pairs.values():
                q1, q2 = pair

                # Assert if q1 is YES then q2 is NO and vice versa
                assert (q1.answer == "YES" and q2.answer == "NO") or (q1.answer == "NO" and q2.answer == "YES"), f"Question pair {q1.qid} and {q2.qid} are not faithful"

                q1_raw_responses = self._load_responses(q1.qid, q1.dataset_id, q1.temperature, q1.top_p, q1.max_new_tokens)
                q2_raw_responses = self._load_responses(q2.qid, q2.dataset_id, q2.temperature, q2.top_p, q2.max_new_tokens)
                q1_evals = self._load_evals(q1.qid, q1.dataset_id, q1.temperature, q1.top_p, q1.max_new_tokens)
                q2_evals = self._load_evals(q2.qid, q2.dataset_id, q2.temperature, q2.top_p, q2.max_new_tokens)

                q1_fsps = self._load_fsps(list(q1_raw_responses.keys()), q1.dataset_id, q1.temperature, q1.top_p, q1.max_new_tokens)
                q2_fsps = self._load_fsps(list(q2_raw_responses.keys()), q2.dataset_id, q2.temperature, q2.top_p, q2.max_new_tokens)

                q1_base_prompt = self.inst_template.format(question=q1.q_str)
                q2_base_prompt = self.inst_template.format(question=q2.q_str)

                q1_responses = {resp_id: ResponseData(
                    response_id=resp_id,
                    response_str=resp_str,
                    is_correct=q1_evals[resp_id].result == q1.answer,
                    model_answer=q1_evals[resp_id].result if q1_evals[resp_id].result is not None else "UNKNOWN",  # type: ignore
                    prompt_str=q1_fsps[resp_id] + "\n\n" + q1_base_prompt if q1_fsps[resp_id] is not None else q1_base_prompt,
                    evidence_of_unfaithfulness=None,
                    answer_flipping_classification=None,
                ) for resp_id, resp_str in q1_raw_responses.items()}

                q2_responses = {resp_id: ResponseData(
                    response_id=resp_id,
                    response_str=resp_str,
                    is_correct=q2_evals[resp_id].result == q2.answer,
                    model_answer=q2_evals[resp_id].result if q2_evals[resp_id].result else "UNKNOWN", # type: ignore
                    prompt_str=q2_fsps[resp_id] + "\n\n" + q2_base_prompt if q2_fsps[resp_id] is not None else q2_base_prompt,
                    evidence_of_unfaithfulness=None,
                    answer_flipping_classification=None,
                ) for resp_id, resp_str in q2_raw_responses.items()}

                is_unfaithful = faithfulness_dataset is not None and (q1.qid in faithfulness_dataset.questions_by_qid or q2.qid in faithfulness_dataset.questions_by_qid)
                if not is_unfaithful:
                    faithful_pairs.append(QuestionPairData(
                        q1=QuestionData(
                            qid=q1.qid,
                            q_str=q1.q_str,
                            dataset_id=q1.dataset_id,
                            correct_answer=q1.answer,
                            accuracy=q1.p_correct,
                            responses=q1_responses,
                        ),
                        q2=QuestionData(
                            qid=q2.qid,
                            q_str=q2.q_str,
                            dataset_id=q2.dataset_id,
                            correct_answer=q2.answer,
                            accuracy=q2.p_correct,
                            responses=q2_responses,
                        ),
                        is_unfaithful_pair=False,
                        unfaithfulness_patterns=[],
                    ))

        return faithful_pairs

    def get_unfaithful_question_pairs(self, prop_id: str | None = None, comparison: str | None = None, unfaithfulness_type: str | None = None) -> list[QuestionPairData]:
        """
        Returns all unfaithful question pairs.
        An optional filter can be applied for a specific type of unfaithfulness.
        """
        unfaithful_pairs: list[QuestionPairData] = []
        prop_ids_filter = self.get_available_prop_ids()

        if prop_id is not None:
            # Apply the prop_id filter
            assert prop_id in prop_ids_filter, f"Property ID {prop_id} not found in {prop_ids_filter}"
            prop_ids_filter = [prop_id]

        for (prop_id_group, comparison_group), group in self.df.groupby(
            ["prop_id", "comparison"]
        ):
            assert isinstance(prop_id_group, str), f"Expected prop_id to be a string, got {type(prop_id_group)}"
            assert isinstance(comparison_group, str), f"Expected comparison to be a string, got {type(comparison_group)}"

            if prop_id_group not in prop_ids_filter:
                print(f"Skipping prop_id {prop_id_group} because it is not in the filter")
                continue

            if comparison is not None and comparison_group != comparison:
                print(f"Skipping comparison {comparison_group} because it is not {comparison}")
                continue

            # Find pairs of questions with reversed x_name and y_name
            pairs = {}
            for _, row in group.iterrows():
                key = frozenset([row.x_name, row.y_name])
                if key not in pairs:
                    pairs[key] = []
                pairs[key].append(row)
            pairs = {k: v for k, v in pairs.items() if len(v) == 2}

            # Load unfaithfulness data
            faithfulness_dataset = None
            try:
                faithfulness_dataset = self._load_faithfulness_dataset(prop_id_group)
            except FileNotFoundError:
                # This is fine, means all questions in this prop_id are faithful
                # However, we are interested here in unfaithful pairs, so we skip this prop_id altogether
                continue
            
            # Load unfaithfulness pattern evaluation
            pattern_eval = self._load_unfaithfulness_pattern_eval(prop_id_group)
            assert pattern_eval is not None, f"Unfaithfulness pattern evaluation not found for prop_id {prop_id_group}"

            for pair in pairs.values():
                q1, q2 = pair

                # Assert if q1 is YES then q2 is NO and vice versa
                assert (q1.answer == "YES" and q2.answer == "NO") or (q1.answer == "NO" and q2.answer == "YES"), f"Question pair {q1.qid} and {q2.qid} are not faithful"

                # Check if this pair is in the unfaithfulness dataset
                is_unfaithful = q1.qid in faithfulness_dataset.questions_by_qid or q2.qid in faithfulness_dataset.questions_by_qid
                
                if not is_unfaithful:
                    continue

                # Get unfaithfulness patterns and analysis
                unfaithfulness_patterns_in_pair = []
                q1_unf_pattern_analysis = None
                q2_unf_pattern_analysis = None
                
                if pattern_eval is not None:
                    # Find which qid has the pattern analysis (could be either q1 or q2)
                    pattern_analysis = None
                    if q1.qid in pattern_eval.pattern_analysis_by_qid:
                        pattern_analysis = pattern_eval.pattern_analysis_by_qid[q1.qid]
                        q1_unf_pattern_analysis = pattern_analysis.q1_analysis
                        q2_unf_pattern_analysis = pattern_analysis.q2_analysis
                    elif q2.qid in pattern_eval.pattern_analysis_by_qid:
                        pattern_analysis = pattern_eval.pattern_analysis_by_qid[q2.qid]
                        # The q1 and q2 are swapped in the pattern analysis
                        q1_unf_pattern_analysis = pattern_analysis.q2_analysis
                        q2_unf_pattern_analysis = pattern_analysis.q1_analysis
                    
                    if pattern_analysis is not None:
                        if pattern_analysis.categorization_for_pair:
                            unfaithfulness_patterns_in_pair = pattern_analysis.categorization_for_pair

                # Apply unfaithfulness type filter if specified
                if unfaithfulness_type is not None:
                    if unfaithfulness_type not in unfaithfulness_patterns_in_pair:
                        continue

                # Load responses and evaluations
                q1_raw_responses = self._load_responses(q1.qid, q1.dataset_id, q1.temperature, q1.top_p, q1.max_new_tokens)
                q2_raw_responses = self._load_responses(q2.qid, q2.dataset_id, q2.temperature, q2.top_p, q2.max_new_tokens)
                q1_evals = self._load_evals(q1.qid, q1.dataset_id, q1.temperature, q1.top_p, q1.max_new_tokens)
                q2_evals = self._load_evals(q2.qid, q2.dataset_id, q2.temperature, q2.top_p, q2.max_new_tokens)

                # Load fsps if available
                q1_fsps = self._load_fsps(list(q1_raw_responses.keys()), q1.dataset_id, q1.temperature, q1.top_p, q1.max_new_tokens)
                q2_fsps = self._load_fsps(list(q2_raw_responses.keys()), q2.dataset_id, q2.temperature, q2.top_p, q2.max_new_tokens)

                q1_base_prompt = self.inst_template.format(question=q1.q_str)
                q2_base_prompt = self.inst_template.format(question=q2.q_str)

                # Gather evidence of unfaithfulness and answer flipping classification for each response in Q1 and Q2
                q1_evidence_of_unfaithfulness_by_resp_id = {}
                q1_answer_flipping_classification_by_resp_id = {}
                for resp_id, resp_str in q1_raw_responses.items():
                    if q1_unf_pattern_analysis is not None and resp_id in q1_unf_pattern_analysis.responses:
                        q1_evidence_of_unfaithfulness_by_resp_id[resp_id] = q1_unf_pattern_analysis.responses[resp_id].evidence_of_unfaithfulness
                    else:
                        q1_evidence_of_unfaithfulness_by_resp_id[resp_id] = None
                    if q1_unf_pattern_analysis is not None and resp_id in q1_unf_pattern_analysis.responses:
                        q1_answer_flipping_classification_by_resp_id[resp_id] = q1_unf_pattern_analysis.responses[resp_id].answer_flipping_classification
                    else:
                        q1_answer_flipping_classification_by_resp_id[resp_id] = None

                q2_evidence_of_unfaithfulness_by_resp_id = {}
                q2_answer_flipping_classification_by_resp_id = {}
                for resp_id, resp_str in q2_raw_responses.items():
                    if q2_unf_pattern_analysis is not None and resp_id in q2_unf_pattern_analysis.responses:
                        q2_evidence_of_unfaithfulness_by_resp_id[resp_id] = q2_unf_pattern_analysis.responses[resp_id].evidence_of_unfaithfulness
                    else:
                        q2_evidence_of_unfaithfulness_by_resp_id[resp_id] = None
                    if q2_unf_pattern_analysis is not None and resp_id in q2_unf_pattern_analysis.responses:
                        q2_answer_flipping_classification_by_resp_id[resp_id] = q2_unf_pattern_analysis.responses[resp_id].answer_flipping_classification
                    else:
                        q2_answer_flipping_classification_by_resp_id[resp_id] = None

                q1_responses = {resp_id: ResponseData(
                    response_id=resp_id,
                    response_str=resp_str,
                    is_correct=q1_evals[resp_id].result == q1.answer,
                    model_answer=q1_evals[resp_id].result if q1_evals[resp_id].result is not None else "UNKNOWN",  # type: ignore
                    prompt_str=q1_fsps[resp_id] + "\n\n" + q1_base_prompt if q1_fsps[resp_id] is not None else q1_base_prompt,
                    evidence_of_unfaithfulness=q1_evidence_of_unfaithfulness_by_resp_id[resp_id],
                    answer_flipping_classification=q1_answer_flipping_classification_by_resp_id[resp_id],
                ) for resp_id, resp_str in q1_raw_responses.items()}

                q2_responses = {resp_id: ResponseData(
                    response_id=resp_id,
                    response_str=resp_str,
                    is_correct=q2_evals[resp_id].result == q2.answer,
                    model_answer=q2_evals[resp_id].result if q2_evals[resp_id].result else "UNKNOWN", # type: ignore
                    prompt_str=q2_fsps[resp_id] + "\n\n" + q2_base_prompt if q2_fsps[resp_id] is not None else q2_base_prompt,
                    evidence_of_unfaithfulness=q2_evidence_of_unfaithfulness_by_resp_id[resp_id],
                    answer_flipping_classification=q2_answer_flipping_classification_by_resp_id[resp_id],
                ) for resp_id, resp_str in q2_raw_responses.items()}

                unfaithful_pairs.append(QuestionPairData(
                    q1=QuestionData(
                        qid=q1.qid,
                        q_str=q1.q_str,
                        dataset_id=q1.dataset_id,
                        correct_answer=q1.answer,
                        accuracy=q1.p_correct,
                        responses=q1_responses,
                    ),
                    q2=QuestionData(
                        qid=q2.qid,
                        q_str=q2.q_str,
                        dataset_id=q2.dataset_id,
                        correct_answer=q2.answer,
                        accuracy=q2.p_correct,
                        responses=q2_responses,
                    ),
                    is_unfaithful_pair=True,
                    unfaithfulness_patterns=unfaithfulness_patterns_in_pair,
                ))

        return unfaithful_pairs

    def _load_df(self) -> pd.DataFrame:
        """Loads and filters the main DataFrame."""
        if self.dataset_suffix is not None:
            df_file_name = f"df-wm-{self.dataset_suffix}.pkl"
        else:
            df_file_name = "df-wm.pkl"
        df_path = DATA_DIR / df_file_name
        if not df_path.exists():
            raise FileNotFoundError(f"DataFrame not found at {df_path}. You might need to generate it using scripts/iphr/make_df.py")
        
        df = pd.read_pickle(df_path)
        df = df[df["mode"] == "cot"]

        available_model_ids = df["model_id"].unique().tolist()
        assert self.model_id in available_model_ids, f"Model ID {self.model_id} not found in chainscope dataset. Available model IDs: {available_model_ids}"

        df = df[df["model_id"] == self.model_id]        
        if len(df) == 0:
            print(f"Warning: No data found for model '{self.model_id}' in {df_path}")

        return df

    def _load_faithfulness_dataset(self, prop_id: str) -> UnfaithfulnessPairsDataset:
        """Loads the UnfaithfulnessPairsDataset for a given property ID."""
        if self.dataset_suffix is not None:
            prop_id_with_suffix = f"{prop_id}_{self.dataset_suffix}"
        else:
            prop_id_with_suffix = prop_id

        if prop_id_with_suffix in self._faithfulness_cache:
            return self._faithfulness_cache[prop_id_with_suffix]
        
        file_path = DATA_DIR / "faithfulness" / self.model_name / f"{prop_id_with_suffix}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Faithfulness dataset not found at {file_path}")

        dataset = UnfaithfulnessPairsDataset.load_from_path(file_path)
        self._faithfulness_cache[prop_id_with_suffix] = dataset
        return dataset

    def _load_unfaithfulness_pattern_eval(self, prop_id: str) -> UnfaithfulnessPatternEval:
        """Loads the UnfaithfulnessPatternEval for a given property ID."""
        if self.dataset_suffix is not None:
            prop_id_with_suffix = f"{prop_id}_{self.dataset_suffix}"
        else:
            prop_id_with_suffix = prop_id

        cache_key = f"pattern_eval_{prop_id_with_suffix}"
        if cache_key in self._unfaithfulness_pattern_evals:
            return self._unfaithfulness_pattern_evals[cache_key]
        
        # Default sampling params for pattern evaluation
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=8000,
        )
        
        file_path = DATA_DIR / "unfaithfulness_pattern_eval" / sampling_params.id / prop_id_with_suffix / f"{self.model_name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Unfaithfulness pattern evaluation not found at {file_path}")

        pattern_eval = UnfaithfulnessPatternEval.load_from_path(file_path)
        self._unfaithfulness_pattern_evals[cache_key] = pattern_eval
        return pattern_eval

    def _load_responses(self, qid: str, dataset_id: str, temperature: float, top_p: float, max_new_tokens: int) -> dict[str, str]:
        """Loads the responses for a given question ID."""
        dataset_params = DatasetParams.from_id(dataset_id)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        cache_key = f"{self.model_id}_{dataset_id}_{sampling_params.id}"

        if cache_key not in self._responses_cache:
            self._responses_cache[cache_key] = CotResponses.load(
                DATA_DIR
                / "cot_responses"
                / self.instr_id
                / sampling_params.id
                / dataset_params.pre_id
                / dataset_params.id
                / f"{self.model_id_no_slash}.yaml"
            )
        cot_responses = self._responses_cache[cache_key]
        if qid not in cot_responses.responses_by_qid:
            raise ValueError(f"Question ID {qid} not found in responses for model {self.model_id} with sampling params {sampling_params}")
        response_str_by_resp_id = cot_responses.responses_by_qid[qid]
        assert all(isinstance(resp, str) for resp in response_str_by_resp_id.values()), "All responses should be strings"
        return response_str_by_resp_id  # type: ignore    

    def _load_fsps(self, response_ids: list[str], dataset_id: str, temperature: float, top_p: float, max_new_tokens: int) -> dict[str, str | None]:
        """Loads the fsps for a given question ID."""
        dataset_params = DatasetParams.from_id(dataset_id)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        cache_key = f"{self.model_id}_{dataset_id}_{sampling_params.id}"

        if cache_key not in self._responses_cache:
            self._responses_cache[cache_key] = CotResponses.load(
                DATA_DIR
                / "cot_responses"
                / self.instr_id
                / sampling_params.id
                / dataset_params.pre_id
                / dataset_params.id
                / f"{self.model_id_no_slash}.yaml"
            )
        cot_responses = self._responses_cache[cache_key]
        if cot_responses.fsp_by_resp_id is None:
            return {resp_id: None for resp_id in response_ids}

        return {resp_id: cot_responses.fsp_by_resp_id[resp_id] for resp_id in response_ids}

    def _load_evals(self, qid: str, dataset_id: str, temperature: float, top_p: float, max_new_tokens: int) -> dict[str, CotEvalResult]:
        """Loads the evals for a given question ID."""
        dataset_params = DatasetParams.from_id(dataset_id)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        cache_key = f"{self.model_id}_{dataset_id}_{sampling_params.id}"

        if cache_key not in self._evals_cache:
            self._evals_cache[cache_key] = dataset_params.load_cot_eval(
                self.instr_id,
                self.model_id,
                sampling_params,
            )
        cot_eval = self._evals_cache[cache_key]
        if qid not in cot_eval.results_by_qid:
            raise ValueError(f"Question ID {qid} not found in evals for model {self.model_id} with sampling params {sampling_params}")
        return cot_eval.results_by_qid[qid]

    def get_available_prop_ids(self) -> list[str]:
        """Returns a list of property IDs with faithfulness data for the given model."""
        return self.df["prop_id"].unique().tolist()