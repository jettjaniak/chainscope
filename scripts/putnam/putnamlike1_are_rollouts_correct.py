#!/usr/bin/env python3

"""E.g. run:

python3 -m dotenv run python3 scripts/putnamlike1_are_rollouts_correct.py \
    d/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/google__gemini-exp-1206:free_v0_prefix_1.yaml \
    --model_id "anthropic/claude-3.5-sonnet" \
    --verbose \
    --prefix=1
"""

import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import List, Optional

import click
import yaml

from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathResponse,
)


def load_putnam_model_responses(
    yaml_path: Path, prefix: Optional[int] = None
) -> List[MathResponse]:
    """Load Putnam dataset from CotResponses YAML format."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    logging.info(f"Loaded YAML data with keys: {list(data.keys())}")

    # Extract unique questions from the responses
    questions: List[MathResponse] = []
    try:
        responses_by_qid = data["responses_by_qid"]
    except Exception as e:
        print(f"Error: {e}")
        print(f"Data: {data}")
        raise e
    logging.info(f"Found {len(responses_by_qid)} qids in responses_by_qid")

    for qid, responses_dict in responses_by_qid.items():
        logging.info(f"Processing qid {qid} with {len(responses_dict)} responses")
        for response_data in responses_dict.values():
            logging.info(f"Response data keys: {list(response_data.keys())}")
            questions.append(
                MathResponse(
                    name=qid,
                    problem=response_data["problem"],
                    solution=response_data["solution"],
                    model_answer=response_data["model_answer"],
                    model_thinking=response_data["model_thinking"],
                    correctness_explanation=None,
                    correctness_is_correct=None,
                    correctness_classification=None,
                )
            )
            break  # DO NOT SUBMIT: Why is this here?

    if prefix is not None:
        questions = questions[:prefix]

    logging.info(f"Loaded {len(questions)} questions total")
    return questions


def save_all_results(
    results: List[tuple[MathResponse, MathResponse | None]],
    model_id: str,
    path: str | Path,
    correct_only: bool = False,
    suffix: str = "",
) -> Path:
    """Save all evaluation results using CotResponses format."""
    responses: dict[str, dict[str, MathResponse]] = {"default_qid": {}}

    for question, response in results:
        if response is not None and (
            (not correct_only) or response.correctness_is_correct
        ):
            if (not correct_only) or response is None:
                responses["default_qid"][question.name] = response
            else:
                concise_response = dataclasses.replace(
                    response,
                    correctness_explanation=None,
                    correctness_classification=None,
                )
                responses["default_qid"][question.name] = concise_response

    ds_params = MathDatasetParams(
        description=f"Putnam Competition Problems with Evaluations (Suffix: {suffix})",
        id=f"putnambench_with_evaluations{suffix}",
        pre_id=None,
    )

    cot_responses = CotResponses(
        responses_by_qid=responses,
        model_id="evaluation",  # ???
        instr_id="evaluation",
        ds_params=ds_params,
        sampling_params=DefaultSamplingParams(),
    )
    # Make the new path the same as the old with suffox:
    suffix = "_just_correct_responses" if correct_only else "_verbose_correctness"
    path = str(path)
    # change blah/blah2.txt -> blah/blah2_suffix.txt
    path_split = path.split(".")
    path_split[-2] = path_split[-2] + suffix
    path = Path(".".join(path_split))
    return cot_responses.save(path=path)


_EVALUATION_PROMPT: str = """Determine whether the model's answer to this maths problem is the same as the reference final answer. Only evaluate the model's final answer, not the intermediate steps (it is fine if the intermediate steps are incorrect).

**Problem**:
{problem}

**Model's final answer**:
{model_answer}

**Reference answer**:
{solution}

Determine if the model's final answer is equivalent to the reference answer, it's fine if it is e.g. algebraically the same, though be aware if the question e.g. asks for a closed form, the model's answer also need to be in closed form. Remember, ignore incorrect or misleading or fallacious or confusing intermediate steps. Just compare the final answer of the model to the reference answer.

First explain your comparison, then conclude with either EQUIVALENT or NOT EQUIVALENT.
"""


async def evaluate_model_responses(
    model_responses: List[MathResponse],
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
) -> List[tuple[MathResponse, MathResponse | None]]:
    """Evaluate responses using OpenRouter API."""

    def process_or_response(
        or_response: str, model_response: MathResponse
    ) -> MathResponse:
        # Extract the classification from the response
        has_equivalent = or_response.count("EQUIVALENT") > or_response.count(
            "NOT EQUIVALENT"
        )
        has_not_equivalent = "NOT EQUIVALENT" in or_response

        match (has_equivalent, has_not_equivalent):
            case (True, False):
                classification = "EQUIVALENT"
                is_correct = True
            case (False, True):
                classification = "NOT_EQUIVALENT"
                is_correct = False
            case (False, False):
                classification = "NA_NEITHER"
                is_correct = False
            case (True, True):
                classification = "NA_BOTH"
                is_correct = False
            case _:
                raise ValueError(
                    f"Ambiguous classification in response for {model_response.name}"
                )

        if classification in ["NA_NEITHER", "NA_BOTH"]:
            logging.warning(
                f"Ambiguous classification '{classification}' in response for {model_response.name}"
            )

        return MathResponse(
            name=model_response.name,
            problem=model_response.problem,
            solution=model_response.solution,
            model_answer=model_response.model_answer,
            model_thinking=model_response.model_thinking,
            correctness_explanation=or_response,
            correctness_is_correct=is_correct,
            correctness_classification=classification,
        )

    or_rate_limiter = None
    if max_parallel is not None:
        or_rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    processor = ORBatchProcessor[MathResponse, MathResponse](
        model_id=model_id,
        max_retries=max_retries,
        max_new_tokens=1000,
        temperature=0.0,
        process_response=process_or_response,
        rate_limiter=or_rate_limiter,
    )

    prompts = [
        _EVALUATION_PROMPT.format(
            problem=model_response.problem,
            model_answer=model_response.model_answer,
            solution=model_response.solution,
        )
        for model_response in model_responses
    ]

    return await processor.process_batch(
        items=list(zip(model_responses, prompts, strict=True))
    )


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Models for evaluation",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=1,
    help="Maximum retries for failed requests",
)
@click.option(
    "--max_parallel",
    "-p",
    type=int,
    default=None,
    help="Maximum number of parallel requests",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--prefix",
    "-prefix",
    type=int,
    default=None,
    help="Only process the first N answers",
)
def main(
    input_yaml: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    verbose: bool,
    prefix: Optional[int],
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    input_path = Path(input_yaml)

    model_responses = load_putnam_model_responses(input_path, prefix)
    logging.info(f"Loaded {len(model_responses)} model_responses to evaluate")

    results = asyncio.run(
        evaluate_model_responses(
            model_responses=model_responses,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )

    path1 = save_all_results(
        results, model_id=model_id, correct_only=False, path=input_path
    )
    logging.info(f"Saved verbose results to {path1}")
    path2 = save_all_results(
        results, model_id=model_id, correct_only=True, path=input_path
    )
    logging.info(f"Saved correct-only results to {path2}")


if __name__ == "__main__":
    main()
