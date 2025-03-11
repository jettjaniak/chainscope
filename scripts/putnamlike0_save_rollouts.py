#!/usr/bin/env python3
"""E.g. run:

python3 -m dotenv run python3 scripts/putnamlike0_save_rollouts.py \
    d/putnam2/minimal_fork_of_putnambench_with_clear_answers.yaml \
    --model_id "google/gemini-exp-1206:free" \
    --max_retries=1 \
    --prefix=1 \
    --verbose

python3 -m dotenv run python3 scripts/putnamlike0_save_rollouts.py \
    d/putnam2/minimal_fork_of_putnambench_with_clear_answers.yaml \
    --model_id "qwen/qwen-2.5-72b-instruct" \
    --max_retries=3 \
    --verbose
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import yaml

from chainscope.api_utils.deepseek_utils import (
    DeepSeekBatchProcessor,
    DeepSeekRateLimiter,
)
from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathQsDataset,
    MathQuestion,
    MathResponse,
)


def load_putnam_results_as_df(yaml_path: Path) -> pd.DataFrame:
    """Load Putnam results from YAML into a pandas DataFrame."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def create_putnam_dataset(df: pd.DataFrame) -> MathQsDataset:
    """Create a MathQsDataset from a Putnam DataFrame."""
    # Sort problems by year and type
    df = df.sort_values(
        by="problem_name",
        key=lambda x: pd.Series(
            [
                # Extract year and problem type (e.g. 'a1', 'b2')
                (int(name.split("_")[1]), name.split("_")[2])
                for name in x
            ]
        ).map(
            lambda t: (
                {
                    "a1": 0,
                    "b1": 1,
                    "a2": 2,
                    "b2": 3,
                    "a3": 4,
                    "b3": 5,
                    "a4": 6,
                    "b4": 7,
                    "a5": 8,
                    "b5": 9,
                    "a6": 10,
                    "b6": 11,
                }[t[1]],
                -t[0],
            )
        ),
    )

    return MathQsDataset(
        questions=[
            MathQuestion(
                name=row["problem_name"],
                problem=row["informal_statement"],
                solution=row["informal_solution"],
            )
            for _, row in df.iterrows()
        ],
        params=MathDatasetParams(
            description="Putnam Competition Problems",
            id="filtered_putnambench",
            pre_id=None,
        ),
    )


def create_processor(
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    force_open_router: bool = False,
):
    """Create the appropriate processor based on the model ID."""

    def get_tuple_or_str_response(
        response: tuple[str, str] | str, other: Any
    ) -> tuple[str | None, str]:
        logging.info(f"Inner response: {response}")

        if isinstance(response, tuple):
            assert (
                len(response) == 2
            ), f"Expected tuple of length 2, got {len(response)}"
            return response
        else:
            return (None, response)

    if DeepSeekBatchProcessor.is_model_supported(model_id) and not force_open_router:
        # DeepSeek processor
        logging.info(f"Using DeepSeek model {model_id}")
        rate_limiter = None
        if max_parallel is not None:
            rate_limiter = DeepSeekRateLimiter(
                requests_per_minute=max_parallel
                * 60,  # Convert per second to per minute
            )
        return DeepSeekBatchProcessor[MathQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=8_192,
            temperature=0.0,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
            # NOTE: Only used when thinking is also returned
            format_thinking=lambda thinking,
            answer: f"**WORKING**: {thinking.lstrip()}\n\n**ANSWER**: {answer.lstrip()}",
        )
    else:
        # OpenRouter processor
        logging.info(f"Using OpenRouter model {model_id}")
        rate_limiter = None
        if max_parallel is not None:
            rate_limiter = ORRateLimiter(
                requests_per_interval=max_parallel,
                interval_seconds=1,
            )
        return ORBatchProcessor[MathQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=32_000,
            temperature=0.0,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
        )


async def generate_rollouts(
    dataset: MathQsDataset,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    prefix: Optional[int] = None,
    force_open_router: bool = False,
    preamble: str = "",
) -> CotResponses:
    """Generate rollouts for each problem in the dataset."""

    processor = create_processor(
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        force_open_router=force_open_router,
    )

    # Prepare questions for processing
    questions = dataset.questions[:prefix] if prefix else dataset.questions

    logging.warning("USING A THINK STEP-BY-STEP PREFIX!")
    batch_items = [
        (
            q,
            f"{preamble}{q.problem}",
        )
        for q in questions
    ]  # TODO(arthur): Fully verify we used this format for all

    # Process all questions in batch
    logging.info(f"Processing {len(batch_items)} problems")
    results = await processor.process_batch(batch_items)

    # Collect responses
    responses_by_qid = {}
    for question, (_, thinking_and_answer) in zip(questions, results):
        if thinking_and_answer is None or thinking_and_answer[-1] is None:
            logging.warning(
                f"Skipping failed response for {question.name} {thinking_and_answer=}"
            )
            continue

        thinking, answer = thinking_and_answer
        responses_by_qid[question.name] = {
            str(uuid.uuid4())[:8]: MathResponse(
                name=question.name,
                problem=question.problem,
                solution=question.solution,
                model_thinking=thinking,
                model_answer=[answer],  # Unsplit
            )
        }

    return CotResponses(
        responses_by_qid=responses_by_qid,
        model_id=model_id,
        instr_id="instr-v0",
        ds_params=dataset.params,
        sampling_params=DefaultSamplingParams(),
    )


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3-opus",
    help="Model ID for generating rollouts (OpenRouter or DeepSeek model)",
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
@click.option(
    "--prefix",
    "-prefix",
    type=int,
    default=None,
    help="Only process the first N problems",
)
@click.option(
    "--preamble",
    type=str,
    default="Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n",
    help="Preamble text to add before each problem",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--open_router",
    is_flag=True,
    help="Force using OpenRouter even for DeepSeek models",
)
def main(
    input_yaml: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    prefix: Optional[int],
    verbose: bool,
    open_router: bool,
    preamble: str,
):
    """Generate rollouts for Putnam problems using OpenRouter or DeepSeek models."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load and prepare dataset
    input_path = Path(input_yaml)
    df = load_putnam_results_as_df(input_path)
    dataset = create_putnam_dataset(df)

    # Generate rollouts
    results = asyncio.run(
        generate_rollouts(
            dataset=dataset,
            model_id=model_id,
            preamble=preamble,
            max_retries=max_retries,
            max_parallel=max_parallel,
            prefix=prefix,
            force_open_router=open_router,
        )
    )

    # Save results
    for i in range(0, 100):
        output_path = results.get_path(
            f"_v{i}" + (f"_prefix_{prefix}" if prefix else "")
        )
        if not os.path.exists(output_path):
            break

    saved_path = results.save(path=output_path)
    logging.info(f"Saved rollouts to {saved_path}")


if __name__ == "__main__":
    main()
