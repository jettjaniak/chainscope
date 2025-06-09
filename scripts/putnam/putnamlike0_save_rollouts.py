#!/usr/bin/env python3
"""E.g. run:

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_historical \
    --model_id "anthropic/claude-3.7-sonnet:thinking" \
    --open_router \
    --max_retries=1 \
    --prefix=1 \
    --verbose

Or:

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_historical \
    --model_id "qwen/qwen-2.5-72b-instruct" \
    --max_retries=3 \
    --verbose

Or (with temperature and 2024 Putnam problems):

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_2024 \
    --model_id "anthropic/claude-3.7-sonnet:thinking" \
    --open_router \
    --temperature=0.3 \
    --max_retries=1 \
    --prefix=1 \
    --epochs=2 \
    --verbose

Or (for the specific NeurIPS Sonnet non-thinking experiment):

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_neurips_sonnet_nonthinking \
    --model_id "anthropic/claude-3.7-sonnet" \
    --open_router \
    --epochs=2 \
    --max_retries=1 \
    --verbose

"""

import asyncio
import logging
import os
import uuid
from enum import StrEnum
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
from chainscope.api_utils import anthropic_utils  # import ANBatchProcessor
from chainscope.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathQsDataset,
    MathQuestion,
    MathResponse,
)


class DatasetType(StrEnum):
    PUTNAM_HISTORICAL = "putnam_historical"  # For the historical dataset
    PUTNAM_2024 = "putnam_2024"  # For 2024 problems
    PUTNAM_NEURIPS_SONNET_NONTHINKING = "putnam_neurips_sonnet_nonthinking" # For the specific NeurIPS experiment

    @property
    def dataset_id(self) -> str:
        """Get the dataset ID for this type."""
        match self:
            case DatasetType.PUTNAM_HISTORICAL:
                return "filtered_putnambench"
            case DatasetType.PUTNAM_2024:
                return "ten_putnam_2024_problems"
            case DatasetType.PUTNAM_NEURIPS_SONNET_NONTHINKING:
                return "putnam_neurips_sonnet_nonthinking_experiment"

    @property
    def description(self) -> str:
        """Get the dataset description for this type."""
        match self:
            case DatasetType.PUTNAM_HISTORICAL:
                return "Historical Putnam Competition Problems"
            case DatasetType.PUTNAM_2024:
                return "Putnam Competition Problems 2024"
            case DatasetType.PUTNAM_NEURIPS_SONNET_NONTHINKING:
                return "Putnam Problems from NeurIPS Sonnet Non-Thinking Experiment"

    @property
    def yaml_path(self) -> str:
        """Get the YAML file path for this dataset type."""
        match self:
            case DatasetType.PUTNAM_HISTORICAL:
                return "d/putnam2/minimal_fork_of_putnambench_with_clear_answers.yaml"
            case DatasetType.PUTNAM_2024:
                return "d/putnam2/ten_putnam_2024_problems.yaml"
            case DatasetType.PUTNAM_NEURIPS_SONNET_NONTHINKING:
                # This path should be relative to the workspace root if the script is run from there,
                # or it needs to be an absolute path or adjusted based on execution context.
                # For now, assuming it's relative to the workspace root as per user's notebook file.
                return "chainscope/data/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/putnam_neurips_experiment_claude_sonnet_nonthinking.yaml"


def load_putnam_results_as_df(yaml_path: Path) -> pd.DataFrame:
    """Load Putnam results from YAML into a pandas DataFrame."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def create_putnam_dataset(dataset_type: DatasetType) -> MathQsDataset:
    """Create a MathQsDataset based on the dataset type.
    
    Args:
        dataset_type: Type of dataset to create
        
    Returns:
        A MathQsDataset containing the problems for the specified type
    """
    # Load and convert to DataFrame
    df = load_putnam_results_as_df(Path(dataset_type.yaml_path))
    
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
            description=dataset_type.description,
            id=dataset_type.dataset_id,
            pre_id=None,
        ),
    )


def create_processor(
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    temperature: float = 0.0,
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

    if anthropic_utils.ANBatchProcessor.is_model_supported(model_id) and not force_open_router:
        # Anthropic processor
        logging.info(f"Using Anthropic model {model_id}")
        rate_limiter = None
        if max_parallel is not None:
            rate_limiter = ORRateLimiter(
                requests_per_interval=max_parallel,
                interval_seconds=1,
            )
        return anthropic_utils.ANBatchProcessor[MathQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            # If _32k budget then do 1.25* that many tokens etc:
            max_new_tokens=32_000 if "_" not in model_id else int(int(model_id.split("_")[-1][:-1]) * 1.25),
            temperature=temperature,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
        )
    elif DeepSeekBatchProcessor.is_model_supported(model_id) and not force_open_router:
        return DeepSeekBatchProcessor[MathQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=8_192,
            temperature=temperature,
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
            temperature=temperature,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
        )


async def generate_rollouts(
    dataset: MathQsDataset,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    temperature: float = 0.0,
    prefix: Optional[int] = None,
    force_open_router: bool = False,
    preamble: str = "",
    epochs: int = 1,
) -> CotResponses:
    """Generate rollouts for each problem in the dataset.
    
    Args:
        epochs: Number of times to process each problem. If > 1, will generate multiple responses per problem.
    """
    processor = create_processor(
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        temperature=temperature,
        force_open_router=force_open_router,
    )

    # Prepare questions for processing
    questions = dataset.questions[:prefix] if prefix else dataset.questions

    logging.warning("USING THINK STEP-BY-STEP PREFIX! ('preamble')")
    
    # Create batch items for all questions × epochs
    batch_items = []
    for _ in range(epochs):
        batch_items.extend([
            (
                q,
                f"{preamble}{q.problem}",
            )
            for q in questions
        ])
    
    # Process all questions in a single batch
    logging.info(f"Processing {len(batch_items)} problems")
    results = await processor.process_batch(batch_items)

    # Process all questions in batch
    responses_by_qid = {}
    
    # Group responses by question
    for batch_idx, ((question, _), (_, thinking_and_answer)) in enumerate(zip(batch_items, results)):
        if thinking_and_answer is None or thinking_and_answer[-1] is None:
            logging.warning(
                f"Skipping failed response for {question.name} {thinking_and_answer=}"
            )
            continue

        thinking, answer = thinking_and_answer
        
        # For multiple epochs, append attempt number to question name:
        if epochs > 1:
            # Vibe code slop but w/e:
            attempt_number = batch_idx // len(dataset.questions[:prefix] if prefix else dataset.questions) + 1
            question_name = f"{question.name}_attempt_{attempt_number}"
        else:
            question_name = question.name
        
        # Initialize dict for this question if it doesn't exist
        if question_name not in responses_by_qid:
            responses_by_qid[question_name] = {}
            
        # Add this response with a unique ID
        responses_by_qid[question_name][str(uuid.uuid4())[:8]] = MathResponse(
            name=question_name,
            problem=question.problem,
            solution=question.solution,
            model_thinking=thinking,
            model_answer=[answer],  # Unsplit
        )

    # Sort responses by question name after all are collected
    def sort_key(name: str) -> tuple:
        # Handle both formats: putnam_2024_a1 and putnam_2024_a1_attempt_1
        parts = name.split('_')
        if len(parts) >= 4:  # Has problem number
            year = int(parts[1])
            prob_type = parts[2][0]  # 'a' or 'b'
            prob_num = int(parts[2][1])
            attempt = int(parts[-1]) if len(parts) > 4 else 0
            return (year, prob_type, prob_num, attempt)
        return (0, '', 0, 0)  # Fallback for unexpected formats

    sorted_responses = dict(sorted(responses_by_qid.items(), key=lambda x: sort_key(x[0])))

    return CotResponses(
        responses_by_qid=sorted_responses,
        model_id=model_id,
        instr_id="instr-v0",
        ds_params=dataset.params,
        sampling_params=DefaultSamplingParams(),
    )


@click.command()
@click.option(
    "--dataset_type",
    "-d",
    type=click.Choice([t.value for t in DatasetType], case_sensitive=False),
    required=True,
    help="Type of dataset being processed",
)
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
    "--temperature",
    "-t",
    type=float,
    default=0.0,
    help="Sampling temperature for the model",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=1,
    help="Number of times to process each problem",
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
    dataset_type: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    temperature: float,
    epochs: int,
    prefix: Optional[int],
    verbose: bool,
    open_router: bool,
    preamble: str,
):
    """Generate rollouts for Putnam problems using OpenRouter or DeepSeek models."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Convert dataset type string to enum
    dataset_type_enum = DatasetType(dataset_type)

    # Create dataset directly based on type
    dataset = create_putnam_dataset(dataset_type_enum)

    # Generate rollouts
    results = asyncio.run(
        generate_rollouts(
            dataset=dataset,
            model_id=model_id,
            preamble=preamble,
            max_retries=max_retries,
            max_parallel=max_parallel,
            temperature=temperature,
            epochs=epochs,
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
