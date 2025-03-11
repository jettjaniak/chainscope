#!/usr/bin/env python3

"""Clean up AXIOM solutions to match their modified problems.

Example usage:
python3 -m dotenv run python3 scripts/clean_axiom_putnam_solutions.py \
    /home/arthur/Documents/COT/chainscope/chainscope/data/putnam2/axiom_putnam_alongside_ground_truth.yaml \
    /home/arthur/Documents/COT/chainscope/chainscope/data/putnam2/axiom_putnam_cleaned_by_claudefull.yaml \
    --or_model_ids "anthropic/claude-3.5-sonnet" \
    --verbose
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
import yaml

from chainscope.open_router_utils import ORBatchProcessor, ORRateLimiter


@dataclass
class PutnamProblemPair:
    problem_name: str
    axiom_statement: str
    axiom_solution: str
    original_statement: str
    original_solution: str


@dataclass
class CleaningResult:
    problem: PutnamProblemPair
    explanation: str
    cleaned_solution: str
    is_more_flawed: bool = False


def load_problem_pairs(
    yaml_path: Path, prefix: Optional[int] = None
) -> List[PutnamProblemPair]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    problems = []
    for item in data:
        problems.append(
            PutnamProblemPair(
                problem_name=item["problem_name"],
                axiom_statement=item["AXIOM_informal_statement"],
                axiom_solution=item["AXIOM_informal_solution"],
                original_statement=item["original_informal_statement"],
                original_solution=item["original_informal_solution"],
            )
        )
        if prefix is not None and len(problems) >= prefix:
            logging.info(f"Loaded {len(problems)} problems, no more.")
            break
    return problems


_CLEANING_PROMPT = """I am cleaning up a dataset of new variations to Putnam problems. There is supposed to be small details changed in the problem and solution, such as different constants and different algebra. However, often only the problem is edited, not the solution. Edit AXIOM_informal_solution to make sense as a solution to AXIOM_informal_problem.

If the transformed AXIOM problem has more fundamental flaws (i.e. major details are changed such that the problem statement is not solvable and you cannot write a fixed solution), respond with "MORE_FLAWED_THAN_THAT:" followed by your explanation.

Original Problem:
{original_statement}

Original Solution:
{original_solution}

Modified AXIOM Problem:
{axiom_statement}

Current AXIOM Solution (may need editing):
{axiom_solution}

First explain what needs to be changed in the solution (or why it's fundamentally flawed), then either:
1. Provide the corrected solution starting with "CORRECTED_SOLUTION:" or
2. If fundamentally flawed, start with "MORE_FLAWED_THAN_THAT:". Make sure the solution matches the modified problem's details exactly."""


async def clean_solutions(
    problems: List[PutnamProblemPair],
    or_model_ids: List[str],
    max_retries: int,
    max_parallel: Optional[int],
) -> List[tuple[PutnamProblemPair, CleaningResult | None]]:
    def process_response(
        or_response: str, problem: PutnamProblemPair
    ) -> CleaningResult:
        # Check if response indicates more fundamental flaws
        if "MORE_FLAWED_THAN_THAT:" in or_response:
            parts = or_response.split("MORE_FLAWED_THAN_THAT:")
            explanation = parts[1].strip()
            return CleaningResult(
                problem=problem,
                explanation=explanation,
                cleaned_solution="MORE_FLAWED_THAN_THAT",
                is_more_flawed=True,
            )

        # Handle normal case
        parts = or_response.split("CORRECTED_SOLUTION:")
        if len(parts) != 2:
            raise ValueError(f"Malformed response for {problem.problem_name}")

        explanation = parts[0].strip()
        cleaned_solution = parts[1].strip()

        return CleaningResult(
            problem=problem,
            explanation=explanation,
            cleaned_solution=cleaned_solution,
            is_more_flawed=False,
        )

    or_rate_limiter = None
    if max_parallel is not None:
        or_rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    processor = ORBatchProcessor[PutnamProblemPair, CleaningResult](
        or_model_ids=or_model_ids,
        max_retries=max_retries,
        max_new_tokens=1000,
        temperature=0.0,
        process_response=process_response,
        or_rate_limiter=or_rate_limiter,
    )

    prompts = [
        _CLEANING_PROMPT.format(
            original_statement=problem.original_statement,
            original_solution=problem.original_solution,
            axiom_statement=problem.axiom_statement,
            axiom_solution=problem.axiom_solution,
        )
        for problem in problems
    ]

    return await processor.process_batch(
        items=list(zip(problems, prompts, strict=True))
    )


def save_results(
    results: List[tuple[PutnamProblemPair, CleaningResult | None]], output_path: Path
):
    output_data = []
    for problem, result in results:
        output_data.append(
            {
                "problem_name": problem.problem_name,
                "AXIOM_informal_statement": "MORE_FLAWED_THAN_THAT"
                if result and result.is_more_flawed
                else problem.axiom_statement,
                "AXIOM_informal_solution": problem.axiom_solution
                if result is None
                else result.cleaned_solution,
                "original_informal_statement": problem.original_statement,
                "original_informal_solution": problem.original_solution,
                "cleaning_explanation": "NA_RESULT_IS_NONE"
                if result is None
                else result.explanation,
                "is_more_flawed": False if result is None else result.is_more_flawed,
            }
        )

    with open(output_path, "w") as f:
        yaml.safe_dump(output_data, f, allow_unicode=True, sort_keys=False)


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.argument("output_yaml", type=click.Path())
@click.option(
    "--or_model_ids",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Comma-separated list of models for cleaning",
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
    help="Only process the first N problems",
)
def main(
    input_yaml: str,
    output_yaml: str,
    or_model_ids: str,
    max_retries: int,
    max_parallel: Optional[int],
    verbose: bool,
    prefix: Optional[int],
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    problems = load_problem_pairs(Path(input_yaml), prefix)
    logging.info(f"Loaded {len(problems)} problems to clean")

    results = asyncio.run(
        clean_solutions(
            problems=problems,
            or_model_ids=or_model_ids.split(","),
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )

    save_results(results, Path(output_yaml))
    logging.info(f"Saved cleaned results to {output_yaml}")


if __name__ == "__main__":
    main()
