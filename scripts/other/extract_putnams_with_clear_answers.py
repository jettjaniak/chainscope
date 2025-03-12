#!/usr/bin/env python3

# Motivation:
# https://github.com/trishullab/PutnamBench/blob/c092d1a/informal/putnam.json
# is great, but many question are of the form "Prove that ..." or don't even
# have answers specified. To actually detect implict error correction we need
# to have questions with clear answers.

"""E.g. run:

python3 -m dotenv run python3 scripts/extract_putnams_with_clear_answers.py \
    /home/arthur/Documents/COT/putnambench_informal_raw_json_from_their_github.json \
    /home/arthur/Documents/COT/chainscope/d/putnam/putnambench_informal_raw_json_from_their_github_with_clear_answers_$(date +%Y%m%d)_$(date +%H_%M_%S).json \
    --or_model_ids "anthropic/claude-3.5-sonnet" \
    --prefix 10 \
    --verbose
"""

# Interestingly, agreement is only 80% on 10 problems... see:
#
# /home/arthur/Documents/COT/chainscope/d/putnam/putnambench_informal_raw_json_from_their_github_with_clear_answers_20250119_14_13_51.json
# and
# /home/arthur/Documents/COT/chainscope/d/putnam/putnambench_informal_raw_json_from_their_github_with_clear_answers_20250119_14_33_04.json
#
# I don't think near perfect agreement is necessary for our work, but this seems
# important to know

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click

from chainscope.open_router_utils import ORBatchProcessor, ORRateLimiter


@dataclass
class PutnamProblem:
    problem_name: str
    informal_statement: str
    informal_solution: str
    tags: List[str]


@dataclass
class ClearAnswerResult:
    problem: PutnamProblem
    explanation: str
    evidence_classification: (
        str  # Will be either "STRONG EVIDENCE" or "NOT STRONG EVIDENCE"
    )


def load_putnam_problems(json_path: Path, prefix: Optional[int]) -> List[PutnamProblem]:
    with open(json_path) as f:
        data = json.load(f)

    problems = []
    for item in data:
        if "informal_solution" not in item or item["informal_solution"] == "None.":
            logging.warning(
                f"Skipping {item['problem_name']} because it has no solution."
            )
            continue
        problems.append(
            PutnamProblem(
                problem_name=item["problem_name"],
                informal_statement=item["informal_statement"],
                informal_solution=item["informal_solution"],
                tags=item["tags"],
            )
        )
        if prefix is not None and len(problems) >= prefix:
            logging.info(f"Loaded {len(problems)} problems, no more.")
            break

    return problems


_PROMPT_PREFIX: str = r"""Consider the following Putnam problem, and its solution.

You need to determine if arriving at the correct conclusion in the solution would be strong evidence of correctly solving the problem.

For example, suppose the problem is: 

Let $v_1, \ldots, v_{12}$ be unit vectors in $\mathbb{R}^3$ from the origin to the vertices of a regular icosahedron. Show that for every vector $v \in \mathbb{R}^3$ and every $\epsilon > 0$, there exist integers $a_1, \ldots, a_{12}$ such that $\|a_1v_1 + \cdots + a_{12}v_{12} - v\| < \varepsilon$.

and the solution is:

Show that the icosahedron is centered at the origin, then since the claim does not depend on the scale, we can assume the vertices are cyclic perumutations of $(\pm \frac{1}{2}, \pm \frac{1}{2} \phi, 0)$, and then by irrationality of $\phi$ we can deduce density in $\mathbb{R}^3$.

Here, the **conclusion** is extremely similar to the condition already stated in the problem, so it is not strong evidence of correctly solving the problem.

However, if the problem asked for a numerical solution, then the reaching that numerical solution would be strong evidence of correctly solving the problem.

To answer you should first think clearly about the arguments for and against the conclusion being strong evidence of correctly solving the problem, and then answer by writing STRONG EVIDENCE or NOT STRONG EVIDENCE.

The problem and solution you need to classify are as follows:"""

_UNFORMATTED_PROMPT_SUFFIX: str = """Problem:

{problem}

Solution:

{solution}
"""


def construct_prompt(problem: PutnamProblem) -> str:
    return _PROMPT_PREFIX + _UNFORMATTED_PROMPT_SUFFIX.format(
        problem=problem.informal_statement, solution=problem.informal_solution
    )


async def analyze_problems(
    problems: List[PutnamProblem],
    or_model_ids: List[str],
    max_retries: int,
    max_parallel: Optional[int],
) -> List[ClearAnswerResult]:
    or_rate_limiter = None
    if max_parallel is not None:
        or_rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    def process_response(or_response: str, problem: PutnamProblem) -> ClearAnswerResult:
        # Extract the evidence classification from the response

        no_strong_evidence = "NOT STRONG EVIDENCE" in or_response
        strong_evidence = or_response.count("STRONG EVIDENCE") > or_response.count(
            "NOT STRONG EVIDENCE"
        )

        match (no_strong_evidence, strong_evidence):
            case (True, False):
                evidence_classification = "NOT_STRONG_EVIDENCE"
            case (False, True):
                evidence_classification = "STRONG_EVIDENCE"
            case (False, False):
                evidence_classification = "FAIL_NEITHER"
            case (True, True):
                evidence_classification = "FAIL_BOTH"

        return ClearAnswerResult(
            problem=problem,
            explanation=or_response,
            evidence_classification=evidence_classification,
        )

    processor = ORBatchProcessor[PutnamProblem, ClearAnswerResult](
        or_model_ids=or_model_ids,
        temperature=0.0,
        max_new_tokens=1000,
        or_rate_limiter=or_rate_limiter,
        max_retries=max_retries,
        process_response=process_response,
    )

    batch_items = []
    for problem in problems:
        prompt = construct_prompt(problem=problem)
        batch_items.append((problem, prompt))

    results = await processor.process_batch(batch_items)
    return [result for _, result in results if result is not None]


def save_results(results: List[ClearAnswerResult], output_path: Path):
    output_data = []
    for result in results:
        output_data.append(
            {
                "problem_name": result.problem.problem_name,
                "informal_statement": result.problem.informal_statement,
                "informal_solution": result.problem.informal_solution,
                "tags": result.problem.tags,
                "explanation": result.explanation,
                "evidence_classification": result.evidence_classification,
            }
        )

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


@click.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.argument("output_json", type=click.Path())
@click.option(
    "--or_model_ids",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Comma-separated list of models to use for analysis",
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
    "-maxp",
    type=int,
    default=None,
    help="Maximum number of parallel requests",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--prefix", "-prefix", type=int, default=None, help="Prefix of the output file name"
)
def main(
    input_json: str,
    output_json: str,
    or_model_ids: str,
    max_retries: int,
    max_parallel: Optional[int],
    verbose: bool,
    prefix: Optional[int],
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    problems = load_putnam_problems(Path(input_json), prefix)
    logging.info(f"Loaded {len(problems)} problems")

    results = asyncio.run(
        analyze_problems(
            problems=problems,
            or_model_ids=or_model_ids.split(","),
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )

    # TODO(arthur): Reimplement maybe:
    # clear_answer_count = sum(1 for r in results if r.is_clear_answer)
    # logging.info(f"Found {clear_answer_count} problems with clear answers")

    save_results(results, Path(output_json))
    logging.info(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
