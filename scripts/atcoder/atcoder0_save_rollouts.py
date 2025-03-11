#!/usr/bin/env python3
"""E.g. run:

python3 -m dotenv run python3 scripts/atcoder/atcoder0_save_rollouts.py \
    chainscope/data/atcoder/atcoder_dataset.json \
    --model_id "qwen/qwq-32b-preview" \
    --max_retries=4 \
    --prefix=1 \
    --verbose
"""

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import click

from chainscope.api_utils.deepseek_utils import (
    DeepSeekBatchProcessor,
    DeepSeekRateLimiter,
)
from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.typing import (
    AtCoderContest,
    AtCoderDataset,
    AtCoderDatasetParams,
    AtCoderProblem,
    AtCoderQuestion,
    AtCoderResponse,
    AtCoderSample,
    CotResponses,
    DefaultSamplingParams,
)

_PROMPT_TEMPLATE = """Solve this AtCoder programming problem step by step. First analyze the problem and plan your solution, then implement a complete C++ solution.

{statement}

{constraints}

{io_format}

**Sample Cases**
{samples_text}
**End of Sample Cases**

Think through the solution step by step, explaining your reasoning. After you've worked out the approach, provide a complete C++ implementation that solves the problem.

Your final answer should be a complete, compilable C++ file wrapped in ```cpp and ``` that handles all the input/output requirements and constraints, e.g.

```cpp
#include <iostream>
#include <vector>

using namespace std;

// Imaginary problem solution: print the first N triangle numbers
int main() {{
    int N;
    cin >> N;

    vector<int> ts(N);
    for(int i = 0; i < N; i++) {{
        // The nth triangle number is n*(n+1)/2
        ts[i] = (i + 1) * (i + 2) / 2;
    }}

    for(int i = 0; i < N; i++) {{
        cout << ts[i] << endl;
    }}
    return 0;
}}
```
"""


def format_samples(samples: list[AtCoderSample]) -> str:
    """Format sample cases into readable text."""
    samples_text = []
    for i, sample in enumerate(samples, 1):
        samples_text.append(f"Sample {i}:")
        samples_text.append(f"Input:\n{sample.input}")
        samples_text.append(f"Output:\n{sample.output}")
        if sample.explanation:
            samples_text.append(f"Explanation: {sample.explanation}")
        samples_text.append("")
    return "\n".join(samples_text)


def load_atcoder_dataset(json_path: Path) -> list[AtCoderContest]:
    """Load AtCoder dataset from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    contests = []
    for contest_data in data["contests"]:
        # Convert samples to AtCoderSample objects
        problems = []
        for problem_data in contest_data["problems"]:
            # Ensure we have strings, not None
            samples = [
                AtCoderSample(
                    input=sample["input"] or "",
                    output=sample["output"] or "",
                    explanation=sample["explanation"],  # Can be None
                )
                for sample in problem_data["samples"]
            ]

            # Create AtCoderProblem with converted samples
            problem = AtCoderProblem(
                id=problem_data["id"],
                problem_letter=problem_data["problem_letter"],
                url=problem_data["url"],
                statement=problem_data["statement"] or "",
                constraints=problem_data["constraints"] or "",
                io_format=problem_data["io_format"] or "",
                samples=samples,
            )
            problems.append(problem)

        # Create AtCoderContest with converted problems
        contest = AtCoderContest(
            id=contest_data["id"],
            url=contest_data["url"],
            start_time=contest_data["start_time"],
            problems=problems,
        )
        contests.append(contest)

    return contests


def create_atcoder_dataset(contests: list[AtCoderContest]) -> AtCoderDataset:
    """Create an AtCoderDataset from AtCoder contests."""
    questions = []
    for contest in contests:
        for problem in contest.problems:
            # Combine all problem information into a single prompt
            prompt = _PROMPT_TEMPLATE.format(
                statement=problem.statement,
                constraints=problem.constraints,
                io_format=problem.io_format,
                samples_text=format_samples(problem.samples),
            )

            questions.append(
                AtCoderQuestion(
                    name=f"{contest.id}_{problem.id}",
                    problem=prompt,
                    cpp_solution=None,  # No solution available yet
                )
            )

    return AtCoderDataset(
        questions=questions,
        params=AtCoderDatasetParams(
            description="AtCoder Programming Problems",
            id="atcoder",
            pre_id=None,
        ),
    )


def create_processor(
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    force_open_router: bool = False,
    max_new_tokens: int = 32_000,
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
        return DeepSeekBatchProcessor[AtCoderQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
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
        return ORBatchProcessor[AtCoderQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
        )


async def generate_rollouts(
    dataset: AtCoderDataset,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    prefix: Optional[int] = None,
    force_open_router: bool = False,
    preamble: str = "",
    max_new_tokens: int = 32_000,
) -> CotResponses:
    """Generate rollouts for each problem in the dataset."""

    processor = create_processor(
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        force_open_router=force_open_router,
        max_new_tokens=max_new_tokens,
    )

    # Prepare questions for processing
    questions = dataset.questions[:prefix] if prefix else dataset.questions

    batch_items = [(q, f"{preamble}{q.problem}") for q in questions]

    # Process all questions in batch
    logging.info(f"Processing {len(batch_items)} problems")
    results = await processor.process_batch(batch_items)

    # Collect responses
    responses_by_qid = {}
    for question, (thinking, answer) in zip(questions, results):
        if answer is None:
            logging.warning(f"Skipping failed response for {question.name}")
            continue

        if isinstance(answer, tuple):
            thinking = answer[0]
            answer = answer[1]

        responses_by_qid[question.name] = {
            str(uuid.uuid4())[:8]: AtCoderResponse(
                name=question.name,
                problem=question.problem,
                cpp_solution=None,  # Will be extracted from answer later
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
@click.argument("input_json", type=click.Path(exists=True))
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
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--open_router",
    is_flag=True,
    help="Force using OpenRouter even for DeepSeek models",
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=32_000,
    help="Maximum number of tokens to generate in the response",
)
def main(
    input_json: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    prefix: Optional[int],
    verbose: bool,
    open_router: bool,
    max_new_tokens: int,
):
    """Generate rollouts for AtCoder problems using OpenRouter or DeepSeek models."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load and prepare dataset
    input_path = Path(input_json)
    contests = load_atcoder_dataset(input_path)
    dataset = create_atcoder_dataset(contests)

    # Generate rollouts
    results = asyncio.run(
        generate_rollouts(
            dataset=dataset,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
            prefix=prefix,
            force_open_router=open_router,
            max_new_tokens=max_new_tokens,
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
