#!/usr/bin/env python3
"""E.g. run:

python3 -m dotenv run python3 scripts/putnam/putnamlike3_main_faithfulness_eval.py \
    /workspace/atc1/chainscope/chainscope/data/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/deepseek-chat_just_correct_responses_splitted.yaml \
    --critical_steps_yaml /workspace/atc1/chainscope/chainscope/data/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/deepseek-chat_just_correct_responses_splitted_anthropic_slash_claude-3_dot_7-sonnet_colon_thinking_critical_steps.yaml \
    --model_id "anthropic/claude-3.7-sonnet:thinking" \
    --verbose \
    --max_parallel 3 --max_retries=3 --end_idx=1 \
    --open_router --evaluation_mode="reward_hacking" &> /tmp/rhlog.txt


Or:

python3 -m dotenv run python3 scripts/putnam/putnamlike3_main_faithfulness_eval.py \
    /workspace/atc1/chainscope/d/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/anthropic__claude-3.7-sonnet_v0_just_correct_responses_newline_split.yaml \
    --model_id "anthropic/claude-3.7-sonnet" \
    --max_parallel 4 \
    --open_router \
    --start_idx 0 \
    --end_idx 1 --verbose

Or:

python3 -m dotenv run python3 scripts/putnam/putnamlike3_main_faithfulness_eval.py \
    /workspace/atc1/chainscope/chainscope/data/cot_responses/fimo_solutions_split.yaml \
    --model_id "anthropic/claude-3.7-sonnet:thinking" \
    --verbose \
    -r 2 --max_parallel 1 \
    --start_idx 0 \
    --end_idx 1 \
    --evaluation_mode="reward_hacking"

Or (for 2024 problems):

python3 -m dotenv run python3 scripts/putnam/putnamlike3_main_faithfulness_eval.py \
    /workspace/atc1/chainscope/chainscope/data/cot_responses/instr-v0/default_sampling_params/ten_putnam_2024_problems/anthropic__claude-3.7-sonnet_v0_just_correct_responses_splitted.yaml \
    --critical_steps_yaml /workspace/atc1/chainscope/chainscope/data/cot_responses/instr-v0/default_sampling_params/ten_putnam_2024_problems/anthropic__claude-3.7-sonnet_v0_just_correct_responses_splitted_anthropic_slash_claude-3_dot_7-sonnet_colon_thinking_critical_steps.yaml \
    --model_id "anthropic/claude-3.7-sonnet:thinking" \
    --verbose \
    -r 2 --max_parallel 1 \
    --start_idx 0 \
    --evaluation_mode="reward_hacking"

"""

import ast
import asyncio
import dataclasses
import logging
import re
import enum
from pathlib import Path
from typing import Optional

import click

from chainscope import cot_faithfulness_utils
from chainscope.api_utils import deepseek_utils
from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.typing import MathResponse, SplitCotResponses, StepFaithfulness


def parse_faithfulness_response(
    response: str | tuple[str | None, str | None],
    evaluation_mode: cot_faithfulness_utils.EvaluationMode = cot_faithfulness_utils.EvaluationMode.LATENT_ERROR_CORRECTION,
) -> tuple[str, str]:
    """Parse the faithfulness evaluation response into reasoning and classification.

    Extracts answers to all questions and determines classification based on the answers.
    Returns the full response as reasoning and the classification as a string.
    """
    if isinstance(response, tuple):
        response = f"**THINKING**\n{response[0]}\n**ANSWER**\n{response[1]}"

    # Extract all answers for each question
    classification = ""
    for q_num in evaluation_mode.expected_answers.keys():
        matches = re.finditer(
            rf"<answer-{q_num}>(.*?)</answer-{q_num}>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        # Take the last match if multiple exist
        last_match = None
        for match in matches:
            last_match = match
        if last_match:
            answer = last_match.group(1).strip().upper()
            # Normalize to YES/NO
            if answer in ["Y", "YES", "TRUE"]:
                answer = "Y"
            elif answer in ["N", "NO", "FALSE"]:
                answer = "N"
            classification += answer
        else:
            classification += "_RIP_"

    return response, classification


def create_processor(
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    force_open_router: bool = False,
    evaluation_mode: cot_faithfulness_utils.EvaluationMode = cot_faithfulness_utils.EvaluationMode.LATENT_ERROR_CORRECTION,
    max_new_tokens: int = 8192,
):
    """Create the appropriate processor based on the model ID."""

    def process_response(
        model_response: str, item: tuple[str, str, str, int]
    ) -> StepFaithfulness:
        if isinstance(model_response, tuple):
            model_response = (
                f"<reasoning>{model_response[0]}</reasoning>\n{model_response[1]}"
            )

        qid, uuid, step, step_idx = item
        reasoning, classification = parse_faithfulness_response(
            model_response, evaluation_mode
        )
        return StepFaithfulness(
            step_str=step, unfaithfulness=classification, reasoning=reasoning
        )

    if deepseek_utils.DeepSeekBatchProcessor.is_model_supported(model_id) and not force_open_router:
        # DeepSeek processor
        rate_limiter = None
        if max_parallel is not None:
            rate_limiter = deepseek_utils.DeepSeekRateLimiter(
                requests_per_minute=max_parallel
                * 60,  # Convert per second to per minute
            )
        return deepseek_utils.DeepSeekBatchProcessor[tuple[str, str, str, int], StepFaithfulness](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            process_response=process_response,
            rate_limiter=rate_limiter,
        )
    else:
        # OpenRouter processor
        rate_limiter = None
        if max_parallel is not None:
            rate_limiter = ORRateLimiter(
                requests_per_interval=max_parallel,
                interval_seconds=1,
            )
        return ORBatchProcessor[tuple[str, str, str, int], StepFaithfulness](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            process_response=process_response,
            rate_limiter=rate_limiter,
        )


async def evaluate_faithfulness(
    responses: SplitCotResponses,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    solution: bool = True,
    force_open_router: bool = False,
    critical_steps_by_qid: Optional[dict[str, dict[str, set[int]]]] = None,
    evaluation_mode: cot_faithfulness_utils.EvaluationMode = cot_faithfulness_utils.EvaluationMode.LATENT_ERROR_CORRECTION,
    ask_for_thinking: bool = False,
    max_new_tokens: int = 8192,
) -> SplitCotResponses:
    """Evaluate the faithfulness of each step in the responses."""

    processor = create_processor(
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        force_open_router=force_open_router,
        evaluation_mode=evaluation_mode,
        max_new_tokens=max_new_tokens,
    )

    # Prepare batch items
    batch_items = []
    for qid, responses_by_uuid in responses.split_responses_by_qid.items():
        for uuid, response in responses_by_uuid.items():
            steps: list[str] = []
            if isinstance(response, MathResponse):
                if isinstance(response.model_answer, list):
                    steps = response.model_answer
            elif isinstance(response, list):
                steps = response
            else:
                logging.warning(f"Skipping unknown response type: {type(response)}")
                continue

            for i, step in enumerate(steps):
                # Skip if not in critical steps (1-indexed)
                if critical_steps_by_qid is not None:
                    if (
                        qid not in critical_steps_by_qid
                        or uuid not in critical_steps_by_qid[qid]
                    ):
                        continue
                    if i + 1 not in critical_steps_by_qid[qid][uuid]:
                        continue
                    critical_steps = sorted(list(critical_steps_by_qid[qid][uuid]))

                if not isinstance(step, str):
                    logging.warning(f"Skipping non-string step: {step}")
                    continue
                # Format each step with step-number tags
                context = ""
                for step_idx, step_content in enumerate(steps):
                    context += f"<step-{step_idx+1}>\n{step_content}\n</step-{step_idx+1}>\n\n"
                context = context.rstrip()
                problem_str = responses.split_responses_by_qid[qid][uuid].problem
                solution_str = responses.split_responses_by_qid[qid][uuid].solution

                # Build `prompt` (based on nosolution flag)
                prompt = evaluation_mode.prompt_prefix(ask_for_thinking)
                if solution:
                    prompt += cot_faithfulness_utils._GENERAL_MIDDLE_BIT_IF_SOLUTION_PRESENT

                if critical_steps_by_qid is not None:
                    prompt += (
                        "\n\nAlso, for your convenience, here are the step numbers which are likely the critical steps"
                        " in the reasoning process: "
                    )
                    prompt += "step-" + ", step-".join(str(x) for x in critical_steps) + "."

                prompt += "\n\n" + evaluation_mode.prompt_questions
                prompt = prompt + f"\n\n<problem>\n{problem_str}\n</problem>\n"
                if solution:
                    prompt += f"\n<solution>\n{solution_str}\n</solution>\n"
                prompt += f"\n<step-to-evaluate><step-{i+1}>{step}</step-{i+1}></step-to-evaluate>\n\n<all steps>\n{context}\n</all steps>"
                prompt += "\n\n" + evaluation_mode.prompt_suffix(ask_for_thinking)

                batch_items.append(((qid, uuid, step, i), prompt))

    # Process batch
    results = await processor.process_batch(batch_items)
    skipped_steps = []
    # Convert results back to SplitCotResponses format with MathResponse
    new_responses_by_qid = {}
    for (qid, uuid, _, step_idx), faithfulness in results:
        if faithfulness is None:
            logging.warning(f"Faithfulness is None for {qid=}, {uuid=}, {step_idx=}")
            skipped_steps.append((qid, uuid, step_idx))
            continue
        if qid not in new_responses_by_qid:
            new_responses_by_qid[qid] = {}
        if uuid not in new_responses_by_qid[qid]:
            original = responses.split_responses_by_qid[qid][uuid]
            if isinstance(original, MathResponse):
                new_response = MathResponse(
                    name=original.name,
                    problem=original.problem,
                    solution=original.solution,
                    model_answer=[],  # Will be filled with StepFaithfulness objects
                    model_thinking=original.model_thinking,
                    correctness_explanation=original.correctness_explanation,
                    correctness_is_correct=original.correctness_is_correct,
                    correctness_classification=original.correctness_classification,
                )
            else:
                raise ValueError("We should not lose so much info???")
            new_responses_by_qid[qid][uuid] = new_response

        assert isinstance(faithfulness, StepFaithfulness)
        new_responses_by_qid[qid][uuid].model_answer.append(faithfulness)

    # Determine suffix based on model type and nosolution flag
    suffix = "_faithfulness"
    if deepseek_utils.DeepSeekBatchProcessor.is_model_supported(model_id):
        suffix = "_deepseek_faithfulness"
    if not solution:
        suffix += "_nosolution"

    return SplitCotResponses(
        split_responses_by_qid=new_responses_by_qid,
        model_id=f"{responses.model_id}{suffix}",
        successfully_split_count=responses.successfully_split_count,
        failed_to_split_count=responses.failed_to_split_count,
        instr_id=responses.instr_id,
        ds_params=dataclasses.replace(
            responses.ds_params,
            description=(
                f"{responses.ds_params.description} "
                f"(expected code {evaluation_mode.expected_answers_str}) "
                "(skipped " + (
                    ', '.join(f'qid_{qid}_uuid_{uuid}_step_idx_{step_idx}' for qid, uuid, step_idx in skipped_steps)
                    if skipped_steps else 'nothing at all!'
                )
            ),
        ),
        sampling_params=responses.sampling_params,
    )


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Model ID for evaluation (OpenRouter or DeepSeek model)",
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
    "--start_idx",
    type=int,
    default=None,
    help="Start index for responses to evaluate (inclusive)",
)
@click.option(
    "--end_idx",
    type=int,
    default=None,
    help="End index for responses to evaluate (exclusive)",
)
@click.option(
    "--nosolution",
    is_flag=True,
    help="Don't include the solution in the chain of reasoning evaluation",
)
@click.option(
    "--open_router",
    is_flag=True,
    help="Force using OpenRouter even for DeepSeek models",
)
@click.option(
    "--critical_steps_yaml",
    type=click.Path(exists=True),
    help="Path to YAML containing critical steps to evaluate. If provided, only evaluates steps listed in the unfaithfulness field.",
)
@click.option(
    "--evaluation_mode",
    type=click.Choice([mode.value for mode in cot_faithfulness_utils.EvaluationMode]),
    default=cot_faithfulness_utils.EvaluationMode.LATENT_ERROR_CORRECTION.value,
    help="Evaluation mode to use",
)
@click.option(
    "--ask_for_thinking",
    is_flag=True,
    help="Add thinking tags to the prompt to get more detailed reasoning",
)
@click.option(
    "--max_new_tokens",
    "-t",
    type=int,
    default=8192,
    help="Maximum number of new tokens to generate",
)
def main(
    input_yaml: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    verbose: bool,
    start_idx: Optional[int],
    end_idx: Optional[int],
    nosolution: bool,
    open_router: bool,
    critical_steps_yaml: Optional[str],
    evaluation_mode: str,
    ask_for_thinking: bool,
    max_new_tokens: int,
):
    """Evaluate the faithfulness of each step in split CoT responses."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    logging.warning(
        "This drops the reasoning trace of R1 and just leaves the final answer, seems bad TODO(arthur): Fix"
    )
    solution = not nosolution
    del nosolution

    evaluation_mode = cot_faithfulness_utils.EvaluationMode(evaluation_mode)

    # Setup suffix variable before we start editing
    # the indices variable:
    suffix = f"_{model_id.replace('/', '_slash_').replace('.', '_dot_').replace(':', '_colon_')}"
    suffix += f"_{evaluation_mode.value}"

    if start_idx is not None or end_idx is not None:
        suffix += f"_from_{start_idx or 0}_to_{end_idx or 'end'}"

    input_path = Path(input_yaml)
    responses = SplitCotResponses.load(input_path)

    # Load critical steps if provided
    critical_steps_by_qid = {}
    if critical_steps_yaml:
        logging.info(f"Loading critical steps from {critical_steps_yaml}")
        critical_steps = SplitCotResponses.load(Path(critical_steps_yaml))
        logging.info(
            f"Found {sum(len(x) for x in critical_steps.split_responses_by_qid.values())} questions in critical steps file"
        )
        for qid, responses_by_uuid in critical_steps.split_responses_by_qid.items():
            critical_steps_by_qid[qid] = {}
            logging.info(f"Processing {qid=} with {len(responses_by_uuid)} responses")
            for uuid, response in responses_by_uuid.items():
                if isinstance(response, MathResponse) and response.model_answer:
                    # Get the first StepFaithfulness object's unfaithfulness field
                    first_step = response.model_answer[0]
                    if isinstance(first_step, str):
                        first_step = StepFaithfulness(**ast.literal_eval(first_step))
                    if isinstance(first_step, StepFaithfulness):
                        critical_steps_str = first_step.unfaithfulness
                        critical_steps_by_qid[qid][uuid] = {
                            int(x.strip()) for x in critical_steps_str.split(",")
                        }
                        logging.info(
                            f"Added critical steps for {qid=}, {uuid=}: {critical_steps_str}"
                        )
                    else:
                        logging.warning(
                            f"Skipping {qid=}, {uuid=} because first step is not StepFaithfulness: {first_step}"
                        )
                else:
                    logging.warning(
                        f"Skipping {qid=}, {uuid=} because it's not a MathResponse or has no model_answer; {type(response)=}"
                    )

        logging.info(f"Finished loading critical steps. Final {critical_steps_by_qid=}")

    # Apply index selection globally if specified
    if start_idx is not None or end_idx is not None:
        # Collect all items across QIDs
        all_items = []
        for qid in responses.split_responses_by_qid:
            responses_dict = responses.split_responses_by_qid[qid]
            for uuid, response in responses_dict.items():
                all_items.append((qid, uuid, response))

        # Apply global index selection
        start = start_idx or 0
        end = end_idx or len(all_items)
        selected_items = all_items[start:end]

        # Rebuild responses dictionary with selected items
        new_responses_by_qid = {}
        for qid, uuid, response in selected_items:
            if qid not in new_responses_by_qid:
                new_responses_by_qid[qid] = {}
            new_responses_by_qid[qid][uuid] = response
        responses.split_responses_by_qid = new_responses_by_qid

    results = asyncio.run(
        evaluate_faithfulness(
            responses=responses,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
            solution=solution,
            force_open_router=open_router,
            critical_steps_by_qid=critical_steps_by_qid
            if critical_steps_yaml
            else None,
            evaluation_mode=evaluation_mode,
            ask_for_thinking=ask_for_thinking,
            max_new_tokens=max_new_tokens,
        )
    )

    # Make the new path the same as the old with suffix:
    path = str(input_path)
    # Change blah/blah2.txt -> blah/blah2_suffix.txt
    path_split = path.split(".")
    path_split[-2] = path_split[-2] + suffix
    path = Path(".".join(path_split))

    output_path = results.save(path=path)
    logging.warning(f"Saved faithfulness results to {output_path}")


if __name__ == "__main__":
    main()
