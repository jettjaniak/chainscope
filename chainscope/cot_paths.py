import asyncio
import logging
import re
import uuid

from chainscope.api_utils.api_selector import APIPreferences, APISelector
from chainscope.typing import *


def process_response(response: str) -> list[str]:
    """Split a CoT response into steps."""
    steps = []
    current_step = ""
    lines = response.strip().split("\n")

    for line in lines:
        if not line:
            continue

        # Check for various step number formats
        # TODO: Look for the specific step number that is next in the sequence
        step_patterns = [
            r"^\d+\.",  # "1."
            r"^Step \d+:",  # "Step 1:"
            r"^Step \d+\.",  # "Step 1."
            r"^\*\*\d+\.\*\*",  # "**1.**"
            r"^\*\*Step \d+:\*\*",  # "**Step 1:**"
            r"^\*\*Step \d+\.\*\*",  # "**Step 1.**"
            r"^\*\*Step \d+\*\*:",  # **Step 1**:
            r"^\*\*Step \d+\*\*\.",  # **Step 1**.
            r"^\*\*\d+\. ",  # "**1. "
            r"^\*\*Step \d+: ",  # "**Step 1: "
            r"^\*\*Step \d+\. ",  # "**Step 1. "
            r"^\*\*\d+: ",  # "**1: "
            r"^\*\*\d+:\*\*",  # "**1:**"
            r"^#_{1,4} Step \d+:",  # "# Step 1:", "## Step 1:", "### Step 1:", "#### Step 1:"
            r"^#_{1,4} Step \d+.",  # "# Step 1.", "## Step 1.", "### Step 1.", "#### Step 1."
        ]

        is_new_step = any(re.match(pattern, line) for pattern in step_patterns)

        if is_new_step:
            if current_step:
                steps.append(current_step)
            current_step = line
        else:
            if current_step:
                current_step += "\n" + line
            else:
                current_step = line

    if current_step:
        steps.append(current_step)

    steps_str = "\n".join(f" -> {step}" for step in steps)
    logging.info(f"Split `{response}`\n into {len(steps)} steps:\n{steps_str}")

    return steps


async def gen_cot_paths_async(
    model_id: str,
    problem_dataset_name: str,
    sampling_params: SamplingParams,
    n_paths: int,
    existing_paths: CoTPath | None,
    api_preferences: APIPreferences,
) -> CoTPath:
    """Generate CoT paths in parallel."""

    cot_instruction = """Write down your answer step by step, and number each step ("1.", "2.", etc.)."""

    def process_model_response(
        response: str | tuple[str | None, str | None], qid: str
    ) -> list[str]:
        """Process model response into steps."""
        if isinstance(response, tuple):
            response = response[1] or ""
        return process_response(response)

    assert api_preferences.selects_at_least_one_api(), "Must specify at least one API"
    processor = APISelector[str, list[str]](api_preferences).get_batch_processor(
        model_id=model_id,
        temperature=sampling_params.temperature,
        max_new_tokens=sampling_params.max_new_tokens,
        max_retries=2,
        process_response=process_model_response,
    )

    # Load the problem dataset
    problem_dataset = ProblemDataset.load(problem_dataset_name)

    # Prepare batch items
    batch_items = []
    for qid, problem in problem_dataset.problems_by_qid.items():
        # Skip if we already have enough paths for this problem
        if existing_paths and qid in existing_paths.cot_path_by_qid:
            existing_count = len(existing_paths.cot_path_by_qid[qid])
            if existing_count >= n_paths:
                continue
            paths_needed = n_paths - existing_count
        else:
            paths_needed = n_paths

        for _ in range(paths_needed):
            prompt = f"{problem.q_str}\n\n{cot_instruction}"
            batch_items.append((qid, prompt))

    # Process batch
    if len(batch_items) > 0:
        results = await processor.process_batch(batch_items)
    else:
        results = []

    # Process results
    cot_path_by_qid: dict[
        str, dict[str, dict[int, str]]
    ] = {}  # qid -> {response_uuid -> {step_number -> step_str}}

    # Initialize with existing paths if any
    if existing_paths:
        cot_path_by_qid = existing_paths.cot_path_by_qid.copy()

    for qid, steps_list in results:
        if qid not in cot_path_by_qid:
            cot_path_by_qid[qid] = {}

        if steps_list is None:
            logging.error(f"Unable to generate CoT paths for qid={qid}")
            continue

        response_uuid = str(uuid.uuid4())
        cot_path_by_qid[qid][response_uuid] = {}
        for step_number, step in enumerate(steps_list):
            cot_path_by_qid[qid][response_uuid][step_number] = step

    return CoTPath(
        cot_path_by_qid=cot_path_by_qid,
        model_id=model_id,
        problem_dataset_name=problem_dataset_name,
        sampling_params=sampling_params,
    )


def gen_cot_paths(
    model_id: str,
    problem_dataset_name: str,
    sampling_params: SamplingParams,
    n_paths: int,
    existing_paths: CoTPath | None,
    api_preferences: APIPreferences,
) -> CoTPath:
    """Generate CoT paths for a given model and problem dataset."""
    return asyncio.run(
        gen_cot_paths_async(
            model_id=model_id,
            problem_dataset_name=problem_dataset_name,
            sampling_params=sampling_params,
            n_paths=n_paths,
            existing_paths=existing_paths,
            api_preferences=api_preferences,
        )
    )
