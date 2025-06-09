import asyncio
import logging
from typing import Mapping, TypeVar

import chainscope.typing as ctyping
from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter

ResponseType = TypeVar("ResponseType", ctyping.MathResponse, ctyping.AtCoderResponse)


def _format_thinking_and_final_answer(thinking: str, final_answer: str) -> str:
    return f"**WORKING**:\n\n{thinking}\n\n**ANSWER**:\n{final_answer}"


def format_response_as_working_answer(
    response: str | ctyping.MathResponse | ctyping.AtCoderResponse,
) -> str:
    """Format a response into the **WORKING** and **ANSWER** format.

    Args:
        response: Either a string containing both working and answer, or a MathResponse/AtCoderResponse object

    Returns:
        A formatted string with **WORKING** and **ANSWER** sections
    """

    # Remove all \n and \r and \t:
    if isinstance(response, str):
        return response
    elif response.model_thinking is None and isinstance(response.model_answer, list):
        assert (
            len(response.model_answer) == 1
        ), f"Expected exactly one model answer, got {response.model_answer=}"
        assert isinstance(
            response.model_answer[0], (str, ctyping.StepFaithfulness)
        ), f"Expected model_answer to be a string or StepFaithfulness, got {response.model_answer=}"
        [model_answer] = response.model_answer
        if isinstance(model_answer, str):
            return model_answer
        else:
            return model_answer.step_str
    elif isinstance(response.model_thinking, str) and isinstance(
        response.model_answer, list
    ):
        assert (
            len(response.model_answer) == 1
        ), f"Expected exactly one model answer, got {response.model_answer=}"
        assert isinstance(
            response.model_answer[0], (str, ctyping.StepFaithfulness)
        ), f"Expected model_answer to be a string or StepFaithfulness, got {response.model_answer=}"
        model_thinking = response.model_thinking
        [model_answer] = response.model_answer
        assert isinstance(
            model_thinking, str
        ), f"Expected model_thinking to be a string, got {model_thinking=}"
        return _format_thinking_and_final_answer(model_thinking, model_answer)
    elif isinstance(response.model_thinking, list) and isinstance(
        response.model_answer, str
    ):
        assert (
            len(response.model_answer) == 1
        ), f"Expected exactly one model answer, got {response.model_answer=}"
        assert isinstance(
            response.model_answer[0], str
        ), f"Expected model_answer to be a string, got {response.model_answer=}"
        assert (
            len(response.model_thinking) == 1
        ), f"Expected exactly one model thinking, got {response.model_thinking=}"
        assert isinstance(
            response.model_thinking[0], str
        ), f"Expected model_thinking to be a list of strings, got {response.model_thinking=}"
        [model_answer], [model_thinking] = (
            response.model_answer,
            response.model_thinking,
        )
        return _format_thinking_and_final_answer(model_thinking, model_answer)
    else:
        raise ValueError(
            f"Unexpected model_thinking type: {type(response.model_thinking)=}"
            f" and model_answer type: {type(response.model_answer)=}"
        )


def check_steps_are_valid_split(
    original_response: str,
    steps: list[str],
) -> bool:
    """Check if the steps are a valid split of the original response.

    Args:
        original_response: The original CoT response
        steps: List of extracted reasoning steps

    Returns:
        True if steps are valid, False otherwise
    """
    step_str = "\n".join(steps)  # TODO(arthur): Maybe "" instead of \n?

    if len(step_str) > 1.1 * len(original_response) or len(step_str) < 0.9 * len(
        original_response
    ):
        logging.warning(
            f"Step string length {len(step_str)} is not within 10% of original response length {len(original_response)}"
        )
        return False

    return True


def parse_model_split_response(split_text: str) -> list[str]:
    """Parse the model split response into a list of steps."""
    # Extract sections between <section N> tags

    # Remove all ```markdown and ```
    split_text = split_text.replace("```markdown", "").replace("```", "")
    sections = []
    current_pos = 0

    # Find if there is any text before the first section
    first_section_start = split_text.find("<section")
    if first_section_start > 0:
        sections.append(split_text[:first_section_start].strip())

    while True:
        # Find the start of the next section
        start = split_text.find("<section", current_pos)
        if start == -1:
            break

        # Find the end of the section tag
        tag_end = split_text.find(">", start)
        if tag_end == -1:
            break

        # Find the start of the next section (if any)
        next_start = split_text.find("<section", tag_end)

        # Extract the section content
        if next_start == -1:
            # This is the last section
            section_text = split_text[tag_end + 1 :]
        else:
            section_text = split_text[tag_end + 1 : next_start]

        # Remove any closing section tags with or without numbers
        while True:
            close_tag_start = section_text.find("</section")
            if close_tag_start == -1:
                break
            close_tag_end = section_text.find(">", close_tag_start)
            if close_tag_end == -1:
                break
            section_text = (
                section_text[:close_tag_start] + " " + section_text[close_tag_end + 1 :]
            )

        # Remove leading `
        section_text = section_text.lstrip("`")

        # Remove trailing `
        section_text = section_text.rstrip("`")

        # Remove leading and trailing whitespace
        section_text = section_text.strip()

        if section_text:
            # Only add if it's not empty
            sections.append(section_text)

        current_pos = next_start if next_start != -1 else len(split_text)

    return sections


async def split_cot_responses_async(
    responses: ctyping.CotResponses,
    model_id: str,
    max_retries: int,
    max_parallel: int | None,
    max_new_tokens_override: int | None = None,
    prefix: int | None = None,
) -> ctyping.SplitCotResponses:
    """Async version of split_cot_responses with rate limiting and retries."""

    rate_limiter = None
    if max_parallel is not None:
        rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    def process_response(
        response: str | tuple[str | None, str | None], item: tuple[str, str]
    ) -> list[str] | None:
        qid, uuid = item

        # TODO(arthur): unfaithful-shortcuts had this stuff...
        # surprising that it's more rigorous and found there!

        [real_rollout] = responses.responses_by_qid[qid][uuid].model_answer

        if isinstance(response, tuple):
            or_response = response[-1] or ""
        else:
            or_response = response

        logging.info(f"Response: {or_response}")
        sections = parse_model_split_response(or_response)
        logging.info(f"Sections: {sections}")

        logging.info(f"Item: {item}")

        if len(sections) <= 2:
            raise ValueError("Not enough sections.")

        if not check_steps_are_valid_split(
            original_response=real_rollout,
            steps=sections,
        ):
            raise ValueError("Steps are not valid.")

        return sections

        # if isinstance(response, tuple):
        #     or_response = response[0] or ""
        # else:
        #     or_response = response
        # logging.info(f"OR response:\n{or_response}")
        # sections = parse_model_split_response(or_response)
        # return sections  # TODO(arthur): Re-enable checking

    default_max_new_tokens = (
        hasattr(responses.sampling_params, "max_new_tokens")
        and int(responses.sampling_params.max_new_tokens * 1.25)
        or int(10000 * 1.25)
    )

    processor = ORBatchProcessor[tuple[str, str], list[str]](
        model_id=model_id,
        temperature=0.0
        if isinstance(responses.sampling_params, ctyping.DefaultSamplingParams)
        else responses.sampling_params.temperature,
        max_new_tokens=max_new_tokens_override or default_max_new_tokens,
        rate_limiter=rate_limiter,
        max_retries=max_retries,
        process_response=process_response,
    )

    # Prepare batch items
    batch_items = []
    for qid, response_by_uuid in responses.responses_by_qid.items():
        for uuid, response in response_by_uuid.items():
            prompt = (
                "Below is a chain-of-thought reasoning. Insert section markers (<section 1>, <section 2>, etc.) "
                "at the start of each logical step in the reasoning, but do NOT modify the original text in any way except adding the markers. "
                "Each new section should represent a distinct step in the reasoning process. "
                "There should be at least 3 steps, and possibly far more than that. "
                "If there is any text before the first logical step, include it as part of the first section. "
                "Do NOT leave any text out of the sections. "
                "Preserve all original formatting, including any "
                "bullet points, whitespace, numbers, exact latex formatting, typos (do NOT correct them, keep the text identical), or other list markers in the text. "
                "If there are numbered steps in the reasoning, treat them as different sections. "
                "Make sure to use <section N> tags for each step in the reasoning. Here is the text to split:"
            )
            if "**WORKING**" in format_response_as_working_answer(response):
                prompt += "You MUST include the **WORKING**: header (along with all text in the prompt, verbatim)."

            prompt += "\n\n" f"{format_response_as_working_answer(response)}"
            batch_items.append(((qid, uuid), prompt))

    # Apply prefix limit if specified
    if prefix is not None:
        batch_items = batch_items[:prefix]

    # Process batch
    results = await processor.process_batch(batch_items)
    # Process results
    split_responses_by_qid: dict[str, dict[str, ResponseType]] = {}
    success_count = 0
    failure_count = 0

    dataset_type = None

    for (qid, uuid), split_response in results:
        if qid not in split_responses_by_qid:
            split_responses_by_qid[qid] = {}

        if split_response is None:
            logging.info(
                f"Unable to split CoT response for qid={qid}, uuid={uuid} after {max_retries} retries"
            )
            failure_count += 1
        else:
            logging.info(
                f"Split response for qid={qid}, uuid={uuid} into {len(split_response)} sections"
            )
            success_count += 1
            original_response = responses.responses_by_qid[qid][uuid]
            if isinstance(original_response, ctyping.MathResponse):
                split_responses_by_qid[qid][uuid] = ctyping.MathResponse(
                    model_answer=split_response,
                    model_thinking=None,
                    name=original_response.name,
                    problem=original_response.problem,
                    solution=original_response.solution,
                )
                assert dataset_type in [None, "math"]
                dataset_type = "math"
            elif isinstance(original_response, ctyping.AtCoderResponse):
                split_responses_by_qid[qid][uuid] = ctyping.AtCoderResponse(
                    model_answer=split_response,
                    model_thinking=None,
                    name=original_response.name,
                    problem=original_response.problem,
                    cpp_solution=original_response.cpp_solution,
                )
                assert dataset_type in [None, "atcoder"]
                dataset_type = "atcoder"
            else:
                raise ValueError(f"Unexpected response type: {type(original_response)}")

    logging.error(f"Success: {success_count}, Failure: {failure_count}")

    assert dataset_type is not None
    # Create a new SplitCotResponses with the appropriate type
    if dataset_type == "math":
        split_responses: Mapping[str, Mapping[str, ctyping.MathResponse]] = {
            qid: {
                uuid: resp
                for uuid, resp in split_responses_by_qid[qid].items()
                if isinstance(resp, ctyping.MathResponse)
            }
            for qid, responses in split_responses_by_qid.items()
        }
        logging.info(f"Got split responses: {split_responses}")
        return ctyping.SplitCotResponses(
            split_responses_by_qid=split_responses,
            model_id=responses.model_id,
            successfully_split_count=success_count,
            failed_to_split_count=failure_count,
            instr_id=responses.instr_id,
            ds_params=responses.ds_params,
            sampling_params=responses.sampling_params,
        )
    elif dataset_type == "atcoder":
        # Create a new type for AtCoder split responses
        split_responses: Mapping[str, Mapping[str, ctyping.AtCoderResponse]] = {
            qid: {
                uuid: resp
                for uuid, resp in split_responses_by_qid[qid].items()
                if isinstance(resp, ctyping.AtCoderResponse)
            }
            for qid, responses in split_responses_by_qid.items()
        }
        logging.info(f"Got split responses via path 2: {split_responses}")
        assert isinstance(responses.ds_params, ctyping.AtCoderDatasetParams)

        ds_params = responses.ds_params
        return ctyping.SplitCotResponses(
            split_responses_by_qid=split_responses,
            model_id=responses.model_id,
            successfully_split_count=success_count,
            failed_to_split_count=failure_count,
            instr_id=responses.instr_id,
            ds_params=ds_params,
            sampling_params=responses.sampling_params,
        )


def split_cot_responses(
    responses: ctyping.CotResponses,
    model_id: str,
    max_retries: int,
    max_parallel: int | None,
    max_new_tokens_override: int | None = None,
    prefix: int | None = None,
) -> ctyping.SplitCotResponses:
    """Synchronous wrapper for the async implementation."""
    return asyncio.run(
        split_cot_responses_async(
            responses=responses,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
            max_new_tokens_override=max_new_tokens_override,
            prefix=prefix,
        )
    )
