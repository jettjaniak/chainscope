import asyncio
import logging

from chainscope.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.typing import *


def check_steps_are_valid_split(original_response: str, steps: list[str]) -> bool:
    """Check if the steps are a valid split of the original response.

    Args:
        original_response: The original CoT response
        steps: List of extracted reasoning steps

    Returns:
        True if steps are valid, False otherwise
    """
    # Normalize whitespace in original response
    normalized_response = " ".join(original_response.split()).strip()

    # Check if each normalized step appears in the normalized response
    for step in steps:
        normalized_step = " ".join(step.split()).strip()
        if normalized_step not in normalized_response:
            logging.warning(f"Step not found in original response: {step}")
            return False

    # Remove each normalized step from the normalized response
    remaining_text = normalized_response
    for step in steps:
        normalized_step = " ".join(step.split()).strip()
        remaining_text = remaining_text.replace(normalized_step, "")

    # Remove all whitespace
    remaining_text_no_spaces = "".join(remaining_text.split()).strip()

    if remaining_text_no_spaces:
        logging.warning(f"Text remains after removing all steps: {remaining_text}")
        return False

    return True


def remove_all_symbols(text: str) -> str:
    """Remove all symbols and special characters from text, keeping only alphanumeric characters.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text with only alphanumeric characters and spaces
    """
    return "".join(char for char in text if char.isalnum())


def parse_model_split_response(split_text: str) -> list[str]:
    """Parse the model split response into a list of steps."""
    # Extract sections between <section N> tags
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
    responses: CotResponses,
    or_model_ids: list[str],
    max_retries: int,
    max_parallel: int | None,
) -> SplitCotResponses:
    """Async version of split_cot_responses with rate limiting and retries."""

    or_rate_limiter = None
    if max_parallel is not None:
        or_rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    def process_response(or_response: str, item: tuple[str, str]) -> list[str] | None:
        qid, uuid = item
        logging.info(f"OR response:\n{or_response}")
        sections = parse_model_split_response(or_response)
        steps_str = "\n".join(sections)
        logging.info(f"Parsed sections:\n{steps_str}")
        original_response = responses.responses_by_qid[qid][uuid]
        return (
            sections
            if check_steps_are_valid_split(original_response, sections)
            else None
        )

    processor = ORBatchProcessor[tuple[str, str], list[str]](
        or_model_ids=or_model_ids,
        temperature=responses.sampling_params.temperature,
        max_new_tokens=int(responses.sampling_params.max_new_tokens * 1.25),
        or_rate_limiter=or_rate_limiter,
        max_retries=max_retries,
        process_response=process_response,
    )

    # Prepare batch items
    batch_items = []
    for qid, response_by_uuid in responses.responses_by_qid.items():
        for uuid, response in response_by_uuid.items():
            if remove_all_symbols(response.lower()) in ["", "yes", "no"]:
                continue

            prompt = (
                "Below is a chain-of-thought reasoning. Insert section markers (<section 1>, <section 2>, etc.) "
                "at the start of each logical step in the reasoning, but do NOT modify the original text in any way. "
                "Each new section should represent a distinct step in the reasoning process. "
                "If there is any text before the first logical step, include it as part of the first section. "
                "Do NOT leave any text out of the sections. "
                "Preserve all original formatting, including any "
                "bullet points, whitespace, numbers, or other list markers in the text. "
                "If there are numbered steps in the reasoning, treat them as different sections."
                "\n\n"
                f"`{response}`"
            )
            batch_items.append(((qid, uuid), prompt))

    # Process batch
    results = await processor.process_batch(batch_items)

    # Process results
    split_responses_by_qid: dict[str, dict[str, list[str]]] = {}
    success_count = 0
    failure_count = 0

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
            split_responses_by_qid[qid][uuid] = split_response

    logging.error(f"Success: {success_count}, Failure: {failure_count}")

    return SplitCotResponses(
        split_responses_by_qid=split_responses_by_qid,
        model_id=responses.model_id,
        or_model_ids=or_model_ids,
        successfully_split_count=success_count,
        failed_to_split_count=failure_count,
        instr_id=responses.instr_id,
        ds_params=responses.ds_params,
        sampling_params=responses.sampling_params,
    )


def split_cot_responses(
    responses: CotResponses,
    or_model_ids: list[str],
    max_retries: int,
    max_parallel: int | None,
) -> SplitCotResponses:
    """Synchronous wrapper for the async implementation."""
    return asyncio.run(
        split_cot_responses_async(
            responses=responses,
            or_model_ids=or_model_ids,
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )
