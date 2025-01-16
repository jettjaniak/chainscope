import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp
import openai

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


@dataclass
class OpenRouterLimits:
    credits: float
    requests_per_interval: int
    interval: str
    is_free_tier: bool


async def get_openrouter_limits() -> OpenRouterLimits:
    """Get rate limits and credits from OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"Failed to get OpenRouter limits: {await response.text()}"
                )

            data = (await response.json())["data"]
            return OpenRouterLimits(
                credits=float(data.get("limit", 1) or 1),  # Default to 1 if unlimited
                requests_per_interval=data["rate_limit"]["requests"],
                interval=data["rate_limit"]["interval"],
                is_free_tier=data["is_free_tier"],
            )


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


async def split_cot_response_async(
    response: str,
    split_model_ids: list[str],
    client: openai.AsyncOpenAI,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
) -> list[str] | None:
    """Async version of split_cot_response with retry logic.

    Args:
        response: The CoT response to split
        split_model_ids: List of model IDs to use for splitting
        client: The OpenAI client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model

    Returns:
        List of split sections or None if splitting failed after all retries
    """
    if remove_all_symbols(response.lower()) in ["", "yes", "no"]:
        # No point in wasting tokens on these
        return None

    logging.info(f"Splitting response:\n{response}")

    for model_id in split_model_ids:
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info(
                        f"Retry attempt {attempt} of {max_retries} for splitting response"
                    )

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

                split_response = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )

                if not split_response or not split_response.choices:
                    continue

                split_text = split_response.choices[0].message.content

                logging.info(f"Model {model_id} split text:\n{split_text}")

                sections = parse_model_split_response(split_text)
                steps_str = "\n".join(sections)
                logging.info(f"Parsed sections:\n{steps_str}")

                validation_result = check_steps_are_valid_split(response, sections)
                if validation_result:
                    logging.info("Found valid split!")
                    return sections

                logging.info(
                    f"Invalid split on attempt {attempt + 1} for model {model_id}, retrying..."
                )
                continue

            except Exception as e:
                if attempt == max_retries:
                    logging.info(
                        f"Failed to split CoT response after {max_retries} retries for model {model_id}: {str(e)}"
                    )
                    return None
                logging.info(
                    f"Error on attempt {attempt + 1} for model {model_id}: {str(e)}, retrying..."
                )
                continue

    return None


@dataclass
class RateLimiter:
    requests_per_interval: int
    interval_seconds: int
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self):
        self.tokens = self.requests_per_interval
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update

            # Replenish tokens based on time passed
            self.tokens = min(
                self.requests_per_interval,
                self.tokens
                + (time_passed * self.requests_per_interval / self.interval_seconds),
            )

            if self.tokens < 1:
                wait_time = (
                    (1 - self.tokens)
                    * self.interval_seconds
                    / self.requests_per_interval
                )
                logging.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1
            self.last_update = now


async def process_batch(
    batch: list[tuple[str, str, str]],
    split_model_ids: list[str],
    temperature: float,
    max_new_tokens: int,
    rate_limiter: RateLimiter,
    max_retries: int,
) -> list[tuple[str, str, list[str] | None]]:
    """Process a batch of responses in parallel with rate limiting and retries."""
    client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1")

    async def process_single(
        qid: str, uuid: str, response: str
    ) -> tuple[str, str, list[str] | None]:
        await rate_limiter.acquire()
        logging.info(f"Starting request for uuid={uuid}")
        result = await split_cot_response_async(
            response=response,
            split_model_ids=split_model_ids,
            client=client,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
        )
        logging.info(f"Completed request for uuid={uuid}")
        return (qid, uuid, result)

    tasks = [process_single(*item) for item in batch]
    return await asyncio.gather(*tasks)


async def split_cot_responses_async(
    responses: CotResponses,
    split_model_ids: list[str],
    max_retries: int,
    max_parallel: int | None,
) -> SplitCotResponses:
    """Async version of split_cot_responses with rate limiting and retries."""

    if max_parallel is None:
        limits = await get_openrouter_limits()
        logging.info(f"Using OpenRouter limits: {limits}")

        interval_seconds = int(limits.interval.replace("s", ""))
        rate_limiter = RateLimiter(
            requests_per_interval=limits.requests_per_interval,
            interval_seconds=interval_seconds,
        )
    else:
        rate_limiter = RateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    all_requests = [
        (qid, uuid, response)
        for qid, response_by_uuid in responses.responses_by_qid.items()
        for uuid, response in response_by_uuid.items()
    ]

    split_responses_by_qid: dict[str, dict[str, list[str]]] = {}
    success_count = 0
    failure_count = 0

    max_new_tokens = int(responses.sampling_params.max_new_tokens * 1.25)
    temperature = responses.sampling_params.temperature

    # Process all requests in a single batch with controlled concurrency
    results = await process_batch(
        batch=all_requests,
        split_model_ids=split_model_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        rate_limiter=rate_limiter,
        max_retries=max_retries,
    )

    # Process results
    for qid, uuid, split_response in results:
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
        split_model_ids=split_model_ids,
        successfully_split_count=success_count,
        failed_to_split_count=failure_count,
        instr_id=responses.instr_id,
        ds_params=responses.ds_params,
        sampling_params=responses.sampling_params,
    )


def split_cot_responses(
    responses: CotResponses,
    split_model_ids: list[str],
    max_retries: int,
    max_parallel: int | None,
) -> SplitCotResponses:
    """Synchronous wrapper for the async implementation."""
    return asyncio.run(
        split_cot_responses_async(
            responses=responses,
            split_model_ids=split_model_ids,
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )
