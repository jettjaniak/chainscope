import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

import openai
import requests


@dataclass
class OpenAILimits:
    requests_per_interval: int
    tokens_per_interval: int
    remaining_requests: int
    remaining_tokens: int
    requests_reset_seconds: float
    tokens_reset_seconds: float


@dataclass
class OARateLimiter:
    requests_per_interval: int
    interval_seconds: int
    tokens_per_interval: int
    tokens: float = field(init=False)
    request_tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self):
        self.tokens = self.tokens_per_interval
        self.request_tokens = self.requests_per_interval
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update

            # Add a minimum time check to prevent excessive updates
            if time_passed < 0.001:  # 1ms minimum
                time_passed = 0.001

            # Replenish tokens based on time passed
            self.tokens = min(
                self.tokens_per_interval,
                self.tokens
                + (time_passed * self.tokens_per_interval / self.interval_seconds),
            )
            self.request_tokens = min(
                self.requests_per_interval,
                self.request_tokens
                + (time_passed * self.requests_per_interval / self.interval_seconds),
            )

            # Calculate wait time if either token type is depleted
            if self.tokens < 1 or self.request_tokens < 1:
                tokens_wait = (
                    0
                    if self.tokens >= 1
                    else (
                        (1 - self.tokens)
                        * self.interval_seconds
                        / self.tokens_per_interval
                    )
                )
                requests_wait = (
                    0
                    if self.request_tokens >= 1
                    else (
                        (1 - self.request_tokens)
                        * self.interval_seconds
                        / self.requests_per_interval
                    )
                )
                wait_time = max(tokens_wait, requests_wait)

                # Add a small buffer to prevent edge cases
                wait_time *= 1.1

                logging.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)

                # Recalculate tokens after waiting
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.tokens_per_interval,
                    self.tokens
                    + (time_passed * self.tokens_per_interval / self.interval_seconds),
                )
                self.request_tokens = min(
                    self.requests_per_interval,
                    self.request_tokens
                    + (
                        time_passed * self.requests_per_interval / self.interval_seconds
                    ),
                )

            self.tokens = max(0, self.tokens - 1)
            self.request_tokens = max(0, self.request_tokens - 1)
            self.last_update = now

    async def acquire_with_backoff(self, max_retries=3):
        for attempt in range(max_retries):
            try:
                await self.acquire()
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2**attempt) + random.uniform(0, 1)
                logging.warning(
                    f"Rate limit acquisition failed, retrying in {wait_time:.2f}s: {str(e)}"
                )
                await asyncio.sleep(wait_time)


def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string like '1s', '6m0s', or '6ms' to seconds."""
    if not time_str:
        return 0.0

    total_seconds = 0.0
    current_num = ""

    i = 0
    while i < len(time_str):
        if time_str[i].isdigit():
            current_num += time_str[i]
            i += 1
        elif time_str[i : i + 2] == "ms":
            if current_num:
                total_seconds += float(current_num) / 1000
            current_num = ""
            i += 2
        elif time_str[i] == "m" and (i + 1 >= len(time_str) or time_str[i + 1] != "s"):
            if current_num:
                total_seconds += float(current_num) * 60
            current_num = ""
            i += 1
        elif time_str[i] == "s":
            if current_num:
                total_seconds += float(current_num)
            current_num = ""
            i += 1
        else:
            i += 1

    return total_seconds


def get_openai_limits() -> OpenAILimits:
    """Get rate limits from OpenAI API headers."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    # Make a minimal API call to get the headers
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )

    if response.status_code != 200:
        raise ValueError(f"Failed to get OpenAI limits: {response.text}")

    headers = response.headers
    logging.info(f"OpenAI headers: {headers}")

    return OpenAILimits(
        requests_per_interval=int(headers["x-ratelimit-limit-requests"]),
        tokens_per_interval=int(headers["x-ratelimit-limit-tokens"]),
        remaining_requests=int(headers["x-ratelimit-remaining-requests"]),
        remaining_tokens=int(headers["x-ratelimit-remaining-tokens"]),
        requests_reset_seconds=parse_time_to_seconds(
            headers["x-ratelimit-reset-requests"]
        ),
        tokens_reset_seconds=parse_time_to_seconds(headers["x-ratelimit-reset-tokens"]),
    )


async def generate_oa_response_async(
    prompt: str,
    oa_model_ids: list[str],
    client: openai.AsyncOpenAI,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[[str], Any | None],
) -> Any | None:
    """Generate a response from an OpenAI model.

    Args:
        prompt: The prompt to run on the model
        oa_model_ids: List of model IDs to call
        client: The OpenAI client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"Running prompt:\n{prompt}")

    for oa_model_id in oa_model_ids:
        oa_model_id = oa_model_id.split("/")[-1]
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info(
                        f"Retry attempt {attempt} of {max_retries} for splitting response"
                    )

                # Handle different parameter names for token limits based on model
                if "o1" in oa_model_id:
                    # O1 models use max_completion_tokens instead of max_tokens
                    token_param = "max_completion_tokens"
                    # O1 only supports the default (1) value for temperature
                    completion_temp = 1
                else:
                    token_param = "max_tokens"
                    completion_temp = temperature

                completion_params = {
                    "model": oa_model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": completion_temp,
                    token_param: max_new_tokens,
                }

                oa_response = await client.chat.completions.create(**completion_params)

                if (
                    not oa_response
                    or not oa_response.choices
                    or not oa_response.choices[0].message.content
                ):
                    continue

                result = get_result_from_response(
                    oa_response.choices[0].message.content
                )
                if result is not None:
                    logging.info("Found valid result!")
                    return result

                logging.info(
                    f"Invalid result on attempt {attempt + 1} for model {oa_model_id}, retrying..."
                )
                continue

            except Exception as e:
                if attempt == max_retries:
                    logging.info(
                        f"Failed to process response after {max_retries} retries for model {oa_model_id}: {str(e)}"
                    )
                    return None
                logging.info(
                    f"Error on attempt {attempt + 1} for model {oa_model_id}: {str(e)}, retrying..."
                )
                continue

    return None


OABatchItem = TypeVar("OABatchItem")  # Type of the input item
OABatchResult = TypeVar("OABatchResult")  # Type of the result


class OABatchProcessor(Generic[OABatchItem, OABatchResult]):
    def __init__(
        self,
        oa_model_ids: list[str],
        temperature: float,
        oa_rate_limiter: OARateLimiter | None,
        max_retries: int,
        process_response: Callable[[str, OABatchItem], OABatchResult | None],
        max_new_tokens: int | None = None,
        max_completion_tokens: int | None = None,
    ):
        self.oa_model_ids = oa_model_ids
        self.temperature = temperature
        self.max_retries = max_retries
        self.process_response = process_response

        # Handle both token limit parameters
        if max_new_tokens is not None and max_completion_tokens is not None:
            raise ValueError(
                "Specify either max_new_tokens or max_completion_tokens, not both"
            )
        self.max_new_tokens = max_new_tokens or max_completion_tokens
        if self.max_new_tokens is None:
            raise ValueError(
                "Must specify either max_new_tokens or max_completion_tokens"
            )

        assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
        self.client = openai.AsyncOpenAI()

        self.oa_rate_limiter = oa_rate_limiter
        if self.oa_rate_limiter is None:
            # If no rate limiter is provided, use the default limits in our account
            limits = get_openai_limits()
            logging.info(f"Using OpenAI limits: {limits}")

            # Use 60 seconds as the default interval
            interval_seconds = 60
            self.oa_rate_limiter = OARateLimiter(
                requests_per_interval=limits.requests_per_interval,
                tokens_per_interval=limits.tokens_per_interval,
                interval_seconds=interval_seconds,
            )

    async def process_batch(
        self, items: list[tuple[OABatchItem, str]]
    ) -> list[tuple[OABatchItem, OABatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """

        async def process_single(
            item: OABatchItem, prompt: str
        ) -> tuple[OABatchItem, OABatchResult | None]:
            await self.oa_rate_limiter.acquire_with_backoff()

            result = await generate_oa_response_async(
                prompt=prompt,
                oa_model_ids=self.oa_model_ids,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(
                    response, item
                ),
            )
            return (item, result)

        tasks = [process_single(item, prompt) for item, prompt in items]
        return await asyncio.gather(*tasks)
