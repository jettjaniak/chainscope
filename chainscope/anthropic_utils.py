import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from anthropic import AsyncAnthropic

from chainscope.typing import *


@dataclass
class AnthropicLimits:
    requests_per_interval: int
    tokens_per_interval: int
    interval_seconds: int


@dataclass
class ANRateLimiter:
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
                wait_time *= 1.1  # Add small buffer

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


async def generate_an_response_async(
    prompt: str,
    an_model_ids: list[str],
    client: AsyncAnthropic,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[[str], Any | None],
) -> Any | None:
    """Generate a response from an Anthropic model.

    Args:
        prompt: The prompt to run on the model
        an_model_ids: List of model IDs to call
        client: The Anthropic client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"Running prompt:\n{prompt}")

    for an_model_id in an_model_ids:
        an_model_id = an_model_id.split("/")[-1]
        model_aliases = {
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-latest",
            "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
            "claude-3-5-haiku": "claude-3-5-haiku-latest",
        }
        an_model_id = model_aliases.get(an_model_id, an_model_id)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info(
                        f"Retry attempt {attempt} of {max_retries} for splitting response"
                    )

                an_response = await client.messages.create(
                    model=an_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )

                if not an_response or not an_response.content:
                    continue

                result = get_result_from_response(an_response.content[0].text)
                if result is not None:
                    logging.info("Found valid result!")
                    return result

                logging.info(
                    f"Invalid result on attempt {attempt + 1} for model {an_model_id}, retrying..."
                )
                continue

            except Exception as e:
                if attempt == max_retries - 1:
                    logging.warning(
                        f"Failed to process response after {max_retries} retries for model {an_model_id}: {str(e)}"
                    )
                    return None
                logging.warning(
                    f"Error on attempt {attempt + 1} for model {an_model_id}: {str(e)}, retrying..."
                )
                continue

    return None


ANBatchItem = TypeVar("ANBatchItem")  # Type of the input item
ANBatchResult = TypeVar("ANBatchResult")  # Type of the result


class ANBatchProcessor(Generic[ANBatchItem, ANBatchResult]):
    def __init__(
        self,
        an_model_ids: list[str],
        temperature: float,
        an_rate_limiter: ANRateLimiter | None,
        max_retries: int,
        process_response: Callable[[str, ANBatchItem], ANBatchResult | None],
        max_new_tokens: int,
    ):
        self.an_model_ids = an_model_ids
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.process_response = process_response

        assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"
        self.client = AsyncAnthropic()

        # Default rate limits for Anthropic API (adjust these based on your tier)
        self.an_rate_limiter = an_rate_limiter or ANRateLimiter(
            requests_per_interval=50,  # Adjust based on your rate limits
            tokens_per_interval=100_000,  # Adjust based on your rate limits
            interval_seconds=60,
        )

    async def process_batch(
        self, items: list[tuple[ANBatchItem, str]]
    ) -> list[tuple[ANBatchItem, ANBatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """

        async def process_single(
            item: ANBatchItem, prompt: str
        ) -> tuple[ANBatchItem, ANBatchResult | None]:
            await self.an_rate_limiter.acquire()

            result = await generate_an_response_async(
                prompt=prompt,
                an_model_ids=self.an_model_ids,
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
