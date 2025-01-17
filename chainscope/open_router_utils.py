import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

import openai
import requests


@dataclass
class OpenRouterLimits:
    credits: float
    requests_per_interval: int
    interval: str
    is_free_tier: bool


@dataclass
class ORRateLimiter:
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


def get_openrouter_limits() -> OpenRouterLimits:
    """Get rate limits and credits from OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    response = requests.get(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.status_code != 200:
        raise ValueError(f"Failed to get OpenRouter limits: {response.text}")

    data = response.json()["data"]
    return OpenRouterLimits(
        credits=float(data.get("limit", 1) or 1),  # Default to 1 if unlimited
        requests_per_interval=data["rate_limit"]["requests"],
        interval=data["rate_limit"]["interval"],
        is_free_tier=data["is_free_tier"],
    )


async def generate_or_response_async(
    prompt: str,
    or_model_ids: list[str],
    client: openai.AsyncOpenAI,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[[str], Any | None],
) -> Any | None:
    """Generate a response from an OpenRouter model.

    Args:
        prompt: The prompt to run on the model
        or_model_ids: List of model IDs to call
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

    for or_model_id in or_model_ids:
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info(
                        f"Retry attempt {attempt} of {max_retries} for splitting response"
                    )

                or_response = await client.chat.completions.create(
                    model=or_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )

                if (
                    not or_response
                    or not or_response.choices
                    or not or_response.choices[0].message.content
                ):
                    continue

                result = get_result_from_response(
                    or_response.choices[0].message.content
                )
                if result is not None:
                    logging.info("Found valid result!")
                    return result

                logging.info(
                    f"Invalid result on attempt {attempt + 1} for model {or_model_id}, retrying..."
                )
                continue

            except Exception as e:
                if attempt == max_retries:
                    logging.info(
                        f"Failed to process response after {max_retries} retries for model {or_model_id}: {str(e)}"
                    )
                    return None
                logging.info(
                    f"Error on attempt {attempt + 1} for model {or_model_id}: {str(e)}, retrying..."
                )
                continue

    return None


ORBatchItem = TypeVar("ORBatchItem")  # Type of the input item
ORBatchResult = TypeVar("ORBatchResult")  # Type of the result


class ORBatchProcessor(Generic[ORBatchItem, ORBatchResult]):
    def __init__(
        self,
        or_model_ids: list[str],
        temperature: float,
        max_new_tokens: int,
        or_rate_limiter: ORRateLimiter | None,
        max_retries: int,
        process_response: Callable[[str, ORBatchItem], ORBatchResult | None],
    ):
        self.or_model_ids = or_model_ids
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.process_response = process_response

        assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is not set"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
        self.client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1")

        self.or_rate_limiter = or_rate_limiter
        if self.or_rate_limiter is None:
            # If no rate limiter is provided, use the default limits in our account
            limits = get_openrouter_limits()
            logging.info(f"Using OpenRouter limits: {limits}")

            interval_seconds = int(limits.interval.replace("s", ""))
            self.or_rate_limiter = ORRateLimiter(
                requests_per_interval=limits.requests_per_interval,
                interval_seconds=interval_seconds,
            )

    async def process_batch(
        self, items: list[tuple[ORBatchItem, str]]
    ) -> list[tuple[ORBatchItem, ORBatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """

        async def process_single(
            item: ORBatchItem, prompt: str
        ) -> tuple[ORBatchItem, ORBatchResult | None]:
            await self.or_rate_limiter.acquire()

            result = await generate_or_response_async(
                prompt=prompt,
                or_model_ids=self.or_model_ids,
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
