import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

import dateutil.parser
from anthropic import Anthropic, AsyncAnthropic


@dataclass
class AnthropicLimits:
    requests_limit: int
    requests_remaining: int
    requests_reset: str
    tokens_limit: int
    tokens_remaining: int
    tokens_reset: str
    input_tokens_limit: int
    input_tokens_remaining: int
    input_tokens_reset: str
    output_tokens_limit: int
    output_tokens_remaining: int
    output_tokens_reset: str
    retry_after: float | None = None
    org_tpm_remaining: int = 80000  # Organization tokens per minute limit
    org_tpm_reset: str = ""  # When the org TPM limit resets


def parse_rfc3339_to_timestamp(rfc3339_str: str) -> float:
    """Convert RFC3339 datetime string to Unix timestamp."""
    dt = dateutil.parser.parse(rfc3339_str)
    return dt.timestamp()


def get_anthropic_limits() -> AnthropicLimits:
    """Extract rate limits from Anthropic API response headers."""
    client = Anthropic()
    response = client.messages.with_raw_response.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    return AnthropicLimits(
        requests_limit=int(
            response.headers.get("anthropic-ratelimit-requests-limit", 0)
        ),
        requests_remaining=int(
            response.headers.get("anthropic-ratelimit-requests-remaining", 0)
        ),
        requests_reset=response.headers.get("anthropic-ratelimit-requests-reset", ""),
        tokens_limit=int(response.headers.get("anthropic-ratelimit-tokens-limit", 0)),
        tokens_remaining=int(
            response.headers.get("anthropic-ratelimit-tokens-remaining", 0)
        ),
        tokens_reset=response.headers.get("anthropic-ratelimit-tokens-reset", ""),
        input_tokens_limit=int(
            response.headers.get("anthropic-ratelimit-input-tokens-limit", 0)
        ),
        input_tokens_remaining=int(
            response.headers.get("anthropic-ratelimit-input-tokens-remaining", 0)
        ),
        input_tokens_reset=response.headers.get(
            "anthropic-ratelimit-input-tokens-reset", ""
        ),
        output_tokens_limit=int(
            response.headers.get("anthropic-ratelimit-output-tokens-limit", 0)
        ),
        output_tokens_remaining=int(
            response.headers.get("anthropic-ratelimit-output-tokens-remaining", 0)
        ),
        output_tokens_reset=response.headers.get(
            "anthropic-ratelimit-output-tokens-reset", ""
        ),
        retry_after=float(response.headers.get("retry-after", 0))
        if "retry-after" in response.headers
        else None,
        org_tpm_remaining=int(
            response.headers.get("anthropic-ratelimit-org-tpm-remaining", 80000)
        ),
        org_tpm_reset=response.headers.get("anthropic-ratelimit-org-tpm-reset", ""),
    )


@dataclass
class ANRateLimiter:
    requests_per_interval: int
    tokens_per_interval: int
    interval_seconds: int
    input_tokens: float = field(init=False)
    output_tokens: float = field(init=False)
    requests: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)
    client: Anthropic = field(default_factory=Anthropic)
    org_tpm_limit: int = 80000
    org_tpm_usage: float = field(init=False)
    org_tpm_last_update: float = field(init=False)

    def __post_init__(self):
        self.input_tokens = self.tokens_per_interval
        self.output_tokens = self.tokens_per_interval
        self.requests = self.requests_per_interval
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        self.org_tpm_usage = 0
        self.org_tpm_last_update = time.time()

        logging.info(
            f"ANRateLimiter initialized with {self.requests_per_interval} requests, "
            f"{self.tokens_per_interval} tokens per {self.interval_seconds} seconds, "
            f"and org TPM limit of {self.org_tpm_limit}"
        )

    async def acquire(self, prompt: str, model: str):
        async with self._lock:
            # Everything else was too complicated to implement
            await asyncio.sleep(0.5)

    def update_token_usage(self, output_tokens: int):
        """Update output token count after receiving a response"""
        self.output_tokens = max(0, self.output_tokens - output_tokens)
        # Update org TPM usage with actual tokens used
        now = time.time()
        time_passed = now - self.org_tpm_last_update
        self.org_tpm_usage = max(
            0,
            self.org_tpm_usage * (1 - time_passed / 60),  # Decay over 1 minute
        )
        self.org_tpm_usage += output_tokens
        self.org_tpm_last_update = now

    async def acquire_with_backoff(self, prompt: str, model: str, max_retries: int = 3):
        """Acquire rate limit with exponential backoff retry logic"""
        import random  # Add at top of file if not already present

        for attempt in range(max_retries):
            try:
                await self.acquire(prompt, model)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2**attempt) + random.uniform(0, 1)
                logging.warning(
                    f"Rate limit acquisition failed, retrying in {wait_time:.2f}s: {str(e)}"
                )
                await asyncio.sleep(wait_time)


async def generate_an_response_async(
    prompt: str,
    an_model_ids: list[str],
    client: AsyncAnthropic,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[[str], Any | None],
    rate_limiter: ANRateLimiter | None = None,
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
                        f"Retry attempt {attempt} of {max_retries} for generating a response"
                    )

                if rate_limiter:
                    await rate_limiter.acquire_with_backoff(prompt, an_model_id)

                # Use acreate instead of create for async operation
                an_response = await client.messages.create(
                    model=an_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )

                if rate_limiter:
                    rate_limiter.update_token_usage(an_response.usage.output_tokens)

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
                if attempt == max_retries:
                    logging.warning(
                        f"Failed to process response after {max_retries} retries "
                        f"for model {an_model_id}: {str(e)}"
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
        if an_rate_limiter:
            self.an_rate_limiter = an_rate_limiter
        else:
            limits = get_anthropic_limits()
            self.an_rate_limiter = ANRateLimiter(
                requests_per_interval=limits.requests_limit,
                tokens_per_interval=limits.tokens_limit,
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
                rate_limiter=self.an_rate_limiter,
            )
            return (item, result)

        tasks = [process_single(item, prompt) for item, prompt in items]
        return await asyncio.gather(*tasks)
