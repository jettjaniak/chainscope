import asyncio
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import openai
import requests
from tqdm.asyncio import tqdm

from chainscope.api_utils.anthropic_utils import (get_budget_tokens,
                                                  is_anthropic_thinking_model)
from chainscope.api_utils.batch_processor import (BatchItem, BatchProcessor,
                                                  BatchResult)
from chainscope.api_utils.deepseek_utils import is_deepseek_thinking_model

# Hard limit of maximum requests per minute to prevent excessive API usage
MAX_OPEN_ROUTER_REQUESTS_LIMIT = 50

# Hard limit of maximum requests per minute for models that require reasoning
MAX_OPEN_ROUTER_REQUESTS_LIMIT_THINKING_MODEL = 10

MAX_REQUEST_TIMEOUT = 60 * 10  # 10 minutes


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


def is_thinking_model(model_id: str) -> bool:
    is_google_thinking_model = "gemini" in model_id and "thinking" in model_id
    is_qwen_thinking_model = "qwen" in model_id and "qwq" in model_id
    return is_deepseek_thinking_model(model_id) \
        or is_anthropic_thinking_model(model_id) \
        or is_google_thinking_model \
        or is_qwen_thinking_model


def get_openrouter_limits(max_requests_limit: int) -> OpenRouterLimits:
    """Get rate limits and credits from OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to get OpenRouter limits: {response.text}")

        data = response.json()["data"]

        return OpenRouterLimits(
            credits=float(data.get("limit", 1) or 1),  # Default to 1 if unlimited
            requests_per_interval=min(
                data["rate_limit"]["requests"], max_requests_limit
            ),
            interval=data["rate_limit"]["interval"],
            is_free_tier=data["is_free_tier"],
        )
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.RequestException,
    ) as e:
        logging.warning(f"Connection error while getting OpenRouter limits: {str(e)}")
        # Return default conservative limits when we can't connect
        return OpenRouterLimits(
            credits=1.0,
            requests_per_interval=1,
            interval="60s",
            is_free_tier=True,
        )


async def generate_or_response_async(
    prompt: str,
    model_id: str,
    client: openai.AsyncOpenAI,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    request_timeout: int,
    get_result_from_response: Callable[
        [str | tuple[str | None, str | None]], Any | None
    ],
    increase_timeout_on_error: bool = True,
) -> Any | None:
    """Generate a response from an OpenRouter model.

    Args:
        prompt: The prompt to run on the model
        model_id: Model ID to call
        client: The OpenAI client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        request_timeout: Timeout for the request (in seconds)
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """

    # Hacky condition to add reasoning to DeepSeek and Anthropic thinking models:
    if is_thinking_model(model_id):
        extra_body = {
            "include_reasoning": True,
            "reasoning": {},
            # "provider": {
            #     "allow_fallbacks": False,
            #     "order": [
            #         "Fireworks",
            #         "Together",
            #     ],
            # },
        }

        if is_anthropic_thinking_model(model_id):
            thinking_budget_tokens = get_budget_tokens(model_id)
            extra_body["reasoning"] = {
                "max_tokens": thinking_budget_tokens,
            }
            max_new_tokens = max_new_tokens + thinking_budget_tokens
            # Remove the budget tokens suffix and add the thinking suffix
            model_id = model_id.split("_")[0] + ":thinking"

        if "qwen" in model_id:
            # increase the max tokens by 4000
            max_new_tokens = max_new_tokens + 4000
    else:
        extra_body = None

    logging.info(f"Running prompt:\n{prompt}")

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Retry attempt {attempt} of {max_retries}")

            try:
                if increase_timeout_on_error:
                    request_timeout = min(request_timeout * 2, MAX_REQUEST_TIMEOUT)

                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        extra_body=extra_body,
                    ),
                    timeout=request_timeout,
                )
            except asyncio.TimeoutError:
                logging.warning(
                    f"Request timed out after {request_timeout} seconds for model {model_id}"
                )
                continue

            if (error := getattr(response, "error", None)) and isinstance(error, dict):
                error_code = error.get("code")
                error_msg = error.get("message", "")

                if error_code == 429:
                    logging.warning(
                        f"OpenRouter free tier daily limit reached for model {model_id}: {error_msg}"
                    )
                else:
                    logging.warning(
                        f"OpenRouter error for model {model_id}: {error_msg}"
                    )

                continue

            if not response or not response.choices or len(response.choices) == 0:
                logging.warning(f"No response or empty response from model {model_id}")
                continue

            if hasattr(response.choices[0].message, "reasoning_content"):
                # Can't remember which models this helps for, but good to keep it
                response = (
                    response.choices[0].message.reasoning_content,
                    response.choices[0].message.content,
                )
            elif hasattr(response.choices[0].message, "reasoning"):
                # Format for DeepSeek R1 models on OpenRouter
                response = (
                    response.choices[0].message.reasoning,
                    response.choices[0].message.content,
                )
            else:
                response = response.choices[0].message.content

            result = get_result_from_response(response)
            if result is not None:
                logging.info("Found valid result!")
                return result

            logging.warning(
                f"Invalid result on attempt {attempt + 1} for model {model_id}, retrying..."
            )
            continue

        except (
            asyncio.TimeoutError,
            openai.APIConnectionError,
            openai.APIError,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
            Exception,
        ) as e:
            logging.warning(
                f"Error on attempt {attempt + 1} for model {model_id}: {str(e)}, retrying..."
            )
            logging.info(traceback.format_exc())
            # if "JSONDecodeError" in str(type(e)):
            #     # Access the raw response content from the internal httpx response
            #     logging.warning(f"Raw response:\n{e.doc}")

    logging.warning(
        f"Failed to process response after {max_retries} retries for model {model_id}"
    )
    return None


class ORBatchProcessor(BatchProcessor[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
        rate_limiter: ORRateLimiter | None,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
        )

        assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is not set"
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        self.client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1")
        self.rate_limiter = rate_limiter

        self.max_requests_limit = MAX_OPEN_ROUTER_REQUESTS_LIMIT
        if is_thinking_model(self.model_id):
            self.max_requests_limit = MAX_OPEN_ROUTER_REQUESTS_LIMIT_THINKING_MODEL

        self.request_timeout = 180  # 3 minutes

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        # OpenRouter supports practically all models
        return True

    async def process_batch(
        self, items: list[tuple[BatchItem, str]]
    ) -> list[tuple[BatchItem, BatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """
        if len(items) == 0:
            return []

        if self.rate_limiter is None:
            # If no rate limiter is provided, use the default limits in our account
            limits = get_openrouter_limits(self.max_requests_limit)
            logging.info(f"Using OpenRouter limits: {limits}")

            interval_seconds = int(limits.interval.replace("s", ""))
            self.rate_limiter = ORRateLimiter(
                requests_per_interval=limits.requests_per_interval,
                interval_seconds=interval_seconds,
            )

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:
            assert self.rate_limiter is not None
            await self.rate_limiter.acquire()

            result = await generate_or_response_async(
                prompt=prompt,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                request_timeout=self.request_timeout,
                get_result_from_response=lambda response: self.process_response(
                    response, item
                ),
            )
            return (item, result)

        try:
            tasks = [process_single(item, prompt) for item, prompt in items]
            return await tqdm.gather(*tasks)
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            raise e
        finally:
            await self.client.close()
