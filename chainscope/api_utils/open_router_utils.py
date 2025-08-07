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
MAX_OPEN_ROUTER_REQUESTS_LIMIT = 100

# Hard limit of maximum requests per minute for models that require reasoning
MAX_OPEN_ROUTER_REQUESTS_LIMIT_THINKING_MODEL = 10

MAX_REQUEST_TIMEOUT = 60 * 10  # 10 minutes


@dataclass
class ORRateLimiter:
    """A simple rate limiter that uses a semaphore to limit concurrent requests."""

    limit: int
    _semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.limit)
        logging.info(
            f"ORRateLimiter initialized with limit of {self.limit} parallel requests"
        )

    async def acquire(self):
        await self._semaphore.acquire()

    def release(self):
        self._semaphore.release()


def is_thinking_model(model_id: str) -> bool:
    is_google_thinking_model = "gemini" in model_id and "thinking" in model_id
    is_qwen_thinking_model = "qwen" in model_id and "qwq" in model_id
    return is_deepseek_thinking_model(model_id) \
        or is_anthropic_thinking_model(model_id) \
        or is_google_thinking_model \
        or is_qwen_thinking_model


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
    rate_limiter: ORRateLimiter | None = None,
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
        if "qwq" in str(model_id):
            extra_body["provider"] = {
                "allow_fallbacks": False,
                "order": [
                    "Nebius",
                    # "Together",
                ],
            }

        if is_anthropic_thinking_model(model_id):
            thinking_budget_tokens = get_budget_tokens(model_id, max_new_tokens)
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
        if rate_limiter:
            await rate_limiter.acquire()
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
        finally:
            if rate_limiter:
                rate_limiter.release()

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
            self.rate_limiter = ORRateLimiter(limit=self.max_requests_limit)

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:

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
                rate_limiter=self.rate_limiter,
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
