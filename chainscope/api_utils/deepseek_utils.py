import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import openai
from tqdm.asyncio import tqdm

from chainscope.api_utils.batch_processor import (BatchItem, BatchProcessor,
                                                  BatchResult)

# Hard limit of maximum requests per minute to prevent excessive API usage
MAX_DEEPSEEK_REQUESTS_LIMIT = 50


@dataclass
class DeepSeekRateLimiter:
    requests_per_minute: int
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self):
        self.tokens = self.requests_per_minute
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update

            # Replenish tokens based on time passed
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + (time_passed * self.requests_per_minute / 60),
            )

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * 60 / self.requests_per_minute
                logging.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1
            self.last_update = now


def is_deepseek_thinking_model(model_id: str) -> bool:
    return "reason" in model_id or ("r1" in model_id and "deepseek" in model_id)


async def generate_deepseek_response_async(
    prompt: str,
    model_id: str,
    client: openai.AsyncOpenAI,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[[str], Any | None],
) -> Any | None:
    """Generate a response from a DeepSeek model.

    Args:
        prompt: The prompt to run on the model
        deepseek_model_ids: List of model IDs to call
        client: The OpenAI client configured for DeepSeek
        temperature: Temperature parameter for generation (note: ignored for deepseek-reasoner)
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"Running prompt:\n{prompt}")
    model_id = model_id.split("/")[-1]

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Retry attempt {attempt} of {max_retries}")

            # Handle different parameter names based on model
            completion_params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(8192, max_new_tokens),
            }

            # Note: temperature is not supported for deepseek-reasoner
            if model_id != "deepseek-reasoner":
                completion_params["temperature"] = temperature

            deepseek_response = await client.chat.completions.create(
                **completion_params
            )

            if (
                not deepseek_response
                or not deepseek_response.choices
                or not deepseek_response.choices[0].message.content
            ):
                continue

            # For deepseek-reasoner, we get both reasoning_content and content
            if model_id == "deepseek-reasoner":
                result = get_result_from_response(
                    deepseek_response.choices[0].message.reasoning_content
                    + "\n"
                    + deepseek_response.choices[0].message.content
                )
            else:
                result = get_result_from_response(
                    deepseek_response.choices[0].message.content
                )

            if result is not None:
                logging.info("Found valid result!")
                return result

            logging.info(
                f"Invalid result on attempt {attempt + 1} for model {model_id}, retrying..."
            )
            continue

        except Exception as e:
            if attempt == max_retries - 1:
                logging.warning(
                    f"Failed to process response after {max_retries} retries for model {model_id}: {str(e)}"
                )
                return None
            logging.warning(
                f"Error on attempt {attempt + 1} for model {model_id}: {str(e)}, retrying..."
            )
            continue

    return None


class DeepSeekBatchProcessor(BatchProcessor[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        rate_limiter: DeepSeekRateLimiter | None = None,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
        )

        assert os.getenv("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY is not set"
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )

        self.rate_limiter = rate_limiter or DeepSeekRateLimiter(
            requests_per_minute=MAX_DEEPSEEK_REQUESTS_LIMIT
        )

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        supported_models = [
            "deepseek-chat",
            "deepseek-reasoner",
        ]
        return any(
            model_id.split("/")[-1].startswith(model) for model in supported_models
        )

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

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:
            await self.rate_limiter.acquire()

            result = await generate_deepseek_response_async(
                prompt=prompt,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
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
