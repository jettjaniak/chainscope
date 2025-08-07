import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import openai
from openai.types.chat.chat_completion_message import Annotation
from tqdm.asyncio import tqdm

from chainscope.api_utils.batch_processor import (BatchItem, BatchProcessor,
                                                  BatchResult)
from chainscope.typing import *

# Hard limit of maximum requests per minute to prevent excessive API usage
MAX_OPEN_AI_REQUESTS_LIMIT = 100


@dataclass
class OARateLimiter:
    """A simple rate limiter that uses a semaphore to limit concurrent requests."""

    _semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(MAX_OPEN_AI_REQUESTS_LIMIT)
        logging.info(
            f"OARateLimiter initialized with limit of {MAX_OPEN_AI_REQUESTS_LIMIT} "
            "parallel requests"
        )

    async def acquire(self):
        await self._semaphore.acquire()

    def release(self):
        self._semaphore.release()


def generate_oa_web_search_response_sync(
    prompt: str,
    model_id: Literal["gpt-4o-search-preview", "gpt-4o-mini-search-preview"],
    max_new_tokens: int = 1000,
    search_context_size: Literal["low", "medium", "high"] = "high",
    user_location_country: str | None = None,
    user_location_city: str | None = None,
    user_location_region: str | None = None,
) -> tuple[str, list[Annotation]] | None:
    web_search_models = ["gpt-4o-search-preview", "gpt-4o-mini-search-preview"]
    assert model_id in web_search_models, f"Model {model_id} is not a web search model. Options are: {web_search_models}"

    # Set web search options
    web_search_options = {
        "search_context_size": search_context_size,
    }

    # Set user location for web search if provided
    user_location = None
    if user_location_country or user_location_city or user_location_region:
        user_location = {
            "type": "approximate",
            "approximate": {}
        }
        if user_location_country:
            user_location["approximate"]["country"] = user_location_country
        if user_location_city:
            user_location["approximate"]["city"] = user_location_city
        if user_location_region:
            user_location["approximate"]["region"] = user_location_region

        web_search_options["user_location"] = user_location
    
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            web_search_options=web_search_options,
        )

        if (
            not response
            or not response.choices
            or not response.choices[0].message.content
        ):
            return None

        return response.choices[0].message.content, response.choices[0].message.annotations
    except Exception as e:
        logging.warning(f"Error generating response: {str(e)}")
        return None
    finally:
        client.close()


def generate_oa_response_sync(
    prompt: str,
    model_id: str,
    temperature: float = 1.0,
    max_new_tokens: int = 1000,
) -> str | None:
    """Generate a synchronous response from an OpenAI model.

    Args:
        prompt: The prompt to run on the model
        model_id: The model ID to use
        temperature: Temperature parameter for generation (default: 1.0)
        max_new_tokens: Maximum number of new tokens to generate (default: 1000)

    Returns:
        The raw response content or None if the request failed
    """
    client = openai.OpenAI()
    try:
        # Handle different parameter names for token limits based on model
        if "o1" in model_id:
            # O1 models use max_completion_tokens instead of max_tokens
            token_param = "max_completion_tokens"
            # O1 only supports the default (1) value for temperature
            completion_temp = 1
        else:
            token_param = "max_tokens" 
            completion_temp = temperature

        completion_params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": completion_temp,
            token_param: max_new_tokens,
        }

        response = client.chat.completions.create(**completion_params)

        if (
            not response
            or not response.choices
            or not response.choices[0].message.content
        ):
            return None

        return response.choices[0].message.content

    except Exception as e:
        logging.warning(f"Error generating response: {str(e)}")
        return None
    finally:
        client.close()


async def generate_oa_response_async(
    prompt: str,
    oa_model_ids: list[str],
    client: openai.AsyncOpenAI,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[[str], Any | None],
    rate_limiter: OARateLimiter | None = None,
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
        rate_limiter: An optional rate limiter to control concurrent requests.

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"Running prompt:\n{prompt}")

    for oa_model_id in oa_model_ids:
        oa_model_id = oa_model_id.split("/")[-1]
        for attempt in range(max_retries):
            if rate_limiter:
                await rate_limiter.acquire()
            try:
                if attempt > 0:
                    logging.info(f"Retry attempt {attempt} of {max_retries}")

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
                if attempt == max_retries -1:
                    logging.warning(
                        f"Failed to process response after {max_retries} retries for model {oa_model_id}: {str(e)}"
                    )
                    return None
                logging.warning(
                    f"Error on attempt {attempt + 1} for model {oa_model_id}: {str(e)}, retrying..."
                )
                continue
            finally:
                if rate_limiter:
                    rate_limiter.release()

    return None


class OABatchProcessor(BatchProcessor[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        rate_limiter: OARateLimiter | None,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        max_new_tokens: int,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
        )

        assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
        self.client = openai.AsyncOpenAI()
        self.rate_limiter = rate_limiter

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        supported_models = [
            "gpt-3.5",
            "gpt-4",
            "o1",
            "chatgpt-4o-latest",
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

        if self.rate_limiter is None:
            self.rate_limiter = OARateLimiter()

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:
            assert self.max_new_tokens is not None
            result = await generate_oa_response_async(
                prompt=prompt,
                oa_model_ids=[self.model_id],
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(
                    response, item
                ),
                rate_limiter=self.rate_limiter
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


def submit_openai_batch(
    prompt_by_qrid: dict[QuestionResponseId, str],
    instr_id: str,
    ds_params: DatasetParams,
    evaluated_model_id: str,  # The model being evaluated/generating
    evaluated_sampling_params: SamplingParams,
    evaluator_model_id: str | None = None,  # The model doing the evaluation (if any)
    evaluator_sampling_params: SamplingParams | None = None,
) -> OpenAIBatchInfo:
    """Submit a batch of prompts to OpenAI's batch API.

    Args:
        prompts: List of tuples containing (item, prompt)
        model_id: ID of the model being evaluated/generating
        instr_id: Instruction ID
        ds_params: Dataset parameters
        sampling_params: Sampling parameters for the evaluator
        evaluator_model_id: ID of the model doing the evaluation (if any)
        evaluated_sampling_params: Sampling parameters for the evaluated model (if any)

    Returns:
        Information about the submitted batch
    """
    # For evaluation, evaluator_model_id is the one making API calls
    api_model_id = evaluator_model_id or evaluated_model_id
    api_model_id = api_model_id.split("/")[-1]
    api_sampling_params = evaluator_sampling_params or evaluated_sampling_params

    # Create custom ID mapping and format requests
    custom_id_map = {}
    batch_requests = []
    for q_resp_id, prompt in prompt_by_qrid.items():
        custom_id = f"{q_resp_id.qid}__{q_resp_id.uuid}"
        custom_id_map[custom_id] = q_resp_id

        batch_requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": api_model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": api_sampling_params.temperature,
                    "top_p": api_sampling_params.top_p,
                    "max_tokens": api_sampling_params.max_new_tokens,
                },
            }
        )

    # Create a temporary JSONL file with the batch requests
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for request in batch_requests:
            json.dump(request, f)
            f.write("\n")
        batch_input_path = f.name

    # Upload the batch input file
    client = openai.OpenAI()
    with open(batch_input_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    # Create the batch
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"Batch for {instr_id} - {evaluated_model_id}"},
    )

    # Clean up the temporary file
    os.unlink(batch_input_path)

    # Create batch info
    batch_info = OpenAIBatchInfo(
        batch_id=batch.id,
        instr_id=instr_id,
        ds_params=ds_params,
        created_at=datetime.now(timezone.utc).isoformat(),
        custom_id_map=custom_id_map,
        evaluated_model_id=evaluated_model_id,
        evaluated_sampling_params=evaluated_sampling_params,
        evaluator_model_id=evaluator_model_id,
        evaluator_sampling_params=evaluator_sampling_params,
        metadata={},
    )
    batch_info.save()
    return batch_info


def process_batch_results(
    batch_info: OpenAIBatchInfo,
) -> list[tuple[QuestionResponseId, str]]:
    """Process results from a completed batch.

    Args:
        batch_info: Information about the batch

    Returns:
        List of tuples containing (item, response)
        response is None if there was an error processing the response
    """
    client = openai.OpenAI()

    # Get batch status
    batch = client.batches.retrieve(batch_info.batch_id)
    if batch.status != "completed":
        raise ValueError(f"Batch not completed. Current status: {batch.status}")

    # Get the output file
    if not batch.output_file_id:
        raise ValueError("No output file available")

    # Download and process results
    file_response = client.files.content(batch.output_file_id)
    processed_results = []

    # Process each line of the output file
    for line in file_response.text.splitlines():
        result = json.loads(line)
        custom_id = result.get("custom_id")

        if custom_id and custom_id in batch_info.custom_id_map:
            q_resp_id = batch_info.custom_id_map[custom_id]

            # Check if there was an error
            if result.get("error"):
                logging.error(f"Error in request {custom_id}: {result['error']}")
                continue

            # Extract the response content
            response = result.get("response", {}).get("body", {})
            choices = response.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                content = choices[0]["message"]["content"]
                processed_results.append((q_resp_id, content))
            else:
                logging.error(f"Invalid content in request {custom_id}")

    # Check for errors file if present
    if batch.error_file_id:
        error_response = client.files.content(batch.error_file_id)
        for line in error_response.text.splitlines():
            error = json.loads(line)
            logging.error(f"Batch error: {error}")

    return processed_results


def cancel_batch(batch_id: str) -> None:
    """Cancel a pending OpenAI batch.

    Args:
        batch_id: The ID of the batch to cancel
    """
    client = openai.OpenAI()
    try:
        client.batches.cancel(batch_id)
    except Exception as e:
        logging.error(f"Error cancelling batch {batch_id}: {str(e)}")
        raise
    finally:
        client.close()
