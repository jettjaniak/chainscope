#!/usr/bin/env python3

from chainscope.api_utils.api_selector import APIPreferences, APISelector
from chainscope.api_utils.batch_processor import BatchItem
from chainscope.typing import *


async def get_responses_async(
    prompts: list[tuple[BatchItem, str]],  # List of (item, prompt) pairs
    model_id: str,
    sampling_params: SamplingParams,
    api: str,
    max_retries: int,
) -> list[tuple[BatchItem, str]]:
    """Process a batch of prompts using the specified API and model.

    Args:
        prompts: List of tuples containing (item, prompt)
        model_id: ID of the model to use
        sampling_params: Parameters for sampling from the model
        api: API to use (one of: "ant", "oai", "or", "ds")
        process_response: Function to process the response
        max_retries: Maximum number of retries for each request

    Returns:
        List of tuples containing (item, result)
    """
    api_preferences = APIPreferences(
        open_router=(api == "or"),
        open_ai=(api == "oai"),
        anthropic=(api == "ant"),
        deepseek=(api == "ds"),
    )

    def process_response(
        response: str | tuple[str | None, str | None], _: BatchItem
    ) -> str | None:
        if isinstance(response, tuple):
            reasoning = response[0] or ""
            answer = response[1] or ""
            if reasoning != "":
                return f"<think>{reasoning}</think>{answer}"
            else:
                return answer
        return response

    processor = APISelector[BatchItem, str](api_preferences).get_batch_processor(
        model_id=model_id,
        temperature=sampling_params.temperature,
        max_new_tokens=sampling_params.max_new_tokens,
        max_retries=max_retries,
        process_response=process_response,
    )

    ret = []
    for item, response in await processor.process_batch(prompts):
        if response is None:
            continue
        ret.append((item, response))

    return ret  # type: ignore
