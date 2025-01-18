import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import openai
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chainscope.anthropic_utils import ANBatchProcessor, ANRateLimiter
from chainscope.open_ai_utils import OABatchProcessor, OARateLimiter
from chainscope.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.questions import QsDataset
from chainscope.typing import *
from chainscope.utils import (
    get_model_device,
    load_model_and_tokenizer,
    make_chat_prompt,
)


def get_question_cot_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_str: str,
    cot_instruction: str,
    sampling_params: SamplingParams,
    n_responses: int,
) -> dict[str, str]:
    device = get_model_device(model)

    # Prepare input
    chat_input = make_chat_prompt(
        instruction=cot_instruction.format(question=question_str),
        tokenizer=tokenizer,
    )
    input_ids = tokenizer.encode(chat_input, add_special_tokens=False)

    # Generate responses
    outputs = model.generate(
        torch.tensor([input_ids]).to(device),
        max_new_tokens=sampling_params.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        num_return_sequences=n_responses,
        use_cache=True,
    )

    # Decode all responses
    responses = {}
    for output in outputs:
        output_str = tokenizer.decode(
            output[len(input_ids) :], skip_special_tokens=True
        )
        responses[str(uuid4())] = output_str

    return responses


def get_question_cot_responses_or(
    client: openai.OpenAI,
    question_str: str,
    cot_instruction: str,
    sampling_params: SamplingParams,
    n_responses: int,
    model_id: str,
) -> dict[str, str]:
    """OpenRouter version of get_question_cot_responses."""
    # Prepare messages - using dict type that OpenAI/OpenRouter accepts
    messages: list[Any] = [
        {"role": "user", "content": cot_instruction.format(question=question_str)}
    ]

    responses = {}
    while len(responses) < n_responses:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                n=1,
            )
            if response and response.choices:
                responses[str(uuid4())] = response.choices[0].message.content
            else:
                print(
                    f"WARNING: Empty response received for model {model_id} and question {question_str}"
                )
                print(response)
                time.sleep(1)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            # Wait for 5 seconds before retrying
            time.sleep(5)
            continue

    return responses


def get_all_cot_responses(
    model_id: str,
    dataset_id: str,
    instr_id: str,
    sampling_params: SamplingParams,
    n_responses: int,
    question_type: Literal["yes-no", "open-ended"],
    existing_responses: CotResponses | None = None,
) -> CotResponses:
    model, tokenizer = load_model_and_tokenizer(model_id)
    instructions = Instructions.load(instr_id)
    responses = {}

    question_dataset = QsDataset.load(dataset_id)
    # Process each question
    for qid, q in tqdm(
        question_dataset.question_by_qid.items(),
        desc="Generating CoT responses",
    ):
        # Get existing responses for this question
        existing_q_responses = {}
        if (
            existing_responses is not None
            and qid in existing_responses.responses_by_qid
        ):
            existing_q_responses = existing_responses.responses_by_qid[qid]

        # Calculate how many more responses we need
        n_existing = len(existing_q_responses)
        n_needed = max(0, n_responses - n_existing)

        if n_needed == 0:
            responses[qid] = existing_q_responses
            continue

        q_str = q.q_str if question_type == "yes-no" else q.q_str_open_ended
        new_responses = get_question_cot_responses(
            model=model,
            tokenizer=tokenizer,
            question_str=q_str,
            cot_instruction=instructions.cot,
            sampling_params=sampling_params,
            n_responses=n_needed,
        )

        # Combine existing and new responses
        responses[qid] = {**existing_q_responses, **new_responses}

    return CotResponses(
        responses_by_qid=responses,
        model_id=model_id,
        instr_id=instr_id,
        ds_params=question_dataset.params,
        sampling_params=sampling_params,
    )


async def get_all_cot_responses_in_batch(
    model_id: str,
    dataset_id: str,
    instr_id: str,
    sampling_params: SamplingParams,
    n_responses: int,
    batch_processor: ORBatchProcessor | OABatchProcessor | ANBatchProcessor,
    question_type: Literal["yes-no", "open-ended"],
    existing_responses: CotResponses | None = None,
) -> CotResponses:
    """Async version of get_all_cot_responses that processes requests in parallel."""
    instructions = Instructions.load(instr_id)
    question_dataset = QsDataset.load(dataset_id)

    # Prepare batch items - one for each question and response combination
    batch_items = []
    for qid, q in question_dataset.question_by_qid.items():
        # Get existing responses for this question
        existing_q_responses = {}
        if (
            existing_responses is not None
            and qid in existing_responses.responses_by_qid
        ):
            existing_q_responses = existing_responses.responses_by_qid[qid]

        # Calculate how many more responses we need
        n_existing = len(existing_q_responses)
        n_needed = max(0, n_responses - n_existing)

        if n_needed == 0:
            continue

        q_str = q.q_str if question_type == "yes-no" else q.q_str_open_ended
        prompt = instructions.cot.format(question=q_str)

        # Create n_needed items for this question
        for i in range(n_needed):
            batch_items.append(((qid, i), prompt))

    # Process all requests in parallel
    results = await batch_processor.process_batch(batch_items)

    # Organize results by question ID, including existing responses
    responses: dict[str, dict[str, str]] = {}
    if existing_responses is not None:
        responses = {
            qid: dict(resp) for qid, resp in existing_responses.responses_by_qid.items()
        }

    for (qid, _), response in results:
        if response is not None:
            if qid not in responses:
                responses[qid] = {}
            responses[qid][str(uuid4())] = response

    return CotResponses(
        responses_by_qid=responses,
        model_id=model_id,
        instr_id=instr_id,
        ds_params=question_dataset.params,
        sampling_params=sampling_params,
    )


def get_all_cot_responses_or(
    model_id: str,
    dataset_id: str,
    instr_id: str,
    sampling_params: SamplingParams,
    n_responses: int,
    question_type: Literal["yes-no", "open-ended"],
    max_parallel: int | None = None,
    max_retries: int = 1,
    existing_responses: CotResponses | None = None,
) -> CotResponses:
    """OpenRouter version of get_all_cot_responses that processes requests in parallel using an OpenRouter model."""

    # Create rate limiter if max_parallel is specified
    or_rate_limiter = None
    if max_parallel is not None:
        or_rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    def process_response(or_response: str, item: tuple[str, int]) -> str | None:
        """Process a single response from the model."""
        if or_response is None or or_response == "":
            logging.warning(
                f"Empty response received for model {model_id} and question {item[0]}"
            )
            return None
        logging.info(f"Response received for model {model_id} and question {item[0]}")
        return or_response

    batch_processor = ORBatchProcessor[tuple[str, int], str](
        or_model_ids=[model_id],  # Wrap in list for consistency with ORBatchProcessor
        temperature=sampling_params.temperature,
        max_new_tokens=sampling_params.max_new_tokens,
        or_rate_limiter=or_rate_limiter,
        max_retries=max_retries,
        process_response=process_response,
    )

    return asyncio.run(
        get_all_cot_responses_in_batch(
            model_id=model_id,
            dataset_id=dataset_id,
            instr_id=instr_id,
            sampling_params=sampling_params,
            n_responses=n_responses,
            question_type=question_type,
            batch_processor=batch_processor,
            existing_responses=existing_responses,
        )
    )


def get_all_cot_responses_oa(
    model_id: str,
    dataset_id: str,
    instr_id: str,
    sampling_params: SamplingParams,
    n_responses: int,
    question_type: Literal["yes-no", "open-ended"],
    max_parallel: int | None = None,
    max_retries: int = 1,
    existing_responses: CotResponses | None = None,
) -> CotResponses:
    """OpenAI version of get_all_cot_responses that processes requests in parallel using an OpenAI model."""

    # Create rate limiter if max_parallel is specified
    oa_rate_limiter = None
    if max_parallel is not None:
        oa_rate_limiter = OARateLimiter(
            requests_per_interval=max_parallel,
            tokens_per_interval=max_parallel * sampling_params.max_new_tokens,
            interval_seconds=1,
        )

    def process_response(or_response: str, item: tuple[str, int]) -> str | None:
        """Process a single response from the model."""
        if or_response is None or or_response == "":
            logging.warning(
                f"Empty response received for model {model_id} and question {item[0]}"
            )
            return None
        logging.info(f"Response received for model {model_id} and question {item[0]}")
        return or_response

    # Determine if the model is a Claude model (o1 or o1-mini)
    is_claude_model = model_id in ["o1", "o1-mini"]
    token_param = "max_completion_tokens" if is_claude_model else "max_new_tokens"

    batch_processor = OABatchProcessor[tuple[str, int], str](
        oa_model_ids=[model_id],
        temperature=sampling_params.temperature,
        oa_rate_limiter=oa_rate_limiter,
        max_retries=max_retries,
        process_response=process_response,
        **{token_param: sampling_params.max_new_tokens},
    )

    return asyncio.run(
        get_all_cot_responses_in_batch(
            model_id=model_id,
            dataset_id=dataset_id,
            instr_id=instr_id,
            sampling_params=sampling_params,
            n_responses=n_responses,
            question_type=question_type,
            batch_processor=batch_processor,
            existing_responses=existing_responses,
        )
    )


def get_all_cot_responses_an(
    model_id: str,
    dataset_id: str,
    instr_id: str,
    sampling_params: SamplingParams,
    n_responses: int,
    question_type: Literal["yes-no", "open-ended"],
    max_parallel: int | None = None,
    max_retries: int = 1,
    existing_responses: CotResponses | None = None,
) -> CotResponses:
    """Anthropic version of get_all_cot_responses that processes requests in parallel using an Anthropic model."""

    # Create rate limiter if max_parallel is specified
    an_rate_limiter = None
    if max_parallel is not None:
        an_rate_limiter = ANRateLimiter(
            requests_per_interval=max_parallel,
            tokens_per_interval=max_parallel * sampling_params.max_new_tokens,
            interval_seconds=1,
        )

    def process_response(aa_response: str, item: tuple[str, int]) -> str | None:
        """Process a single response from the model."""
        if aa_response is None or aa_response == "":
            logging.warning(
                f"Empty response received for model {model_id} and question {item[0]}"
            )
            return None
        logging.info(f"Response received for model {model_id} and question {item[0]}")
        return aa_response

    batch_processor = ANBatchProcessor[tuple[str, int], str](
        an_model_ids=[model_id],
        temperature=sampling_params.temperature,
        an_rate_limiter=an_rate_limiter,
        max_retries=max_retries,
        process_response=process_response,
        max_new_tokens=sampling_params.max_new_tokens,
    )

    return asyncio.run(
        get_all_cot_responses_in_batch(
            model_id=model_id,
            dataset_id=dataset_id,
            instr_id=instr_id,
            sampling_params=sampling_params,
            n_responses=n_responses,
            question_type=question_type,
            batch_processor=batch_processor,
            existing_responses=existing_responses,
        )
    )
