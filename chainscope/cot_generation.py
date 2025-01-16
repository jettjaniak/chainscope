import time
from typing import Any
from uuid import uuid4

import openai
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
        responses[qid] = get_question_cot_responses(
            model=model,
            tokenizer=tokenizer,
            question_str=q.q_str,
            cot_instruction=instructions.cot,
            sampling_params=sampling_params,
            n_responses=n_responses,
        )

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
) -> CotResponses:
    """OpenRouter version of get_all_cot_responses."""
    # Initialize OpenAI client with OpenRouter configuration
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1")

    instructions = Instructions.load(instr_id)
    responses = {}

    question_dataset = QsDataset.load(dataset_id)
    # Process each question
    for qid, q in tqdm(
        question_dataset.question_by_qid.items(),
        desc="Generating CoT responses",
    ):
        responses[qid] = get_question_cot_responses_or(
            client=client,
            question_str=q.q_str,
            cot_instruction=instructions.cot,
            sampling_params=sampling_params,
            n_responses=n_responses,
            model_id=model_id,  # Should be in format "openai/gpt-3.5-turbo"
        )

    return CotResponses(
        responses_by_qid=responses,
        model_id=model_id,
        instr_id=instr_id,
        ds_params=question_dataset.params,
        sampling_params=sampling_params,
    )
