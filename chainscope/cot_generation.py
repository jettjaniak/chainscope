from uuid import uuid4

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chainscope.questions import QsDataset
from chainscope.typing import *
from chainscope.utils import get_model_device, make_chat_prompt


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


def get_all_cot_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_dataset: QsDataset,
    instr_id: str,
    sampling_params: SamplingParams,
    n_responses: int,
) -> CotResponses:
    instructions = Instructions.load(instr_id)
    responses = {}

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
        model_id=model.name_or_path,
        instr_id=instr_id,
        sampling_params=sampling_params,
    )
