from uuid import uuid4

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chainscope.questions import QsDataset
from chainscope.typing import *
from chainscope.utils import get_model_device, make_chat_prompt


def get_question_cot_responses_local(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_str: str,
    cot_instruction: str,
    sampling_params: SamplingParams,
    n_responses: int,
) -> dict[str, str]:
    """This doesn't work anymore, we're using the API now."""
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


def create_batch_of_cot_prompts(
    question_dataset: QsDataset,
    instructions: Instructions,
    question_type: Literal["yes-no", "open-ended"],
    n_responses: int,
    existing_responses: CotResponses | None = None,
) -> list[tuple[QuestionResponseId, str]]:
    """Create a batch of CoT prompts for questions that need responses.

    Args:
        question_dataset: Dataset containing questions
        instructions: Instructions for CoT generation
        question_type: Type of questions to generate responses for
        n_responses: Number of responses needed per question
        existing_responses: Existing responses to skip

    Returns:
        List of tuples containing (question response ID, prompt)
    """
    batch_items: list[tuple[QuestionResponseId, str]] = []
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

        if question_type == "yes-no":
            q_str = q.q_str
            prompt = instructions.cot.format(question=q_str)
        else:
            q_str = q.q_str_open_ended
            prompt = instructions.open_ended_cot.format(question=q_str)

        # Create n_needed items for this question
        for _ in range(n_needed):
            q_response_id = QuestionResponseId(qid=qid, uuid=str(uuid4()))
            batch_items.append((q_response_id, prompt))

    return batch_items


def create_cot_responses(
    responses_by_qid: dict[str, dict[str, MathResponse | AtCoderResponse | str]] | None,
    new_responses: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params: DatasetParams,
    sampling_params: SamplingParams,
) -> CotResponses:
    """Create CotResponses from existing responses and new responses.

    Args:
        responses_by_qid: Existing responses by question ID
        new_responses: New responses to add (item, response)
        model_id: Model ID
        instr_id: Instruction ID
        ds_params: Dataset parameters
        sampling_params: Sampling parameters

    Returns:
        CotResponses object
    """
    # Start with existing responses if any
    responses: dict[str, dict[str, MathResponse | AtCoderResponse | str]] = {}
    if responses_by_qid is not None:
        responses = {qid: dict(resp) for qid, resp in responses_by_qid.items()}

    # Add new responses
    for q_resp_id, response in new_responses:
        if not response:
            continue
        if q_resp_id.qid not in responses:
            responses[q_resp_id.qid] = {}
        responses[q_resp_id.qid][q_resp_id.uuid] = response

    return CotResponses(
        responses_by_qid=responses,
        model_id=model_id,
        instr_id=instr_id,
        ds_params=ds_params,
        sampling_params=sampling_params,
    )
