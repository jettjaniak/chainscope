import random
from uuid import uuid4

import torch as t
from vllm import LLM
from vllm import SamplingParams as VLLMSamplingParams

from chainscope.questions import QsDataset
from chainscope.typing import *
from chainscope.utils import is_instruct_model, make_chat_prompt


def build_fsp_prompt(
    model_id_for_fsp: str,
    fsp_size: int,
    instr_id: str,
    ds_params: DatasetParams,
    sampling_params: SamplingParams,
    fsp_seed: int,
) -> str:
    random.seed(fsp_seed)
    instructions = Instructions.load(instr_id)

    # Load CoT responses from model_id_for_fsp for this dataset
    cot_responses_path = ds_params.cot_responses_path(
        instr_id=instr_id,
        model_id=model_id_for_fsp,
        sampling_params=sampling_params,
    )
    cot_responses = CotResponses.load(cot_responses_path)

    qs_dataset_path = ds_params.qs_dataset_path
    qs_dataset = QsDataset.load_from_path(qs_dataset_path)

    cot_prompts = []
    for qid, responses in cot_responses.responses_by_qid.items():
        q_str = qs_dataset.question_by_qid[qid].q_str
        prompt = instructions.cot.format(question=q_str)
        for resp in responses.values():
            assert isinstance(resp, str)
            prompt_and_resp = f"{prompt}{resp}"
            cot_prompts.append(prompt_and_resp)

    # Choose fsp_size random prompts
    fsp_prompts = random.sample(cot_prompts, fsp_size)
    fsp_prompt = "\n\n".join(fsp_prompts)

    return fsp_prompt


def get_local_responses(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params: DatasetParams,
    sampling_params: SamplingParams,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
) -> list[tuple[QuestionResponseId, str]]:
    assert instr_id == "instr-wm", "Only instr-wm is supported for local generation"

    if model_id_for_fsp is not None:
        assert not is_instruct_model(model_id), "Why?"
        fsp_prompt = build_fsp_prompt(
            model_id_for_fsp=model_id_for_fsp,
            fsp_size=fsp_size,
            instr_id=instr_id,
            ds_params=ds_params,
            sampling_params=sampling_params,
            fsp_seed=fsp_seed,
        )
    else:
        fsp_prompt = None

    # Initialize vLLM engine
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=t.cuda.device_count(),
    )

    instr_prefix = "Here is a question with a clear YES or NO answer"

    # Convert our sampling params to vLLM format
    vllm_params = VLLMSamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=sampling_params.max_new_tokens,
        stop=["**NO**", "**YES**", "\n\nNO", "\n\nYES", instr_prefix],
        include_stop_str_in_output=True,
    )

    # Prepare prompts
    prompt_texts = []
    q_resp_ids = []
    for q_resp_id, prompt in prompts:
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=llm.get_tokenizer(),
            )
        else:
            if fsp_prompt is not None:
                input_str = f"{fsp_prompt}\n\n{prompt}"
            else:
                input_str = prompt

        prompt_texts.append(input_str)
        q_resp_ids.append(q_resp_id)

    # Generate responses using vLLM
    outputs = llm.generate(prompt_texts, vllm_params)

    # Format responses
    responses: list[tuple[QuestionResponseId, str]] = []
    for q_resp_id, output in zip(q_resp_ids, outputs):
        generated_text = output.outputs[0].text

        if instr_prefix in generated_text:
            generated_text = generated_text.replace(instr_prefix, "")

        responses.append((q_resp_id, generated_text))

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
