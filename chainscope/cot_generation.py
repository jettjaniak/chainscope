import logging
import random
from uuid import uuid4

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
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
    instruction_cache: dict[str, Instructions],
    cot_responses_cache: dict[str, CotResponses],
    qs_dataset_cache: dict[str, QsDataset],
) -> str:
    random.seed(fsp_seed)

    # Get Instructions from cache or load them
    if instr_id in instruction_cache:
        instructions = instruction_cache[instr_id]
    else:
        instructions = Instructions.load(instr_id)
        instruction_cache[instr_id] = instructions

    # Load CoT responses from model_id_for_fsp for this dataset
    cot_responses_path = ds_params.cot_responses_path(
        instr_id=instr_id,
        model_id=model_id_for_fsp,
        sampling_params=sampling_params,
    )

    # Convert Path to string for dictionary key
    cot_responses_path_str = str(cot_responses_path)
    if cot_responses_path_str in cot_responses_cache:
        cot_responses = cot_responses_cache[cot_responses_path_str]
    else:
        cot_responses = CotResponses.load(cot_responses_path)
        cot_responses_cache[cot_responses_path_str] = cot_responses

    qs_dataset_path = ds_params.qs_dataset_path

    # Convert Path to string for dictionary key
    qs_dataset_path_str = str(qs_dataset_path)
    if qs_dataset_path_str in qs_dataset_cache:
        qs_dataset = qs_dataset_cache[qs_dataset_path_str]
    else:
        qs_dataset = QsDataset.load_from_path(qs_dataset_path)
        qs_dataset_cache[qs_dataset_path_str] = qs_dataset

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


def get_local_responses_vllm(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params_list: list[DatasetParams],
    sampling_params: SamplingParams,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    qid_to_dataset: dict[str, str],
    batch_size: int = 1024,
) -> list[tuple[QuestionResponseId, str]]:
    assert instr_id == "instr-wm", "Only instr-wm is supported for local generation"
    if model_id_for_fsp is not None:
        assert not is_instruct_model(model_id), "Why?"

    # Initialize caches
    instruction_cache: dict[str, Instructions] = {}
    cot_responses_cache: dict[str, CotResponses] = {}
    qs_dataset_cache: dict[str, QsDataset] = {}

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
    for q_resp_id, prompt in tqdm(prompts, desc="Preparing prompts"):
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=llm.get_tokenizer(),  # type: ignore
            )
        else:
            # Get FSP prompt for this dataset if needed
            if model_id_for_fsp is not None:
                dataset_id = qid_to_dataset[q_resp_id.qid]
                ds_idx = next(
                    i for i, ds in enumerate(ds_params_list) if ds.id == dataset_id
                )
                ds_params = ds_params_list[ds_idx]
                fsp_prompt = build_fsp_prompt(
                    model_id_for_fsp=model_id_for_fsp,
                    fsp_size=fsp_size,
                    instr_id=instr_id,
                    ds_params=ds_params,
                    sampling_params=sampling_params,
                    fsp_seed=fsp_seed,
                    instruction_cache=instruction_cache,
                    cot_responses_cache=cot_responses_cache,
                    qs_dataset_cache=qs_dataset_cache,
                )
                input_str = f"{fsp_prompt}\n\n{prompt}"
            else:
                input_str = prompt

        prompt_texts.append(input_str)
        q_resp_ids.append(q_resp_id)

    # Generate responses using vLLM in batches
    logging.info(
        f"Generating {len(prompt_texts)} responses with batch size {batch_size}"
    )
    all_outputs = []
    for i in tqdm(range(0, len(prompt_texts), batch_size), desc="Processing batches"):
        batch_prompts = prompt_texts[i : i + batch_size]
        batch_outputs = llm.generate(batch_prompts, vllm_params, use_tqdm=True)
        all_outputs.extend(batch_outputs)
    logging.info(f"Generated {len(all_outputs)} responses")

    # Format responses
    responses: list[tuple[QuestionResponseId, str]] = []
    for q_resp_id, output in tqdm(
        zip(q_resp_ids, all_outputs), desc="Processing responses", total=len(q_resp_ids)
    ):
        generated_text = output.outputs[0].text

        if instr_prefix in generated_text:
            generated_text = generated_text.replace(instr_prefix, "")

        responses.append((q_resp_id, generated_text))

    return responses


def get_local_responses_tl(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params_list: list[DatasetParams],
    sampling_params: SamplingParams,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    local_gen_seed: int,
    qid_to_dataset: dict[str, str],
) -> list[tuple[QuestionResponseId, str]]:
    """Generate responses using TransformerLens framework.

    Args:
        prompts: List of (question ID, prompt text) tuples
        model_id: Name of the model to use
        instr_id: Instruction ID
        ds_params_list: List of dataset parameters
        sampling_params: Sampling parameters
        model_id_for_fsp: Model ID for few-shot prompting (optional)
        fsp_size: Number of few-shot examples
        fsp_seed: Seed for few-shot example selection
        local_gen_seed: Seed for generation
        qid_to_dataset: Mapping from question IDs to dataset IDs

    Returns:
        List of (question ID, generated response) tuples
    """
    assert instr_id == "instr-wm", "Only instr-wm is supported for local generation"
    if model_id_for_fsp is not None:
        assert not is_instruct_model(model_id), "Why?"

    # Initialize caches
    instruction_cache: dict[str, Instructions] = {}
    cot_responses_cache: dict[str, CotResponses] = {}
    qs_dataset_cache: dict[str, QsDataset] = {}

    # Set TransformerLens seed for reproducible local generation
    HookedTransformerConfig.set_seed_everywhere(
        None,  # type: ignore
        local_gen_seed,
    )

    # Initialize TransformerLens model
    model = HookedTransformer.from_pretrained(
        model_name=model_id,
        device="cuda",
    )
    assert model.tokenizer is not None, "Tokenizer is not initialized"

    instr_prefix = "Here is a question with a clear YES or NO answer"
    stop_tokens = ["**NO**", "**YES**", "\n\nNO", "\n\nYES", instr_prefix]

    # Prepare prompts
    responses: list[tuple[QuestionResponseId, str]] = []
    for q_resp_id, prompt in tqdm(prompts, desc="Generating responses"):
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=model.tokenizer,  # type: ignore
            )
        else:
            # Get FSP prompt for this dataset if needed
            if model_id_for_fsp is not None:
                dataset_id = qid_to_dataset[q_resp_id.qid]
                ds_idx = next(
                    i for i, ds in enumerate(ds_params_list) if ds.id == dataset_id
                )
                ds_params = ds_params_list[ds_idx]
                fsp_prompt = build_fsp_prompt(
                    model_id_for_fsp=model_id_for_fsp,
                    fsp_size=fsp_size,
                    instr_id=instr_id,
                    ds_params=ds_params,
                    sampling_params=sampling_params,
                    fsp_seed=fsp_seed,
                    instruction_cache=instruction_cache,
                    cot_responses_cache=cot_responses_cache,
                    qs_dataset_cache=qs_dataset_cache,
                )
                input_str = f"{fsp_prompt}\n\n{prompt}"
            else:
                input_str = prompt

        # Tokenize input
        tokens = model.to_tokens(input_str, prepend_bos=True).to(model.cfg.device)
        assert isinstance(tokens, t.Tensor)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1

        # Generate the full sequence at once
        with t.inference_mode():
            generated = model.generate(
                tokens,
                max_new_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                return_type="tokens",
                verbose=False,
            )
            assert isinstance(
                generated, t.Tensor
            )  # : Int[t.Tensor, "1 pos_plus_new_tokens"]
            assert generated.ndim == 2

        # Convert output tokens to text
        generated_text = model.tokenizer.batch_decode(
            generated[:, tokens.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        assert isinstance(
            generated_text, str
        ), f"Generated text is not a string: {type(generated_text)}, {generated_text}"

        # Find the first occurrence of any stop sequence and truncate
        min_stop_idx = len(generated_text)
        for stop_seq in stop_tokens:
            stop_idx = generated_text.find(stop_seq)
            if stop_idx != -1 and stop_idx < min_stop_idx:
                min_stop_idx = stop_idx + len(stop_seq)

        # Truncate at the earliest stop sequence
        generated_text = generated_text[:min_stop_idx]

        # Clean up response
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
