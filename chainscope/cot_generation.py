import random
from typing import (
    List,
    Literal,
    Optional,
    Union,
)
from uuid import uuid4

import torch
import torch as t
from jaxtyping import Float, Int
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
from vllm import LLM
from vllm import SamplingParams as VLLMSamplingParams

from chainscope.questions import QsDataset
from chainscope.typing import *
from chainscope.utils import is_instruct_model, make_chat_prompt


class HookedTransformerWithGenerator:
    def __init__(self, hooked_transformer: HookedTransformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooked_transformer = hooked_transformer

    def __getattr__(self, name):
        if name != "generate":
            return getattr(self.hooked_transformer, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @torch.inference_mode()
    def generate(
        self,
        input: Union[
            str,
            List[str],
            Int[torch.Tensor, "batch pos"],
            Float[torch.Tensor, "batch pos hidden_size"],
        ] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = True,
        padding_side: Optional[Literal["left", "right"]] = "left",
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Union[
        str,
        List[str],
        Int[torch.Tensor, "batch pos_plus_new_tokens"],
        Float[torch.Tensor, "batch pos_plus_new_tokens hidden_size"],
    ]:
        """Sample Tokens from the Model.

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
        (by producing an EOT token), we keep running the model on the entire batch, but throw away
        the output for a finished sequence and just keep adding EOTs to pad.

        Args:
            input (Union[str, List[str], Int[torch.Tensor, "batch pos"], Float[torch.Tensor, "batch pos hidden_size"]]):
                A text string (this will be converted to a batch of tokens with batch
                size 1), a list of strings, batch of tokens or a tensor of precomputed embeddings of shape
                [batch, pos, hidden_size].
            max_new_tokens (int): Maximum number of tokens to generate.
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
                of sentence. If None, use the tokenizer's eos_token_id - required if using
                stop_at_eos. It's also possible to provide a list of token IDs (not just the
                eos_token_id), in which case the generation will stop when any of them are output
                (useful e.g. for stable_lm).
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
                greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens.
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
                we take the top tokens with cumulative probability >= top_p.
            temperature (float): Temperature for sampling. Higher values will make the model more
                random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
                sampling from a uniform distribution).
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
                tokens. Higher values will make the model more random. Works only with str and tokens input.
            use_past_kv_cache (bool): If True, create and use cache to speed up generation.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
            return_type (Optional[str]): The type of the output to return - a string or a list of strings ('str'),
                a tensor of tokens ('tokens'), a tensor of output embeddings ('embeds') or whatever the format of the
                input was ('input').
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            outputs (str, List[str], Int[torch.Tensor, "batch pos_plus_new_tokens"], Float[torch.Tensor,
                "batch pos_plus_new_tokens hidden_size"]): generated sequence. Str, tokens or embeddings.
                If input is embeddings and return type is tokens or string, returns only new generated sequence.
                In other cases returns sequence including input sequence.
        """

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert isinstance(input, (str, torch.Tensor, list)) and (
                isinstance(input, list)
                and all(isinstance(i, str) for i in input)
                or not isinstance(input, list)
            ), "Input must be either string, torch.Tensor, or List[str]"

            assert return_type in [
                "input",
                "str",
                "tokens",
                "embeds",
            ], "return_type must be one of ['input', 'str', 'tokens', 'embeds']"

            if return_type == "input":
                if isinstance(input, (str, list)):
                    return_type = "str"
                elif input.ndim == 2:
                    return_type = "tokens"
                else:
                    return_type = "embeds"

            if isinstance(input, (str, list)):
                input_type = "str"
                # If text, convert to tokens (batch_size=1)
                assert self.tokenizer is not None, (
                    "Must provide a tokenizer if passing a string to the model"
                )
                input = self.to_tokens(
                    input, prepend_bos=prepend_bos, padding_side=padding_side
                )
            elif input.ndim == 2:
                input_type = "tokens"
            else:
                input_type = "embeds"

            input_tokens = input if input_type in ["str", "tokens"] else None
            batch_size, ctx_length = input.shape[0], input.shape[1]
            device = devices.get_device_for_block_index(0, self.cfg)
            input = input.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.cfg, self.cfg.device, batch_size
                )
            else:
                past_kv_cache = None

            shortformer_pos_embed = None
            embeds = input if input_type == "embeds" else self.embed(input)

            assert isinstance(embeds, torch.Tensor) and embeds.ndim == 3

            stop_tokens: List[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tokenizer is not None
                    and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert tokenizer_has_eos_token, (
                        "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
                    )

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id
                        if tokenizer_has_eos_token
                        else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(
                batch_size, dtype=torch.bool, device=self.cfg.device
            )

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            sampled_tokens_list = []
            for index in tqdm(range(max_new_tokens), disable=not verbose):
                pos_offset = self.get_pos_offset(past_kv_cache, batch_size)

                tokens = torch.zeros((embeds.size(0), embeds.size(1))).to(torch.int)
                attention_mask = utils.get_attention_mask(
                    self.tokenizer,
                    tokens,
                    False if prepend_bos is None else prepend_bos,
                ).to(device)
                residual, shortformer_pos_embed = self.get_residual(
                    embeds,
                    pos_offset,
                    return_shortformer_pos_embed=True,
                    device=device,
                    attention_mask=attention_mask,
                )

                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                start_at_layer = 0  # Make forward returns embeddings
                if use_past_kv_cache:
                    # We just take the final tokens, as a [batch, 1] tensor
                    if index > 0:
                        logits = self.forward(
                            residual[:, -1:],
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                            start_at_layer=start_at_layer,
                            shortformer_pos_embed=shortformer_pos_embed,
                        )
                    else:
                        logits = self.forward(
                            residual,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                            start_at_layer=start_at_layer,
                            shortformer_pos_embed=shortformer_pos_embed,
                        )
                else:
                    # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                    # the cache.
                    logits = self.forward(
                        residual,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        start_at_layer=start_at_layer,
                        shortformer_pos_embed=shortformer_pos_embed,
                    )
                final_logits = logits[:, -1, :]

                if do_sample:
                    if input_type in [
                        "str",
                        "tokens",
                    ]:  # Those types of inputs support frequency penalty
                        sampled_tokens = utils.sample_logits(
                            final_logits,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            freq_penalty=freq_penalty,
                            tokens=torch.cat(
                                (input_tokens, torch.cat(sampled_tokens_list, dim=1)),
                                dim=1,
                            )
                            if "sampled_tokens" in locals()
                            else input_tokens,
                        ).to(devices.get_device_for_block_index(0, self.cfg))
                    else:
                        sampled_tokens = utils.sample_logits(
                            final_logits,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                        ).to(devices.get_device_for_block_index(0, self.cfg))
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.cfg)
                    )
                yield sampled_tokens.unsqueeze(1)
                sampled_tokens_list.append(sampled_tokens.unsqueeze(1))
                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )

                embeds = torch.hstack(
                    [embeds, self.embed(sampled_tokens.unsqueeze(-1))]
                )

                if stop_at_eos and finished_sequences.all():
                    break


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


def get_local_responses_vllm(
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


def get_local_responses_tl(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params: DatasetParams,
    sampling_params: SamplingParams,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    local_gen_seed: int,
) -> list[tuple[QuestionResponseId, str]]:
    """Generate responses using TransformerLens framework.

    Args:
        prompts: List of (question ID, prompt text) tuples
        model_id: Name of the model to use
        instr_id: Instruction ID
        ds_params: Dataset parameters
        sampling_params: Sampling parameters
        model_id_for_fsp: Model ID for few-shot prompting (optional)
        fsp_size: Number of few-shot examples
        fsp_seed: Seed for few-shot example selection
        local_gen_seed: Seed for generation

    Returns:
        List of (question ID, generated response) tuples
    """
    assert instr_id == "instr-wm", "Only instr-wm is supported for local generation"

    # Set TransformerLens seed for reproducible local generation
    HookedTransformerConfig.set_seed_everywhere(
        None,  # type: ignore
        local_gen_seed,
    )

    # Get few-shot prompt if needed
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
    model = HookedTransformerWithGenerator(model)
    for q_resp_id, prompt in tqdm(prompts, desc="Generating responses"):
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=model.tokenizer,
            )
        else:
            if fsp_prompt is not None:
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
            generated = []
            try:
                for token in model.generate(
                    tokens,
                    max_new_tokens=sampling_params.max_new_tokens,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    return_type="tokens",
                    verbose=True,
                ):
                    generated.append(token)
            except Exception:
                pass
            generated = torch.cat(generated, dim=1)
            generated = torch.cat((tokens, generated), dim=1)
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
        assert isinstance(generated_text, str), (
            f"Generated text is not a string: {type(generated_text)}, {generated_text}"
        )

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
