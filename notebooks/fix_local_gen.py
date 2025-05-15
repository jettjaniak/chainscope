# %%
import os

os.chdir("/workspace/chainscope")
file = "chainscope/data/cot_responses/instr-v0/T0.0_P1.0_M32768/filtered_putnambench/Qwen__QwQ-32B-Preview_v0_32k_tokens.yaml"


import yaml

with open(file, "r") as f:
    data = yaml.safe_load(f)

# %%
problem_answer = data["responses_by_qid"]["putnam_1975_a1"]["6ef5b43f"]["model_answer"][0]
end_token = problem_answer.find("Human:")
problem_answer = problem_answer[:end_token]
# %%
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("Qwen/QwQ-32B-Preview", n_devices=2)

# %%
preamble = "Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n"

with open(
    "chainscope/data/putnam2/minimal_fork_of_putnambench_with_clear_answers.yaml", "r"
) as f:
    problems = yaml.safe_load(f)

problem = [p for p in problems if p["problem_name"] == "putnam_1975_a1"][0]
problem_statement = problem["informal_statement"]

input_str = f"{preamble}{problem_statement}{problem_answer}"
tokens = model.to_tokens(input_str, prepend_bos=True).to("cuda:0")
temperature = 0.0
top_p = 1.0


# %%
from typing import Union, List, Optional, Literal, Sequence, Tuple, Dict, Any, Callable, Generator
from jaxtyping import Float, Int
class HookedTransformerWithGenerator:
    
    def __init__(self, hooked_transformer: HookedTransformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooked_transformer = hooked_transformer
    
    def __getattr__(self, name):
        if name != "generate":
            return getattr(self.hooked_transformer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
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
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                input = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
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
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            sampled_tokens_list = []
            for index in tqdm(range(max_new_tokens), disable=not verbose):
                pos_offset = self.get_pos_offset(past_kv_cache, batch_size)

                tokens = torch.zeros((embeds.size(0), embeds.size(1))).to(torch.int)
                attention_mask = utils.get_attention_mask(
                    self.tokenizer, tokens, False if prepend_bos is None else prepend_bos
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
                                (input_tokens, torch.cat(sampled_tokens_list, dim=1)), dim=1
                            )
                            if "sampled_tokens" in locals()
                            else input_tokens,
                        ).to(devices.get_device_for_block_index(0, self.cfg))
                    else:
                        sampled_tokens = utils.sample_logits(
                            final_logits, top_k=top_k, top_p=top_p, temperature=temperature
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

                embeds = torch.hstack([embeds, self.embed(sampled_tokens.unsqueeze(-1))])

                if stop_at_eos and finished_sequences.all():
                    break

# %%
import torch
gen_model = HookedTransformerWithGenerator(model)
# %%
#special_tokens = [model.tokenizer.encode(model.tokenizer.pad_token)]

with torch.inference_mode():
    generated = []
    output = model.generate(
        tokens,
        max_new_tokens=10,
        temperature=temperature,
        top_p=top_p,
        return_type="tokens",
        eos_token_id=model.tokenizer.pad_token_id
    )
 

    # for token in model.generate(
    #     tokens,
    #     max_new_tokens=10,
    #     temperature=temperature,
    #     top_p=top_p,
    #     return_type="tokens",
    # ):
    #     generated.append(token)
    #     break

# model.generate(input, max_new_tokens=10)
print(len(generated))
