# %%
from functools import partial

import torch
from jaxtyping import Float

# from chainscope.typing import *
from chainscope.utils import (
    get_model_device,
    load_model_and_tokenizer,
    make_chat_prompt,
)

# %%
# Load model and tokenizer
model_id = "meta-llama/Llama-3.3-70B-Instruct"
model, tokenizer = load_model_and_tokenizer(model_id)

# %%

# Prepare prompt
prompt = """Here is a question with a clear YES or NO answer about historical figures:

Did Gerard Segarelli die at an earlier date than Brian of Brittany?

It requires a few steps of reasoning. So first, think step by step, and only then give a YES / NO answer."""

chat_input = make_chat_prompt(
    instruction=prompt,
    tokenizer=tokenizer,
)

input_ids = tokenizer.encode(
    chat_input, return_tensors="pt", add_special_tokens=False
).to(get_model_device(model))  # type: ignore

print(input_ids.device)
print(tokenizer.decode(input_ids[0]))

# %% Append response

response = """To answer this question, I need to determine the death dates of Gerard Segarelli and Brian of Brittany.

Gerard Segarelli was the founder of the Apostolic Brethren, a Christian sect that emerged in the 13th century. According to historical records, Gerard Segarelli died in 1300.

Brian of Brittany, on the other hand, is not a widely recognized historical figure. However, I found a reference to a Brian of Brittany who was a 13th-century nobleman. Unfortunately, I couldn't find a specific death date for him.

However, after further research, I found that Brian of Brittany might be referring to Brian of Penthi\xe8vre, also known as Brian of Brittany, who died around 1272, or possibly another Brian, but the dates I could find are all earlier than the 14th century.

Given the available information, it appears that Gerard Segarelli died in 1300, which is later than the possible death dates I found for Brian of Brittany.

So, based on the available data, the answer to the question is: YES."""

# %%
# Get all layers (including embedding layer)
layers = list(range(model.config.num_hidden_layers + 1))  # +1 for embedding layer


# %%
def resid_stream_hook_fn(
    module,
    input,
    output,
    layer: int,
    acts_by_layer: dict[int, Float[torch.Tensor, "seq_len model"]],
):
    """Hook function that caches activations for all positions in a layer."""
    if isinstance(output, tuple):
        output = output[0]
    if len(output.shape) != 3:
        print(f"Expected tensor of shape (1, seq_len, d_model), got {output.shape}")
    # we're running batch size 1
    output = output[0]  # shape: (seq_len, d_model)
    acts_by_layer[layer] = output.cpu()


def last_token_attn_pttn_hook_fn(
    module,
    input,
    output,
    component_name: str,
    last_token_attn_pttns: dict[str, Float[torch.Tensor, "model"]],
):
    """Hook function that caches activations for the last token only."""
    if isinstance(output, tuple):
        output = output[0]
    if len(output.shape) != 4:
        print(
            f"Expected tensor of shape (1, seq_len, n_head, d_model), got {output.shape}"
        )
    # we're running batch size 1, get last token
    # head_index, destination_position, source_position
    output = output[0, -1, :]  # shape: (n_head, d_model)
    last_token_attn_pttns[component_name] = output.cpu()


def layer_to_hook_point(layer: int) -> str:
    if layer == 0:
        return "model.embed_tokens"  # Embedding layer
    return f"model.layers.{layer-1}"  # Transformer layers


# %%
# Collect activations for all positions
acts_by_layer = {}
last_token_attn_pttns = {}  # New dictionary for attention patterns of last token
hooks = []

hook_points = set(layer_to_hook_point(i) for i in layers)
hook_points_cnt = len(hook_points)

print("Available modules in model:")
for name, _ in model.named_modules():
    print(name)

# Hook for residual stream (all positions)
for name, module in model.named_modules():
    if name in hook_points:
        hook_points_cnt -= 1
        layer = 0 if name == "model.embed_tokens" else int(name.split(".")[-1]) + 1
        hook_fn = partial(
            resid_stream_hook_fn, layer=layer, acts_by_layer=acts_by_layer
        )
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

# Hook for last token activations from various components
last_token_components = []
for name, module in model.named_modules():
    # Collect activations from key components
    if name.startswith("model.layers.") and name.endswith(".self_attn.o_proj"):
        last_token_components.append(name)
        hook_fn = partial(
            last_token_attn_pttn_hook_fn,
            component_name=name,
            last_token_attn_pttns=last_token_attn_pttns,
        )
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

print(f"Added hooks for {len(last_token_components)} last token components:")
for comp in last_token_components:
    print(f"  {comp}")

assert hook_points_cnt == 0, f"Could not find all hook points: {hook_points}"
try:
    # generated_ids = model.generate(
    #     input_ids,
    #     max_new_tokens=10,
    #     pad_token_id=tokenizer.eos_token_id,
    # )[0]
    # tokenizer.decode(generated_ids)

    with torch.inference_mode():
        model(input_ids)
        # output = model(input_ids)
        # print(output.logits.shape)
finally:
    for hook in hooks:
        hook.remove()


# %%

print("Residual stream activations by layer:")
print(acts_by_layer)

print("\nLast token activations by component:")
print(last_token_attn_pttns)
