# %%
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float

from chainscope.typing import *
from chainscope.utils import (get_model_device, load_model_and_tokenizer,
                              make_chat_prompt)

# %%
# Load model and tokenizer
model_id = "meta-llama/Llama-3.3-70B-Instruct"
model, tokenizer = load_model_and_tokenizer(model_id)

# For computing attention patterns
model.config._attn_implementation = "eager"

# Get the unembedding matrix (W_U = lm_head.weight.T)
# model.lm_head.weight has shape (vocab_size, hidden_dim)
# So, W_U = model.lm_head.weight.T has shape (hidden_dim, vocab_size)
W_U = model.lm_head.weight.T

# %%

# Load the data
df = pd.read_pickle(DATA_DIR / "df-wm-non-ambiguous-hard-2.pkl")
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, dataset_suffix, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

df = df[df["mode"] == "cot"]
df = df[df["model_id"] == model_id]

# %%

# Get YES and NO token IDs
yes_token_strs = [" YES", "YES"]
yes_token_ids = tokenizer.encode(yes_token_strs, add_special_tokens=False)
assert len(yes_token_ids) == len(yes_token_strs), "Tokenizer returned different number of token IDs than expected"

no_token_strs = [" NO", "NO"]
no_token_ids = tokenizer.encode(no_token_strs, add_special_tokens=False)
assert len(no_token_ids) == len(no_token_strs), "Tokenizer returned different number of token IDs than expected"

# %%

qid = "2827dc90917aaa08070b15340cf1142c28716aa4381e07fbf5f2dc69be6d342f"

row = df[df["qid"] == qid].iloc[0]
prop_id = row["prop_id"]
dataset_suffix = row["dataset_suffix"]
expected_answer = row["answer"]

# %% Load faithfulness data and unfaithfulness pattern evaluation data
faithfulness_dataset = UnfaithfulnessPairsDataset.load(model_id, prop_id, dataset_suffix)
unfaithfulness_pattern_eval = UnfaithfulnessPatternEval.load(model_id, prop_id, dataset_suffix)

# %%
prompt = faithfulness_dataset.questions_by_qid[qid].prompt
print(prompt)

# %%
q1_analysis = unfaithfulness_pattern_eval.pattern_analysis_by_qid[qid].q1_analysis
if q1_analysis is None:
    raise ValueError(f"No Q1 analysis found for {qid}")

response_ids_showing_answer_flipping = []
all_response_ids = list(q1_analysis.responses.keys())
for response_id, response_analysis in q1_analysis.responses.items():
    if response_analysis.answer_flipping_classification == "YES":
        response_ids_showing_answer_flipping.append(response_id)

print(f"There are {len(response_ids_showing_answer_flipping)} responses showing answer flipping out of {len(q1_analysis.responses)} total responses")

# %%
def resid_stream_hook_fn(
    module,
    input,
    output,
    layer: int,
    acts_by_layer: dict[int, Float[torch.Tensor, "seq_len d_model"]],
):
    """Hook function that caches activations for all positions in a layer."""
    if isinstance(output, tuple):
        output = output[0]
    if len(output.shape) != 3:
        print(f"Expected tensor of shape (1, seq_len, d_model), got {output.shape}")
    # we're running batch size 1
    output = output[0]  # shape: (seq_len, d_model)
    acts_by_layer[layer] = output.cpu()


def attention_proj_hook_fn(
    module,
    input,
    output,
    layer: int,
    proj_type: str,
    attention_q_projs: dict[int, Float[torch.Tensor, "seq_len n_heads head_dim"]],
    attention_k_projs: dict[int, Float[torch.Tensor, "seq_len n_kv_heads head_dim"]],
    attention_o_projs: dict[int, Float[torch.Tensor, "seq_len n_heads head_dim"]],
    module_name: str,
):
    """Hook function that caches query, key, and output projections for computing attention patterns."""
    if isinstance(output, tuple):
        output = output[0]
    if len(output.shape) != 3:
        print(
            f"Expected tensor of shape (1, seq_len, n_heads*head_dim), got {output.shape}"
        )
    # we're running batch size 1
    output = output[
        0
    ]  # shape: (seq_len, n_heads*head_dim) or (seq_len, n_kv_heads*head_dim)
    if proj_type == "q":
        n_heads = model.config.num_attention_heads
        head_dim = model.config.head_dim
        output = output.view(-1, n_heads, head_dim)
        attention_q_projs[layer] = output.cpu()
    elif proj_type == "k":
        n_kv_heads = model.config.num_key_value_heads
        head_dim = model.config.head_dim
        output = output.view(-1, n_kv_heads, head_dim)
        attention_k_projs[layer] = output.cpu()
    elif proj_type == "o":
        n_heads = model.config.num_attention_heads
        head_dim = model.config.head_dim
        output = output.view(-1, n_heads, head_dim)
        attention_o_projs[layer] = output.cpu()


def layer_to_hook_point(layer: int) -> str:
    if layer == 0:
        return "model.embed_tokens"  # Embedding layer
    return f"model.layers.{layer-1}"  # Transformer layers


def post_attn_ln_hook_fn(
    module,
    input,
    output,
    layer: int,
    post_attn_ln_outputs: dict[int, Float[torch.Tensor, "seq_len d_model"]],
):
    """Hook function that caches post-attention layer norm outputs."""
    if isinstance(output, tuple):
        output = output[0]
    if len(output.shape) != 3:
        print(f"Expected tensor of shape (1, seq_len, d_model), got {output.shape}")
    # we're running batch size 1
    output = output[0]  # shape: (seq_len, d_model)
    post_attn_ln_outputs[layer] = output.cpu()


def mlp_hook_fn(
    module,
    input,
    output,
    layer: int,
    mlp_outputs: dict[int, Float[torch.Tensor, "seq_len d_model"]],
):
    """Hook function that caches MLP outputs."""
    if isinstance(output, tuple):
        output = output[0]
    if len(output.shape) != 3:
        print(f"Expected tensor of shape (1, seq_len, d_model), got {output.shape}")
    # we're running batch size 1
    output = output[0]  # shape: (seq_len, d_model)
    mlp_outputs[layer] = output.cpu()


# %%

# Collect data for all layers (including embedding layer)
layers = list(range(model.config.num_hidden_layers + 1))  # +1 for embedding layer

# Collect activations for all positions
acts_by_layer = {}
attention_q_projs = {}  # layer -> (seq_len, n_heads, head_dim)
attention_k_projs = {}  # layer -> (seq_len, n_kv_heads, head_dim)
attention_o_projs = {}  # layer -> (seq_len, n_heads, head_dim)
post_attn_ln_outputs = {}  # layer -> (seq_len, d_model)
mlp_outputs = {}  # layer -> (seq_len, d_model)
hooks = []

hook_points = set(layer_to_hook_point(i) for i in layers)
hook_points_cnt = len(hook_points)

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
    # Hook for attention projections
    elif (
        name.endswith(".self_attn.q_proj")
        or name.endswith(".self_attn.k_proj")
        or name.endswith(".self_attn.o_proj")
    ):
        layer = int(name.split(".")[2])
        proj_type = (
            "q" if name.endswith("q_proj") else "k" if name.endswith("k_proj") else "o"
        )
        hook_fn = partial(
            attention_proj_hook_fn,
            layer=layer,
            proj_type=proj_type,
            attention_q_projs=attention_q_projs,
            attention_k_projs=attention_k_projs,
            attention_o_projs=attention_o_projs,
            module_name=name,
        )
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)
    # Hook for post-attention layer norms
    elif name.endswith(".post_attention_layernorm"):
        layer = int(name.split(".")[2])
        hook_fn = partial(
            post_attn_ln_hook_fn,
            layer=layer,
            post_attn_ln_outputs=post_attn_ln_outputs,
        )
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)
    # Hook for MLPs
    elif name.endswith(".mlp"):
        layer = int(name.split(".")[2])
        hook_fn = partial(
            mlp_hook_fn,
            layer=layer,
            mlp_outputs=mlp_outputs,
        )
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

assert hook_points_cnt == 0, f"Could not find all hook points: {hook_points}"


def run_model(input_ids: torch.Tensor) -> Float[torch.Tensor, "seq_len vocab_size"]:
    with torch.inference_mode():
        outputs = model(input_ids)
        # Save logits (pre-softmax, shape: [1, seq_len, vocab_size])
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits = logits[0].detach().cpu()  # shape: (seq_len, vocab_size)
        return logits

# %%

seq_position_to_analyze_str = "Token Before Final Answer"

def make_plot_logit_trajectory_token_before_final_answer(response_id: str) -> None:
    q_metadata = faithfulness_dataset.questions_by_qid[qid].metadata
    assert q_metadata is not None, f"Metadata is None for {qid}"
    response_str = q_metadata.q1_all_responses[response_id]
    is_answer_flipping_response = response_id in response_ids_showing_answer_flipping

    chat_input = make_chat_prompt(
        instruction=prompt,
        tokenizer=tokenizer,
        response=response_str,
    )

    input_ids = tokenizer.encode(
        chat_input, return_tensors="pt", add_special_tokens=False
    ).to(get_model_device(model))  # type: ignore

    print(f"Tokenized response: `{tokenizer.decode(input_ids[0])}`")

    # Print all tokens
    print("\nAll tokens:")
    for i in range(len(input_ids[0])):
        print(f"Token {i} ({i - len(input_ids[0])}): `{tokenizer.decode(input_ids[0][i])}` (id: {input_ids[0][i]})")

    # Get the token before the final answer    
    # Traverse backwards from the end of the response to find the YES/NO token
    print("\nTraversing backwards from the end of the response to find the YES/NO token")
    seq_position_to_analyze = None

    # Only look at last 150 tokens. If we don't find it we are probably doing something wrong.
    start_idx = max(len(input_ids[0]) - 150, 0)
    main_yes_token_id = None
    main_no_token_id = None
    for i in range(len(input_ids[0]) - 1, start_idx - 1, -1):
        print(f"Token {i}: `{tokenizer.decode(input_ids[0][i])}` (id: {input_ids[0][i]})")
        if input_ids[0][i] in yes_token_ids:
            seq_position_to_analyze = i
            main_yes_token_id = input_ids[0][i]
            break
        elif input_ids[0][i] in no_token_ids:
            seq_position_to_analyze = i
            main_no_token_id = input_ids[0][i]
            break
    
    assert seq_position_to_analyze is not None, "Could not find YES/NO token in response"
    print(f"Analyzing token at position {seq_position_to_analyze} (index {len(input_ids[0]) + seq_position_to_analyze} out of {len(input_ids[0])}): `{tokenizer.decode(input_ids[0][seq_position_to_analyze])}`")

    if main_yes_token_id is None:
        assert main_no_token_id is not None
        idx = no_token_ids.index(main_no_token_id)
        main_yes_token_id = yes_token_ids[idx]
    if main_no_token_id is None:
        assert main_yes_token_id is not None
        idx = yes_token_ids.index(main_yes_token_id)
        main_no_token_id = no_token_ids[idx]

    logits = run_model(input_ids)
    # print("\nTop model prediction (argmax) for each sequence position:")
    # for i in range(logits.shape[0]):
    #     top_token_id = logits[i].argmax().item()
    #     top_token = tokenizer.decode([top_token_id])
    #     print(f"Position {i}: token_id={top_token_id}, token={top_token!r}")

    # Analyze logit for "YES" token
    print(
        f"Using token ID for YES analysis: {main_yes_token_id} (corresponds to '{tokenizer.decode([main_yes_token_id])}')"
    )
    decoded_token_for_plot_yes = tokenizer.decode([main_yes_token_id])

    unembed_vector_yes = W_U[:, main_yes_token_id].clone().detach().cpu()  # Shape: (hidden_dim,)

    # 3. Calculate logit for "YES" at each layer's final position residual stream
    # acts_by_layer keys are layer indices (0 for embeddings, 1 to N for transformer blocks)
    layer_numbers = sorted(acts_by_layer.keys())
    yes_logits_by_layer = []

    # The hook resid_stream_hook_fn already does .cpu() and stores (seq_len, d_model)
    for layer_idx in layer_numbers:
        if layer_idx not in acts_by_layer:
            print(
                f"Warning: Layer {layer_idx} not found in acts_by_layer. Skipping for YES."
            )
            yes_logits_by_layer.append(float("nan"))
            continue

        layer_activations = acts_by_layer[layer_idx]  # Expected Shape: (seq_len, d_model)

        if not isinstance(layer_activations, torch.Tensor):
            print(
                f"Warning: Activations for layer {layer_idx} (YES) are not a tensor (type: {type(layer_activations)}). Skipping."
            )
            yes_logits_by_layer.append(float("nan"))
            continue

        if (
            layer_activations.ndim == 2 and layer_activations.shape[0] > 0
        ):  # Ensure (seq_len > 0, d_model)
            h_last_prompt_token = layer_activations[seq_position_to_analyze, :]  # Shape: (d_model)
            logit_val_yes = (
                h_last_prompt_token.to(unembed_vector_yes.device) @ unembed_vector_yes
            )
            yes_logits_by_layer.append(logit_val_yes.item())
        else:
            print(
                f"Warning: Layer {layer_idx} (YES) activations have unexpected shape or size: {layer_activations.shape}. Skipping."
            )
            yes_logits_by_layer.append(float("nan"))

    # ---- Analysis for "NO" token ----

    print(
        f"Using token ID for NO analysis: {main_no_token_id} (corresponds to '{tokenizer.decode([main_no_token_id])}')"
    )
    decoded_token_for_plot_no = tokenizer.decode([main_no_token_id])

    # 2. Get unembedding vector for "NO"
    unembed_vector_no = W_U[:, main_no_token_id].clone().detach().cpu()  # Shape: (hidden_dim,)

    # 3. Calculate logit for "NO" at each layer's final position residual stream
    no_logits_by_layer = []

    for layer_idx in layer_numbers:  # Re-use the same layer_numbers from YES analysis
        if layer_idx not in acts_by_layer:
            print(
                f"Warning: Layer {layer_idx} not found in acts_by_layer. Skipping for NO."
            )
            no_logits_by_layer.append(float("nan"))
            continue

        layer_activations = acts_by_layer[layer_idx]

        if not isinstance(layer_activations, torch.Tensor):
            print(
                f"Warning: Activations for layer {layer_idx} (NO) are not a tensor (type: {type(layer_activations)}). Skipping."
            )
            no_logits_by_layer.append(float("nan"))
            continue

        if layer_activations.ndim == 2 and layer_activations.shape[0] > 0:
            h_last_prompt_token = layer_activations[seq_position_to_analyze, :]
            logit_val_no = (
                h_last_prompt_token.to(unembed_vector_no.device) @ unembed_vector_no
            )
            no_logits_by_layer.append(logit_val_no.item())
        else:
            print(
                f"Warning: Layer {layer_idx} (NO) activations have unexpected shape or size: {layer_activations.shape}. Skipping."
            )
            no_logits_by_layer.append(float("nan"))

    # ---- Combined Plot (Optional) ----

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(
        layer_numbers,
        yes_logits_by_layer,
        marker="o",
        linestyle="-",
        color="green",
        label=f"'{decoded_token_for_plot_yes}' (ID: {main_yes_token_id})",
    )
    ax.plot(
        layer_numbers,
        no_logits_by_layer,
        marker="s",
        linestyle="--",
        color="red",
        label=f"'{decoded_token_for_plot_no}' (ID: {main_no_token_id})",
    )

    title_text = f"Logit Trajectory for YES vs NO at {seq_position_to_analyze_str} for response {response_id[:8]}"
    title_text += f"\nExpected Answer: {expected_answer}"
    if is_answer_flipping_response:
        title_text += f". Answer Flipping Detected"
    else:
        title_text += f". No Answer Flipping Detected"
    ax.set_title(title_text, fontsize=16)

    # Add prompt and response below the title
    # Figure out a good y position for the text - start just below the title
    # The title's y position is typically 1.0, but we use tight_layout later,
    # so we need to be careful with absolute figure coordinates.
    # We'll add text and then adjust subplot parameters.
    
    # Construct the text string
    full_text_to_display = f"Prompt:\n{prompt}\nResponse:\n{response_str}"

    # Add the text using figtext. Adjust y_start as needed.
    # These are figure coordinates (0 to 1).
    # (x, y, s, fontdict=None, **kwargs)
    # We set y to be just below where the title usually is, and allow wrapping.
    plt.figtext(0.05, -0.01, full_text_to_display, ha="left", va="top", fontsize=12, wrap=True, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=1))

    ax.set_xlabel("Layer Number (0=Embeddings, 1 to N=Transformer Blocks)", fontsize=12)
    ax.set_ylabel("Logit Value", fontsize=12)
    ax.set_xticks(ticks=layer_numbers, labels=[str(l) for l in layer_numbers])
    max_y = max(max(yes_logits_by_layer), max(no_logits_by_layer))
    min_y = min(min(yes_logits_by_layer), min(no_logits_by_layer))
    ax.set_yticks(ticks=np.arange(min_y, max_y, 0.1))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout(rect=(0, 0, 1, 0.94)) # Adjust rect to leave space for title and fig

    plt.savefig(f"logit_trajectory_token_before_final_answer_{model_id.split('/')[-1]}_qid_{qid[:8]}_response_{response_id[:8]}.png", dpi=300)
    plt.show()


# %%

for response_id in all_response_ids:
    make_plot_logit_trajectory_token_before_final_answer(response_id)
# %%
