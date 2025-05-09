from functools import partial

import torch
from jaxtyping import Float
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chainscope.typing import *
from chainscope.utils import get_model_device, make_chat_prompt


def layer_to_hook_point(layer: int):
    if layer == 0:
        return "model.embed_tokens"
    return f"model.layers.{layer-1}"


def hook_point_to_layer(hook_point: str):
    if hook_point == "model.embed_tokens":
        return 0
    return int(hook_point.split(".")[-1]) + 1


def abstract_hook_fn(
    module,
    input,
    output,
    layer: int,
    locs_to_cache: dict[str, int],
    resid_by_loc_by_layer: dict[int, dict[str, Float[torch.Tensor, "model"]]],
):
    if isinstance(output, tuple):
        output = output[0]
    assert (
        len(output.shape) == 3
    ), f"Expected tensor of shape (1, seq_len, d_model), got {output.shape}"
    # we're running batch size 1
    output = output[0]

    if layer not in resid_by_loc_by_layer:
        resid_by_loc_by_layer[layer] = {}
    # For each location we want to cache
    for loc_name, loc_idx in locs_to_cache.items():
        resid_by_loc_by_layer[layer][loc_name] = output[loc_idx].cpu()


ResidByLocByLayer = dict[int, dict[str, Float[torch.Tensor, "model"]]]


def collect_resid_acts(
    model: PreTrainedModel,
    input_ids: list[int],
    layers: list[int],
    locs_to_cache: dict[str, int],
) -> ResidByLocByLayer:
    """
    Args:
        locs_to_cache: Map from location names to sequence indices to cache
    Returns:
        Nested dict: layer -> location_name -> activation tensor
    """
    device = get_model_device(model)
    resid_by_loc_by_layer = {}
    hooks = []
    hook_points = set(layer_to_hook_point(i) for i in layers)
    hook_points_cnt = len(hook_points)

    for name, module in model.named_modules():
        if name in hook_points:
            hook_points_cnt -= 1
            layer = hook_point_to_layer(name)
            hook_fn = partial(
                abstract_hook_fn,
                layer=layer,
                locs_to_cache=locs_to_cache,
                resid_by_loc_by_layer=resid_by_loc_by_layer,
            )
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    assert hook_points_cnt == 0
    try:
        with torch.inference_mode():
            model(torch.tensor([input_ids]).to(device))
    finally:
        for hook in hooks:
            hook.remove()

    return resid_by_loc_by_layer


def prepare_prompt_locs(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> tuple[list[int], dict[str, int]]:
    # Prepare input
    chat_input = make_chat_prompt(
        instruction=prompt,
        tokenizer=tokenizer,
    )
    input_ids = tokenizer.encode(chat_input, add_special_tokens=False)

    def filter_locs(substring: str):
        ret = [
            i
            for i, token in enumerate(input_ids)
            if substring in tokenizer.decode(token)
        ]
        assert len(ret) >= 1
        return ret

    colon_idx = filter_locs(":")[0]
    qmark_idx = filter_locs("?")[-1]
    reasoning_idx = filter_locs("reasoning")[-1]
    yes_idx = filter_locs("YES")[-1]
    no_idx = filter_locs("NO")[-1]
    answer_idx = filter_locs("answer")[-1]

    locs_to_cache = {
        "colon": colon_idx,
        "colon+1": colon_idx + 1,
        "qmark": qmark_idx,
        "qmark+1": qmark_idx + 1,
        "reasoning": reasoning_idx,
        "reasoning+1": reasoning_idx + 1,
        "yes": yes_idx,
        "no": no_idx,
        "answer": answer_idx,
        "last_prompt": len(input_ids) - 2,
        "turn": len(input_ids) - 1,
    }
    # colon colon+1 qmark qmark+1 reasoning reasoning+1 yes no answer last_prompt turn

    assert list(locs_to_cache.values()) == sorted(set(locs_to_cache.values()))
    return input_ids, locs_to_cache


def collect_questions_acts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_id: str,
    instr_id: str,
    layers: list[int],
    test: bool,
) -> dict[str, ResidByLocByLayer]:
    if dataset_id.startswith("wm-"):
        assert instr_id == "instr-wm"
    instr = Instructions.load(instr_id).cot
    qs_dataset = QsDataset.load(dataset_id)

    # First prepare all inputs and verify locations
    prepared_inputs = {}
    q_by_qid = qs_dataset.question_by_qid.items()
    if test:
        q_by_qid = list(q_by_qid)[:10]
    for qid, qs in q_by_qid:
        prompt = instr.format(question=qs.q_str)
        input_ids, locs_to_cache = prepare_prompt_locs(tokenizer, prompt)
        prepared_inputs[qid] = (input_ids, locs_to_cache)

    # Then collect activations for all questions
    ret = {}
    for qid, (input_ids, locs_to_cache) in tqdm(
        prepared_inputs.items(), desc="Collecting activations"
    ):
        ret[qid] = collect_resid_acts(model, input_ids, layers, locs_to_cache)
    return ret
