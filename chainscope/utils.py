import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# for convenience, you can use any model id directly
MODELS_MAP = {
    # Gemma models
    "G2": "google/gemma-2-2b-it",
    "G9": "google/gemma-2-9b-it",
    "G27": "google/gemma-2-27b-it",
    # Llama models
    "L1": "meta-llama/Llama-3.2-1B-Instruct",
    "L3": "meta-llama/Llama-3.2-3B-Instruct",
    "L8": "meta-llama/Llama-3.1-8B-Instruct",
    "L70": "meta-llama/Llama-3.3-70B-Instruct",
    # Phi models
    "P": "microsoft/Phi-3.5-mini-instruct",
    # Qwen models
    "Q0.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "Q1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "Q3": "Qwen/Qwen2.5-3B-Instruct",
    "Q7": "Qwen/Qwen2.5-7B-Instruct",
    "Q14": "Qwen/Qwen2.5-14B-Instruct",
    "Q32": "Qwen/Qwen2.5-32B-Instruct",
    "Q72": "Qwen/Qwen2.5-72B-Instruct",
}


def load_model_and_tokenizer(
    model_id: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    device = get_model_device(model)

    # get rid of the warnings early
    model(torch.tensor([[tokenizer.eos_token_id]]).to(device))
    model.generate(
        torch.tensor([[tokenizer.eos_token_id]]).to(device),
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    return model, tokenizer


def remove_llama_system_dates(chat_input_str: str) -> str:
    # TODO: this was done for llama 3.2, does it work for other models?
    return re.sub(
        r"\n\nCutting Knowledge Date: .*\nToday Date: .*\n\n", "", chat_input_str
    )


def conversation_to_str_prompt(
    conversation: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase
) -> str:
    str_prompt = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    assert isinstance(str_prompt, str)
    return remove_llama_system_dates(str_prompt)


def make_chat_prompt(instruction: str, tokenizer: PreTrainedTokenizerBase) -> str:
    conversation = [
        {
            "role": "user",
            "content": instruction,
        }
    ]
    return conversation_to_str_prompt(conversation, tokenizer)


def get_model_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device
