import random
import re

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODELS_MAP = {
    # Gemma models
    "G": "google/gemma-2-2b-it",
    "G2": "google/gemma-2-2b-it",
    "G9": "google/gemma-2-9b-it",
    "G27": "google/gemma-2-27b-it",
    # Llama models
    "L": "meta-llama/Llama-3.2-3B-Instruct",
    "L3": "meta-llama/Llama-3.2-3B-Instruct",
    "L8": "meta-llama/Llama-3.1-8B-Instruct",
    "L70": "meta-llama/Llama-3.3-70B-Instruct",
    # Phi models
    "P": "microsoft/Phi-3.5-mini-instruct",
    # Qwen models
    "Q": "Qwen/Qwen2.5-3B-Instruct",
    "Q3": "Qwen/Qwen2.5-3B-Instruct",
    "Q14": "Qwen/Qwen2.5-14B-Instruct",
    "Q32": "Qwen/Qwen2.5-32B-Instruct",
    "Q72": "Qwen/Qwen2.5-72B-Instruct",
}


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_id)


def load_model_and_tokenizer(
    model_id: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model_id = MODELS_MAP.get(model_id, model_id)

    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )

    # get rid of the warnings early
    model(torch.tensor([[tokenizer.bos_token_id]]).cuda())
    model.generate(
        torch.tensor([[tokenizer.bos_token_id]]).cuda(),
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    return model, tokenizer


def is_chat_model(model_id: str) -> bool:
    """Determine if model is a chat model based on its ID."""
    model_id = model_id.lower()
    return any(x in model_id for x in ["instruct", "-it"])


def setup_determinism(seed: int):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def remove_llama_system_dates(chat_input_str: str) -> str:
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
