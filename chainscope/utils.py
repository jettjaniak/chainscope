import random
import re

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase)

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
    # "Q0.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "Q1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "Q3": "Qwen/Qwen2.5-3B-Instruct",
    "Q7": "Qwen/Qwen2.5-7B-Instruct",
    "Q14": "Qwen/Qwen2.5-14B-Instruct",
    "Q32": "Qwen/Qwen2.5-32B-Instruct",
    "Q72": "Qwen/Qwen2.5-72B-Instruct",
    "QwQ": "qwen/qwq-32b",
    # OpenRouter models
    # OpenAI
    # "GPT3.5": "openai/gpt-3.5-turbo",
    "GPT4OM": "openai/gpt-4o-mini",
    "GPT4O": "openai/gpt-4o-2024-08-06",
    "GPT4OL": "openai/chatgpt-4o-latest",
    "GPT4.5": "openai/gpt-4.5-preview",
    "O1M": "openai/o1-mini",
    "O1P": "openai/o1-preview",
    "O1": "openai/o1",
    # Gemini
    "GF1.5": "google/gemini-flash-1.5-8b",
    "GF2": "google/gemini-2.0-flash-exp:free",
    "GF2T": "google/gemini-2.0-flash-thinking-exp:free",
    "GF2.5": "google/gemini-2.5-flash-preview",
    "GF2.5T": "google/gemini-2.5-flash-preview:thinking",
    "GP1.5": "google/gemini-pro-1.5",
    "GP2.5": "google/gemini-2.5-pro-preview",
    # Anthropic
    "C3H": "anthropic/claude-3-haiku",
    "C3.5H": "anthropic/claude-3.5-haiku",
    "C3S": "anthropic/claude-3-sonnet",
    "C3.5S": "anthropic/claude-3.5-sonnet",
    "C3.6S": "anthropic/claude-3.6-sonnet",
    "C3O": "anthropic/claude-3-opus",
    "C3.7S": "anthropic/claude-3.7-sonnet",  # No thinking
    "C3.7S_1K": "anthropic/claude-3.7-sonnet_1k",  # Thinking with 1k budget
    "C3.7S_10K": "anthropic/claude-3.7-sonnet_10k",  # Thinking with 10k budget
    "C3.7S_32K": "anthropic/claude-3.7-sonnet_32k",  # Thinking with 32k budget
    "C3.7S_64K": "anthropic/claude-3.7-sonnet_64k",  # Thinking with 64k budget
    # xAI
    "GK2": "x-ai/grok-2-1212",
    "GK2M": "x-ai/grok-2-mini",
    # DeepSeek
    "DSV3": "deepseek/deepseek-chat",  # DeepkSeek-V3
    "DSR1": "deepseek/deepseek-r1",  # DeepSeek-R1
}

CLOSED_SOURCE_MODELS = [
    k
    for k, v in MODELS_MAP.items()
    if not any(part in v.lower() for part in ["gemma", "llama", "phi", "qwen"])
]


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
    try:
        model.generate(
            torch.tensor([[tokenizer.eos_token_id]]).to(device),
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    except RuntimeError:
        pass

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


def get_param_count(model_name: str) -> float:
    """Extract parameter count from model name in billions using regex."""
    name_lower = model_name.lower()
    match = re.search(r"[-]?(\d+\.?\d*)b", name_lower)
    return float(match.group(1)) if match else float("inf")


def get_model_family(model_id: str) -> str:
    """Extract the family from a model ID."""
    return model_id.split("/")[0]


def get_model_display_name(model_id: str) -> str:
    """Extract the display name from a model ID."""
    return model_id.split("/")[-1]


def get_claude_rank(model_name: str) -> int:
    """Get sorting rank for Claude-3 models."""
    name_lower = model_name.lower()
    if "claude-3" not in name_lower:
        return 999999

    # First priority: version number (3.0 before 3.5)
    base_rank = 0 if "claude-3.5" not in name_lower else 10

    # Second priority: model size (haiku -> sonnet -> opus)
    if "haiku" in name_lower:
        size_rank = 0
    elif "sonnet" in name_lower:
        size_rank = 1
    elif "opus" in name_lower:
        size_rank = 2
    else:
        return 999999

    return base_rank + size_rank


def get_openai_rank(model_name: str) -> int:
    """Get sorting rank for OpenAI models."""
    name_lower = model_name.lower()
    # Check for model name patterns instead of provider name
    if not any(x in name_lower for x in ["gpt-4", "o1"]):
        return 999999

    # First priority: model family (multiply by 100 to ensure it dominates)
    if "gpt-4" in name_lower:
        base_rank = 0 * 100
    elif "o1" in name_lower:
        base_rank = 1 * 100
    else:
        return 999999

    # Second priority: model size
    if "mini" in name_lower:
        size_rank = 0
    elif any(x in name_lower for x in ["preview", ""]):  # base model
        size_rank = 1
    else:
        return 999999

    return base_rank + size_rank


def sort_models(model_ids: list[str]) -> list[str]:
    """Sort model IDs by name prefix and parameter count."""

    def get_sort_prefix(model_id: str) -> str:
        name = get_model_display_name(model_id).split("-")[0].lower()
        # Treat gemma and gemini as the same prefix
        if name in ["gemma", "gemini"]:
            return "gem"
        # Group all OpenAI models together
        if "openai" in model_id.lower():
            return "openai"
        return name

    def get_debug_key(x: str) -> tuple:
        prefix = get_sort_prefix(x)
        claude = get_claude_rank(get_model_display_name(x))
        openai = get_openai_rank(get_model_display_name(x))
        params = get_param_count(get_model_display_name(x))
        return (prefix, claude, openai, params)

    return sorted(model_ids, key=get_debug_key)


def remove_all_symbols(text: str) -> str:
    """Remove all symbols and special characters from text, keeping only alphanumeric characters.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text with only alphanumeric characters and spaces
    """
    return "".join(char for char in text if char.isalnum())


def setup_determinism(seed: int):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit_hash() -> str:
    """Returns the current git commit hash."""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "git_hash_not_found"


def is_instruct_model(model_id: str) -> bool:
    model_id = MODELS_MAP.get(model_id, model_id)
    lower_model_id = model_id.lower()
    return (
        "instruct" in lower_model_id
        or "-it" in lower_model_id
        or "google" in lower_model_id
        or "anthropic" in lower_model_id
        or "deepseek" in lower_model_id
        or "openai" in lower_model_id
        or "x-ai" in lower_model_id
    )
