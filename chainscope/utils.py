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
    # "Q0.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "Q1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "Q3": "Qwen/Qwen2.5-3B-Instruct",
    "Q7": "Qwen/Qwen2.5-7B-Instruct",
    "Q14": "Qwen/Qwen2.5-14B-Instruct",
    "Q32": "Qwen/Qwen2.5-32B-Instruct",
    "Q72": "Qwen/Qwen2.5-72B-Instruct",
    # OpenRouter models
    # OpenAI
    # "GPT3.5": "openai/gpt-3.5-turbo",
    "GPT4OM": "openai/gpt-4o-mini",
    "GPT4O": "openai/gpt-4o",
    "O1M": "openai/o1-mini",
    "O1": "openai/o1",
    # Gemini
    "GF1.5": "google/gemini-flash-1.5-8b",
    "GF2": "google/gemini-2.0-flash-exp:free",
    "GF2T": "google/gemini-2.0-flash-thinking-exp:free",
    # Anthropic
    "C3H": "anthropic/claude-3-haiku",
    "C3.5H": "anthropic/claude-3.5-haiku",
    "C3S": "anthropic/claude-3-sonnet",
    "C3.5S": "anthropic/claude-3.5-sonnet",
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


def get_param_count(model_name: str) -> float:
    """Extract parameter count from model name in billions using regex."""
    name_lower = model_name.lower()
    match = re.search(r"[-]?(\d+\.?\d*)b", name_lower)
    return float(match.group(1)) if match else float("inf")


def get_model_display_name(model_id: str) -> str:
    """Extract the display name from a model ID."""
    return model_id.split("/")[-1]


def sort_models(model_ids: list[str]) -> list[str]:
    """Sort model IDs by name prefix and parameter count."""
    return sorted(
        model_ids,
        key=lambda x: (
            get_model_display_name(x).split("-")[0].lower(),
            get_param_count(get_model_display_name(x)),
        ),
    )
