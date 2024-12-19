import yaml

from chainscope import DATA_DIR


def load_prompt(prompt_id: str) -> str:
    prompts_path = DATA_DIR / "prompts.yaml"
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts[prompt_id]
