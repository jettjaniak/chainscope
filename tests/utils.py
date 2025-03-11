import pytest

from chainscope.utils import load_model_and_tokenizer


@pytest.fixture
def small_model_and_tokenizer():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    return load_model_and_tokenizer(model_id)
