from chainscope.cot_generation import get_question_cot_responses_local
from chainscope.typing import *
from tests.utils import *


@pytest.fixture
def sampling_params():
    return SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=2,
    )


def test_get_question_cot_responses(small_model_and_tokenizer, sampling_params):
    model, tokenizer = small_model_and_tokenizer
    instructions = Instructions.load("instr-v0")
    question = "Is the speed of a Boeing 747 greater than the speed of a Cessna 172?"

    responses = get_question_cot_responses_local(
        model=model,
        tokenizer=tokenizer,
        question_str=question,
        cot_instruction=instructions.cot,
        sampling_params=sampling_params,
        n_responses=2,
    )
    assert len(responses) == 2
