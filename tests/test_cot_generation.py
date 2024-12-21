from chainscope.cot_generation import get_all_cot_responses, get_question_cot_responses
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

    responses = get_question_cot_responses(
        model=model,
        tokenizer=tokenizer,
        question_str=question,
        cot_instruction=instructions.cot,
        sampling_params=sampling_params,
        n_responses=2,
    )
    assert len(responses) == 2


def test_get_all_cot_responses(small_model_and_tokenizer, sampling_params):
    model, tokenizer = small_model_and_tokenizer
    dataset_id = "aircraft-speeds_gt_NO_1_tests"
    cot_responses = get_all_cot_responses(
        model=model,
        tokenizer=tokenizer,
        dataset_id=dataset_id,
        instr_id="instr-v0",
        sampling_params=sampling_params,
        n_responses=2,
    )
    assert isinstance(cot_responses, CotResponses)
    question_dataset = QsDataset.load(dataset_id)
    assert (
        cot_responses.responses_by_qid.keys() == question_dataset.question_by_qid.keys()
    )
    assert all(
        len(responses) == 2 for responses in cot_responses.responses_by_qid.values()
    )
    assert cot_responses.model_id == model.name_or_path
    assert cot_responses.instr_id == "instr-v0"
    assert cot_responses.sampling_params == sampling_params
