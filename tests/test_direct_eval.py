from chainscope.direct_eval import evaluate_direct, get_direct_probs
from chainscope.typing import *
from tests.utils import *


def test_get_direct_probs(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    instructions = Instructions.load("instr-v0")
    # this is not a kind of question we're using, just a test
    question = "Is the sky blue?"
    probs = get_direct_probs(
        model=model,
        tokenizer=tokenizer,
        question_str=question,
        direct_instruction=instructions.direct,
    )
    assert 0 <= probs.p_yes <= 1
    assert 0 <= probs.p_no <= 1
    assert abs(probs.p_yes + probs.p_no - 1.0) < 1e-6


def test_evaluate_direct(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    qs_dataset = QsDataset.load("aircraft-speeds_gt_NO_1_tests")

    eval_result = evaluate_direct(
        model=model,
        tokenizer=tokenizer,
        question_dataset=qs_dataset,
        instr_id="instr-v0",
    )
    assert eval_result.probs_by_qid.keys() == qs_dataset.question_by_qid.keys()
