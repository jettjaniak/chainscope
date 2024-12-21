from chainscope.cot_eval import evaluate_cot_response, evaluate_cot_responses
from chainscope.typing import *


def test_evaluate_cot_response():
    # Test single YES case
    assert evaluate_cot_response("Here's my reasoning... Therefore: YES") == "YES"
    assert evaluate_cot_response("Let me think... The answer is YES.") == "YES"

    # Test single NO case
    assert evaluate_cot_response("After analysis... NO.") == "NO"
    assert evaluate_cot_response("Based on the facts... The answer is NO") == "NO"

    # Test UNKNOWN cases
    assert evaluate_cot_response("It could be YES or NO") == "UNKNOWN"
    assert evaluate_cot_response("YES... but also NO") == "UNKNOWN"
    assert evaluate_cot_response("Not enough information") == "UNKNOWN"


def test_evaluate_cot_responses():
    # Load actual responses from YAML file
    dataset_id = "aircraft-speeds_gt_NO_1_377c39d3"
    instr_id = "instr-v0"
    responses_path = (
        DATA_DIR
        / "cot_responses"
        / instr_id
        / "T0.7_P0.9_M2000"
        / dataset_id
        / "google__gemma-2-2b-it.yaml"
    )
    cot_responses = CotResponses.load(responses_path)

    # Evaluate responses
    eval_result = evaluate_cot_responses(cot_responses)

    # Check structure and content
    assert isinstance(eval_result, CotEval)
    assert eval_result.model_id == "google/gemma-2-2b-it"
    assert eval_result.instr_id == instr_id
    assert eval_result.dataset_id == dataset_id
    assert eval_result.sampling_params == cot_responses.sampling_params

    # Check that all questions and UUIDs are preserved
    assert eval_result.results_by_qid.keys() == cot_responses.responses_by_qid.keys()
    for qid in eval_result.results_by_qid:
        assert (
            eval_result.results_by_qid[qid].keys()
            == cot_responses.responses_by_qid[qid].keys()
        )

    # Check specific results from the file
    yes_response = eval_result.results_by_qid[
        "03d05168fa0d7977b77f6ac96c717324a14e71d97b156881eb08e154be5ca117"
    ]["072385f2-9234-4dcf-af9a-f19733527b54"]
    assert yes_response == "YES"

    unknown_response = eval_result.results_by_qid[
        "03d05168fa0d7977b77f6ac96c717324a14e71d97b156881eb08e154be5ca117"
    ]["4b5f521d-a033-443c-85b8-bd1138795260"]
    assert unknown_response == "UNKNOWN"

    no_response = eval_result.results_by_qid[
        "3569bb31a722844dcea8919e9f2cb1786ce22e7e129cc488d4d2781fcd0a2271"
    ]["0904b0f3-1a53-4e5b-b720-f41ebf4fb4bc"]
    assert no_response == "NO"
