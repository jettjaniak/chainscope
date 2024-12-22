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
    assert evaluate_cot_response("NO\nTherefore, the answer is YES") == "UNKNOWN"

    # Test for specific cases found in the wild
    response = """Yes. The F-4 Phantom II was designed to operate on the North American Aerospace Defense Command (NORAD) radar system, which had been upgraded from the earlier F-16 Fighting Falcon in 1980. The MiG-29 is an aircraft used primarily for reconnaissance and aerial refueling missions. Therefore, the F-4 Phantom II would likely be faster than the MiG-29 due to its superior capabilities and operational requirements."""
    assert evaluate_cot_response(response) == "YES"

    detailed_response = """To determine if the P-51 Mustang and the Airbus A320 are faster than each other, we need to consider several factors such as weight, fuel efficiency, payload capacity, aerodynamics, and technological advancements in both aircraft.

**Step 1: Compare Weight**
The P-51 Mustang weighs around 48 tons (approximately 47 metric tonnes), while the Airbus A320 weighs about 36.7 tons (around 35 metric tonnes). Since the P-51 weighs more than the A320, it implies that the P-51 may be slower due to its heavier weight.

**Step 2: Fuel Efficiency**
Both aircraft have similar engines, but the A320 has been refined significantly for longer range and higher speed. The P-51's engine might not offer the same performance capabilities in terms of power output and thrust as an A320.

**Step 3: Payload Capacity**
Both aircraft can carry significant loads, but the A320 usually carries more cargo due to its larger fuselage and seating arrangements. This means that the A320 would typically carry more passengers or supplies compared to the P-51.

**Step 4: Aerodynamic Performance**
Both aircraft have similar wings, but the A320 has improved wing design features like swept-back wings and tail fins. These enhancements might reduce drag and improve efficiency.

**Step 5: Technology Advancements**
While both aircraft use advanced technologies, the A320 has seen significant improvements in terms of automation, digital systems, and operational efficiencies. The P-51 has older, less sophisticated technology.

Given these points, comparing the speeds of the two aircraft directly is challenging because their weights and various parameters differ. However, considering the differences in weight, fuel efficiency, payload capacity, aerodynamics, and technological advancements, the P-51 Mustang appears to be generally faster than the Airbus A320.

**Conclusion:**
Based on this analysis, I believe the answer is **YES**, the P-51 Mustang is generally faster than the Airbus A320."""
    assert evaluate_cot_response(detailed_response) == "YES"


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
