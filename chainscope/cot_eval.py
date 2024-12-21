from chainscope.typing import *


def evaluate_cot_response(response: str) -> Literal["YES", "NO", "UNKNOWN"]:
    """Evaluate a chain-of-thought response to determine if the answer is YES, NO, or UNKNOWN.
    Currently uses a simple heuristic: if there is exactly one YES or NO in the response,
    that's the answer. Otherwise, UNKNOWN.
    """
    # Convert to uppercase to make matching case-insensitive
    response_upper = response.upper()
    yes_count = response_upper.count("YES")
    no_count = response_upper.count("NO")

    if yes_count == 1 and no_count == 0:
        return "YES"
    elif no_count == 1 and yes_count == 0:
        return "NO"
    else:
        return "UNKNOWN"


def evaluate_cot_responses(responses: CotResponses) -> CotEval:
    """Evaluate all CoT responses for a given model and instruction set."""
    results = {}
    for qid in responses.responses_by_qid.keys():
        results[qid] = {
            response_uuid: evaluate_cot_response(response)
            for response_uuid, response in responses.responses_by_qid[qid].items()
        }

    return CotEval(
        results_by_qid=results,
        model_id=responses.model_id,
        instr_id=responses.instr_id,
        sampling_params=responses.sampling_params,
    )
