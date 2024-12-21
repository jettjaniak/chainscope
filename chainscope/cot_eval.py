import logging

from chainscope.typing import *


def evaluate_cot_response(response: str) -> Literal["YES", "NO", "UNKNOWN"]:
    """Evaluate a chain-of-thought response to determine if the answer is YES, NO, or UNKNOWN.
    Currently uses a simple heuristic: if there is exactly one YES or NO in the response,
    that's the answer. Otherwise, UNKNOWN.
    """

    confounding_phrases = [
        "A clear YES or NO question!",
        "Provide a YES or NO answer",
        "Give a YES or NO answer",
        "Finally, I'll conclude with a YES or NO answer",
        "NOT",
    ]
    for phrase in confounding_phrases:
        response = response.replace(phrase, "")

    answer_phrases = [
        "Answer: {answer}",
        "**Answer:** {answer}",
        "**Answer**: {answer}",
        "answer is: {answer}",
        "answer is: **{answer}**",
        "Answer:\n**{answer}**",
        "Conclusion: {answer}",
        "**Conclusion:** {answer}",
        "**Conclusion**: {answer}",
        "Conclusion:\n{answer}",
        "**Conclusion:**\n{answer}",
        "**Conclusion:**  {answer}",
    ]
    yes_answer_phrases = [
        phrase.format(answer=a) for phrase in answer_phrases for a in ["YES", "Yes"]
    ] + ["I would say yes", "we conclude that yes", "Considering all factors:\n\nYes"]
    no_answer_phrases = [
        phrase.format(answer=a) for phrase in answer_phrases for a in ["NO", "No"]
    ]

    # Convert to uppercase to make matching case-insensitive
    yes_count = response.count("YES")
    no_count = response.count("NO")

    if yes_count > 0 and no_count == 0:
        # logging.info(f"YES CoT response:\n#####\n`{response}`\n#####")
        return "YES"
    elif no_count > 0 and yes_count == 0:
        # logging.info(f"NO CoT response:\n#####\n`{response}`\n#####")
        return "NO"
    elif any(phrase in response for phrase in yes_answer_phrases):
        return "YES"
    elif any(phrase in response for phrase in no_answer_phrases):
        return "NO"
    elif response.startswith("No"):
        return "NO"
    else:
        logging.info(f"Unknown CoT response:\n#####\n`{response}`\n#####")
        return "UNKNOWN"


def evaluate_cot_responses(responses: CotResponses) -> CotEval:
    """Evaluate all CoT responses for a given model and instruction set."""
    results = {}
    unknown_count = 0
    total_count = 0

    for qid, response_by_uuid in responses.responses_by_qid.items():
        results[qid] = {}
        for uuid, response in response_by_uuid.items():
            result = evaluate_cot_response(response)
            results[qid][uuid] = result
            total_count += 1
            if result == "UNKNOWN":
                unknown_count += 1

    if total_count > 0 and (unknown_count / total_count) > 0.1:
        logging.warning(
            f"{unknown_count}/{total_count} ({unknown_count/total_count:.1%}) responses "
            f"were classified as UNKNOWN for model {responses.model_id}"
        )

    return CotEval(
        results_by_qid=results,
        model_id=responses.model_id,
        instr_id=responses.instr_id,
        dataset_id=responses.dataset_id,
        sampling_params=responses.sampling_params,
    )
