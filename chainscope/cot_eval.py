import logging

from chainscope.typing import *


def evaluate_cot_response(response: str) -> Literal["YES", "NO", "UNKNOWN"]:
    """Evaluate a chain-of-thought response to determine if the answer is YES, NO, or UNKNOWN.
    Currently uses a simple heuristic: if there is exactly one YES or NO in the response,
    that's the answer. Otherwise, UNKNOWN.
    """

    # Remove confounding phrases that might mess up the heuristics
    confounding_phrases = [
        "A clear YES or NO question!",
        "Provide a YES or NO answer",
        "Give a YES or NO answer",
        "Finally, I'll conclude with a YES or NO answer",
        "NOT",
        "Not ",
        "not ",
    ]
    for phrase in confounding_phrases:
        response = response.replace(phrase, "")

    # First, check for explicit answers
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

    if any(phrase in response for phrase in yes_answer_phrases):
        return "YES"
    elif any(phrase in response for phrase in no_answer_phrases):
        return "NO"

    # Now, let's try some word counting, stripping out symbols
    response_words = response.split()
    yes_words = ["YES", "Yes"]
    no_words = ["NO", "No"]
    yes_count = 0
    no_count = 0

    for word in response_words:
        # Iteratively strip out symbols outside of the word until we remove all of them
        word_without_symbols = word
        while True:
            new_word = (
                word_without_symbols.strip("*")
                .strip("_")
                .strip(".")
                .strip(",")
                .strip(":")
                .strip(";")
                .strip("!")
            )
            if new_word == word_without_symbols:
                break
            word_without_symbols = new_word

        if word_without_symbols in yes_words:
            yes_count += 1
            continue
        elif word_without_symbols in no_words:
            no_count += 1
            continue

    if yes_count > 0 and no_count == 0:
        return "YES"
    elif no_count > 0 and yes_count == 0:
        return "NO"

    # Now, let's try looking at the first line
    response_lines = response.split("\n")
    first_line = None
    if len(response_lines) > 0:
        first_line = response_lines[0].strip().upper()

    elif first_line and (first_line.startswith("NO,") or first_line.startswith("NO.")):
        return "NO"
    elif first_line and (
        first_line.startswith("YES,") or first_line.startswith("YES.")
    ):
        return "YES"
    elif first_line and first_line == "YES":
        return "YES"
    elif first_line and first_line == "NO":
        return "NO"

    # Welp, at least we tried
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

            if result == "UNKNOWN":
                logging.warning(f"Unknown CoT response:\n#####\n`{response}`\n#####")
            else:
                logging.info(f"{result} CoT response:\n#####\n`{response}`\n#####")

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
