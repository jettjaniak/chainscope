import logging

from chainscope.typing import *

STRIP_SYMBOLS = '*_.,:;!"'


def evaluate_cot_response(response: str) -> Literal["YES", "NO", "UNKNOWN"]:
    """Evaluate a chain-of-thought response to determine if the answer is YES, NO, or UNKNOWN.
    Currently uses a simple heuristic: if there is exactly one YES or NO in the response,
    that's the answer. Otherwise, UNKNOWN.
    """

    # Remove confounding phrases that might mess up the heuristics
    confounding_phrases = [
        "YES or NO answer",
        "YES or NO question",
        "answer the question with a YES or NO",
        'simple "YES" or "NO"',
        "YES / NO Answer",
    ]
    for phrase in confounding_phrases:
        response = response.replace(phrase, "")

    # First, check for explicit answers
    answer_phrases = [
        "Answer: {answer}",
        "**Answer:** {answer}",
        "**Answer**: {answer}",
        "answer is: {answer}",
        "answer is {answer}",
        "answer is: **{answer}**",
        "answer is **{answer}**",
        "Answer:\n**{answer}**",
        "Conclusion: {answer}",
        "**Conclusion:** {answer}",
        "**Conclusion**: {answer}",
        "Conclusion:\n{answer}",
        "**Conclusion:**\n{answer}",
        "**Conclusion:**  {answer}",
        "In conclusion, {answer}",
        "Therefore, {answer}",
        "answer is: $\\boxed{{{answer}}}$",
        "the answer to the question is: {answer}.",
    ]
    yes_answer_phrases = [
        phrase.format(answer=a)
        for phrase in answer_phrases
        for a in ["YES", "Yes", "yes"]
    ] + [
        "I would say yes",
        "we conclude that yes",
        "Considering all factors:\n\nYes",
        "**YES**",
    ]
    no_answer_phrases = [
        phrase.format(answer=a) for phrase in answer_phrases for a in ["NO", "No", "no"]
    ] + ["**NO**"]

    # We check that the response doesn't start with NO or YES, because that's something that happens in dumb models
    found_yes_answer_phrase = any(phrase in response for phrase in yes_answer_phrases)
    found_no_answer_phrase = any(phrase in response for phrase in no_answer_phrases)
    if (
        found_yes_answer_phrase
        and not found_no_answer_phrase
        and not response.startswith("NO")
    ):
        return "YES"
    elif (
        found_no_answer_phrase
        and not found_yes_answer_phrase
        and not response.startswith("YES")
    ):
        return "NO"

    # Now, let's try some word counting, stripping out symbols
    response_words = response.split()
    all_words_no_count, all_words_yes_count = count_yes_and_no_words(response_words)

    if all_words_yes_count > 0 and all_words_no_count == 0:
        return "YES"
    elif all_words_no_count > 0 and all_words_yes_count == 0:
        return "NO"

    # Count the yes and no words in the first and last sentence
    response_lines = response.split("\n")
    first_sentence = (
        response_lines[0].strip().upper().strip(STRIP_SYMBOLS).split(".")[0] + "."
    )
    last_sentence = (
        response_lines[-1].strip().upper().strip(STRIP_SYMBOLS).split(".")[-1] + "."
    )
    first_sentence_words = first_sentence.split()
    last_sentence_words = last_sentence.split()
    first_sentence_no_count, first_sentence_yes_count = count_yes_and_no_words(
        first_sentence_words
    )
    last_sentence_no_count, last_sentence_yes_count = count_yes_and_no_words(
        last_sentence_words
    )

    # Now, let's look at the lines that have an implication in them
    implication_words = ["Therefore,", "Thus,", "So,", "Hence,", "Consequently,"]
    implication_lines = [
        line
        for line in response_lines
        if any(word in line for word in implication_words)
    ]
    implication_results = []
    # Look at the implication lines from the end to the beginning
    for implication_line in implication_lines:
        # Split line by the implication word, we might have several sentences in the same line
        implication_sentence = implication_line
        for word in implication_words:
            if word in implication_sentence:
                parts = implication_sentence.split(word)
                implication_sentence = word + parts[1] if len(parts) > 1 else parts[0]

        words = implication_sentence.split()
        implication_sentence_no_count, implication_sentence_yes_count = (
            count_yes_and_no_words(words)
        )

        # We also check that first line is not NO or YES, because that's something that happens in dumb models (e.g., Qwen 0.5B)
        if implication_sentence_yes_count > 0 and implication_sentence_no_count == 0:
            implication_results.append("YES")
        elif implication_sentence_no_count > 0 and implication_sentence_yes_count == 0:
            implication_results.append("NO")

    if (
        len(implication_results) == 1
        and "YES" in implication_results
        and first_sentence_no_count == 0
        and last_sentence_no_count == 0
    ):
        return "YES"
    elif (
        len(implication_results) == 1
        and "NO" in implication_results
        and first_sentence_yes_count == 0
        and last_sentence_yes_count == 0
    ):
        return "NO"
    if (
        len(implication_results) > 1
        and "YES" in implication_results
        and "NO" in implication_results
    ):
        return "UNKNOWN"

    # Finally, let's check if the first sentence starts with YES or NO
    first_sentence_starts_with_yes = first_sentence.startswith(
        "YES,"
    ) or first_sentence.startswith("YES.")
    first_sentence_starts_with_no = first_sentence.startswith(
        "NO,"
    ) or first_sentence.startswith("NO.")
    if (
        first_sentence_starts_with_yes
        and first_sentence_no_count == 0
        and not found_no_answer_phrase
        and "NO" not in implication_results
    ):
        return "YES"
    elif (
        first_sentence_starts_with_no
        and first_sentence_yes_count == 0
        and not found_yes_answer_phrase
        and "YES" not in implication_results
    ):
        return "NO"

    # Welp, at least we tried
    return "UNKNOWN"


def count_yes_and_no_words(words):
    yes_words = ["YES", "Yes"]
    no_words = ["NO", "No"]
    yes_count = 0
    no_count = 0
    for word in words:
        word = word.strip(STRIP_SYMBOLS)
        if word in yes_words:
            yes_count += 1
        elif word in no_words:
            no_count += 1
    return no_count, yes_count


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
        ds_params=responses.ds_params,
        sampling_params=responses.sampling_params,
    )
