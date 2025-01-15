import logging

import openai

from chainscope.typing import *


def check_steps_are_valid_split(original_response: str, steps: list[str]) -> bool:
    """Check if the steps are a valid split of the original response.

    Args:
        original_response: The original CoT response
        steps: List of extracted reasoning steps

    Returns:
        True if steps are valid, False otherwise
    """
    # Check if each step appears verbatim in the original response
    for step in steps:
        if step not in original_response:
            logging.warning(f"Step not found in original response: {step}")
            logging.warning(f"Original response: {original_response}")
            logging.warning(f"Step not found: {step}")
            return False

    # Remove each step from the original response and check if anything remains
    remaining_text = original_response
    for step in steps:
        remaining_text = remaining_text.replace(step, "")

    # Remove all whitespace and newlines
    remaining_text = "".join(remaining_text.split())

    if remaining_text:
        logging.warning(f"Text remains after removing all steps: {remaining_text}")
        logging.warning(f"Original response: {original_response}")
        steps_str = "\n".join(steps)
        logging.warning(f"Steps:\n{steps_str}")
        return False

    return True


def split_cot_response(
    response: str,
    split_model_id: str,
    client: openai.OpenAI,
    temperature: float,
    max_new_tokens: int,
) -> list[str] | None:
    """Split a CoT response into sections using OpenRouter API.

    Args:
        client: OpenRouter API client
        response: The CoT response to split
        split_model_id: Model ID to use for splitting (e.g. "openai/gpt-3.5-turbo")

    Returns:
        A list of strings representing each section of the reasoning.
        Returns None if splitting fails.
    """
    prompt = (
        "Below is a chain-of-thought reasoning. Insert section markers (<section 1>, <section 2>, etc.) "
        "at the start of each logical step in the reasoning, but do not modify the original text in any way. "
        "Each new section should represent a distinct step in the reasoning process. "
        "Do not leave any text out of the sections. Preserve all original formatting, including any "
        "bullet points, whitespace, numbers, or other list markers in the text.\n\n"
        f"{response}"
    )

    try:
        split_response = client.chat.completions.create(
            model=split_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        if not split_response or not split_response.choices:
            return None

        split_text = split_response.choices[0].message.content

        # Extract sections between <section N> tags
        sections = []
        current_pos = 0

        while True:
            # Find next section start
            start = split_text.find("<section", current_pos)
            if start == -1:
                break

            # Find section end
            next_start = split_text.find("<section", start + 1)
            if next_start == -1:
                # Last section - use rest of text
                section_text = split_text[start:].strip()
            else:
                section_text = split_text[start:next_start].strip()

            # Remove the <section N> tag and any </section> tags
            section_text = section_text.split(">", 1)[1].strip()
            section_text = section_text.replace("</section>", "").strip()
            sections.append(section_text)

            current_pos = next_start if next_start != -1 else len(split_text)

        if not check_steps_are_valid_split(response, sections):
            logging.warning("Split sections failed validation")
            return None

        return sections

    except Exception as e:
        logging.warning(f"Failed to split CoT response: {str(e)}")
        return None


def split_cot_responses(
    responses: CotResponses,
    split_model_id: str,
) -> SplitCotResponses:
    """Split all CoT responses for a given model and instruction set."""
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1")

    split_responses_by_qid = {}
    success_count = 0
    failure_count = 0

    # We increase the max_new_tokens by 25% to account for the additional tokens or other overhead
    max_new_tokens = int(responses.sampling_params.max_new_tokens * 0.25)
    temperature = responses.sampling_params.temperature

    for qid, response_by_uuid in responses.responses_by_qid.items():
        split_responses_by_qid[qid] = {}
        for uuid, response in response_by_uuid.items():
            split_response = split_cot_response(
                response,
                split_model_id,
                client,
                temperature,
                max_new_tokens,
            )

            if not split_response:
                logging.warning(
                    f"Unable to split CoT response:\n#####\n`{response}`\n#####"
                )
                failure_count += 1
            else:
                logging.info(
                    f"Response:\n{response}\n\nSplit response ({len(split_response)} sections):\n{split_response}"
                )
                success_count += 1
                split_responses_by_qid[qid][uuid] = split_response

    logging.info(f"Success: {success_count}, Failure: {failure_count}")

    total_count = success_count + failure_count
    if failure_count > 0 and (failure_count / total_count) > 0.1:
        logging.warning(
            f"{failure_count}/{total_count} ({failure_count/total_count:.1%}) responses "
            f"were not split for model {responses.model_id}"
        )

    return SplitCotResponses(
        split_responses_by_qid=split_responses_by_qid,
        model_id=responses.model_id,
        split_model_id=split_model_id,
        instr_id=responses.instr_id,
        ds_params=responses.ds_params,
        sampling_params=responses.sampling_params,
    )
