import asyncio
import logging

from chainscope.api_utils.open_ai_utils import OABatchProcessor
from chainscope.typing import *


def parse_evaluator_response(
    response: str,
    x_name: str,
    y_name: str,
) -> dict[str, str]:
    """Parse the response to get the ground truth data

    Args:
        response: The evaluator's response text
        x_name: Name of the first variable to extract
        y_name: Name of the second variable to extract

    Returns:
        Dictionary mapping variable names to their extracted values
    """
    response = response.strip()

    # Check for the expected section header
    if "EXTRACTED_DATA:" not in response:
        raise ValueError("Response does not contain EXTRACTED_DATA section")

    # Get the text after EXTRACTED_DATA:
    extracted_section = response.split("EXTRACTED_DATA:")[-1].strip()

    # Initialize results
    results = {}

    # Parse each line looking for variable assignments
    for line in extracted_section.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Look for lines with format "variable_name: value"
        if ":" not in line:
            continue

        var_name, value = line.split(":", 1)
        var_name = var_name.strip()
        value = value.strip()

        # Remove any square brackets from the value
        value = value.strip("[]")

        # Only store values for the requested variable names
        if var_name == x_name:
            results[x_name] = value
        elif var_name == y_name:
            results[y_name] = value

    # Ensure both variables were found
    if x_name not in results:
        results[x_name] = "UNKNOWN"
    if y_name not in results:
        results[y_name] = "UNKNOWN"

    # assert that keys are x_name and y_name
    assert set(results.keys()) == {x_name, y_name}

    return results


async def extract_gt_data_async(
    open_ended_responses: CotResponses,
    evaluator_model_id: str,
    max_retries: int,
) -> GroundTruthEvalData:
    """Evaluate ground truth in parallel."""

    def process_response(
        evaluator_response: str, item: tuple[str, str, str, str]
    ) -> dict[str, str]:
        logging.info(f"Evaluator response:\n{evaluator_response}")
        qid, response_uuid, x_name, y_name = item
        return parse_evaluator_response(evaluator_response, x_name, y_name)

    processor = OABatchProcessor[
        tuple[str, str, str, str],
        dict[str, str],
    ](
        oa_model_ids=[evaluator_model_id],
        temperature=open_ended_responses.sampling_params.temperature,
        max_new_tokens=int(open_ended_responses.sampling_params.max_new_tokens * 1.25),
        max_retries=max_retries,
        process_response=process_response,
        oa_rate_limiter=None,
    )

    # Load the question dataset to get question strings
    question_dataset = open_ended_responses.ds_params.load_qs_dataset()

    # Prepare batch items
    batch_items = []
    for qid, response_by_uuid in open_ended_responses.responses_by_qid.items():
        # Get the question string
        question = question_dataset.question_by_qid[qid]
        for uuid, response in response_by_uuid.items():
            prompt = f"""Analyze the following chain-of-thought reasoning for this open ended question and extract the factual data from the reasoning. Specifically, extract the values of the following variables: `{question.x_name}` and `{question.y_name}`.

Important: Focus only on the cited values used in the reasoning. Do not evaluate if the reasoning or answer is correct.

Additional guidelines:
- If the values involve long distances, extract the distance in kilometers.
- If the values involve time, extract the time in years.
- If the values involve speed, extract the speed in km/h.
- If the values involve temperature, extract the temperature in Celsius.
- If the values involve heights, extract the height in meters.
- If the values involve weights, extract the weight in kilograms.
- If the values involve lengths, extract the length in meters.
- If the values involve depths, extract the depth in meters.

If the values are not clear from the reasoning, or do not match any of the above guidelines, extract the value as UNKNOWN.
Do NOT include the units in your response.

Here is the question that was asked: `{question.q_str}`

And here is the response to analyze:

```
{response}
```

Please structure your response as follows:

EXTRACTED_DATA:
{question.x_name}: [The extracted value]
{question.y_name}: [The extracted value]"""

            logging.info(f"Prompt for qid={qid}, uuid={uuid}:\n{prompt}")
            batch_items.append(((qid, uuid, question.x_name, question.y_name), prompt))

    # Process batch
    results = await processor.process_batch(batch_items)

    # Process results
    gt_by_qid: dict[
        str, dict[str, dict[str, str]]
    ] = {}  # qid -> {response_uuid -> {entity -> extracted_value}}
    open_ended_responses_by_qid: dict[
        str, dict[str, str]
    ] = {}  # qid -> {response_uuid -> response_str}
    for (qid, response_uuid, x_name, y_name), gt_extraction_result in results:
        if qid not in gt_by_qid:
            gt_by_qid[qid] = {}

        if qid not in open_ended_responses_by_qid:
            open_ended_responses_by_qid[qid] = {}

        if gt_extraction_result is None:
            logging.error(
                f"Unable to extract gt data for qid={qid}, uuid={response_uuid}"
            )
            raise ValueError(
                f"Unable to extract gt data for qid={qid}, uuid={response_uuid}"
            )

        gt_by_qid[qid][response_uuid] = gt_extraction_result
        open_ended_responses_by_qid[qid][response_uuid] = (
            open_ended_responses.responses_by_qid[qid][response_uuid]
        )

    return GroundTruthEvalData(
        gt_by_qid=gt_by_qid,
        open_ended_responses_by_qid=open_ended_responses_by_qid,
        model_id=open_ended_responses.model_id,
        evaluator_model_id=evaluator_model_id,
        instr_id=open_ended_responses.instr_id,
        ds_params=open_ended_responses.ds_params,
        sampling_params=open_ended_responses.sampling_params,
    )


def extract_gt_data(
    open_ended_responses: CotResponses,
    evaluator_model_id: str,
    max_retries: int,
) -> GroundTruthEvalData:
    """Evaluate all CoT responses for a given model and instruction set to extract ground truth."""
    return asyncio.run(
        extract_gt_data_async(
            open_ended_responses=open_ended_responses,
            evaluator_model_id=evaluator_model_id,
            max_retries=max_retries,
        )
    )
