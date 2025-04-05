#!/usr/bin/env python3

import itertools
import logging

import click
from beartype import beartype

from chainscope.api_utils.open_ai_utils import \
    process_batch_results as process_openai_batch_results
from chainscope.api_utils.open_ai_utils import submit_openai_batch
from chainscope.properties import get_entity_type
from chainscope.typing import *

PROMPT = """On a scale of 1 to 10, how well-known or popular is the following {entity_type}: `{entity}`?
With 1 being a completely unknown or obscure {entity_type} and 10 being a super popular {entity_type} that people would generally know of. Return only the number."""


def extract_popularity(response: str) -> int:
    try:
        return int(response.strip())
    except ValueError:
        return -1


@beartype
def submit_batch(
    prop_id: str,
    props: Properties,
    evaluator_model_id: str,
    sampling_params: SamplingParams,
) -> OpenAIBatchInfo:
    """Submit a batch of properties for ambiguity evaluation."""
    # Create prompts for each property, num_evals times
    prompt_by_qrid = {}
    for entity_name in props.value_by_name.keys():
        qr_id = QuestionResponseId(qid=entity_name, uuid="prop_eval")
        prompt = PROMPT.format(entity_type=get_entity_type(props), entity=entity_name)
        logging.info(f"Sending prompt for property {entity_name}: `{prompt}`")
        prompt_by_qrid[qr_id] = prompt

    # Submit batch using OpenAI batch API
    batch_info = submit_openai_batch(
        prompt_by_qrid=prompt_by_qrid,
        instr_id="",
        ds_params=DatasetParams(  # Dummy ds params
            prop_id=prop_id,
            comparison="gt",
            answer="YES",
            max_comparisons=0,
        ),
        evaluated_model_id="props_eval",
        evaluated_sampling_params=sampling_params,
        evaluator_model_id=evaluator_model_id,
    )
    return batch_info


@beartype
def process_batch(batch_info: OpenAIBatchInfo) -> PropEval:
    """Process a batch of responses and create an PropEval object."""
    # Initialize data structures for multiple evaluations
    popularity_by_entity_name: dict[str, int] = {}

    # Process the batch
    results = process_openai_batch_results(batch_info)

    # Group results by qid
    for qr_id, response in results:
        entity_name = qr_id.qid
        popularity = extract_popularity(response)

        # Initialize lists for this qid if not already present
        popularity_by_entity_name[entity_name] = popularity

    return PropEval(
        popularity_by_entity_name=popularity_by_entity_name,
        prop_id=batch_info.ds_params.prop_id,
        model_id=batch_info.evaluator_model_id or batch_info.evaluated_model_id,
        sampling_params=batch_info.evaluated_sampling_params,
    )


@click.group()
def cli() -> None:
    """Evaluate properties using OpenAI's batch API."""
    pass


@cli.command()
@click.option("--evaluator-model-id", default="gpt-4o")
@click.option("--temperature", default=0)
@click.option("--top-p", default=0.9)
@click.option("--max-tokens", default=1)
@click.option(
    "--test",
    is_flag=True,
    help="Test mode: only process 10 properties from first dataset",
)
@click.option("-v", "--verbose", is_flag=True)
def submit(
    evaluator_model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    test: bool,
    verbose: bool,
) -> None:
    """Submit batches of properties for evaluation."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    sampling_params = SamplingParams(
        temperature=float(temperature),
        top_p=top_p,
        max_new_tokens=max_tokens,
    )

    # Find all property files
    property_files = list(DATA_DIR.glob("properties/wm-*.yaml"))
    logging.info(f"Found {len(property_files)} property files")

    if test:
        property_files = property_files[:1]
        logging.info("Test mode: using only first dataset")

    for property_file in property_files:
        try:
            logging.info(f"Processing property file {property_file}")
            props = Properties.load_from_path(property_file)

            if test:
                # Take only first 10 properties
                test_properties = dict(
                    itertools.islice(props.value_by_name.items(), 10)
                )
                props.value_by_name = test_properties
                logging.info(f"Test mode: using {len(test_properties)} properties")

            file_name = property_file.stem
            prop_id = file_name.split(".")[0]

            batch_info = submit_batch(
                prop_id=prop_id,
                props=props,
                evaluator_model_id=evaluator_model_id,
                sampling_params=sampling_params,
            )
            logging.info(f"Submitted batch {batch_info.batch_id} for {property_file}")
            logging.info(f"Batch info saved to {batch_info.save()}")
        except Exception as e:
            logging.error(f"Error processing {property_file}: {e}")


@cli.command()
@click.option("-v", "--verbose", is_flag=True)
def process(verbose: bool) -> None:
    """Process all batches of property evaluation responses."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Find all batch info files
    batch_files = list(DATA_DIR.glob("openai_batches/**/props_eval*.yaml"))
    logging.info(f"Found {len(batch_files)} batch files to process")

    for batch_path in batch_files:
        try:
            logging.info(f"Processing batch {batch_path}")
            batch_info = OpenAIBatchInfo.load(batch_path)
            prop_eval = process_batch(batch_info)
            saved_path = prop_eval.save()
            logging.info(f"Processed batch {batch_info.batch_id}")
            logging.info(f"Results saved to {saved_path}")
        except Exception as e:
            logging.error(f"Error processing {batch_path}: {e}")


if __name__ == "__main__":
    cli()
