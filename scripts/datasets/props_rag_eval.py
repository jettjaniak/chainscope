#!/usr/bin/env python3

import itertools
import logging
import traceback

import click
from beartype import beartype
from tqdm import tqdm

from chainscope.api_utils.open_ai_utils import \
    process_batch_results as process_openai_batch_results
from chainscope.api_utils.open_ai_utils import submit_openai_batch
from chainscope.rag import (build_rag_extraction_prompt, build_rag_query,
                            get_openai_web_search_rag_values, get_rag_sources)
from chainscope.typing import *


@beartype
def submit_batch(
    prop_id: str,
    entity_names: list[str],
    query_by_entity_name: dict[str, str],
    sources_by_entity: dict[str, list[RAGSource]],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
) -> OpenAIBatchInfo:
    """Submit a batch of RAG property evaluations."""
    # Create prompts for each property and source
    prompt_by_qrid = {}
    
    for entity_name in entity_names:
        sources = sources_by_entity[entity_name]
        for i, source in enumerate(sources):
            qr_id = QuestionResponseId(qid=f"{entity_name}_{i}", uuid="prop_rag_eval")
            query = query_by_entity_name[entity_name]
            prompt = build_rag_extraction_prompt(query, source)
            logging.info(f"Sending prompt for entity {entity_name}, source {i}: `{prompt}`")
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
        evaluated_model_id="props_rag_eval",
        evaluated_sampling_params=sampling_params,
        evaluator_model_id=evaluator_model_id,
    )
    
    # Store sources in batch_info.metadata instead of separate file
    batch_info.metadata["sources_by_entity"] = sources_by_entity
    batch_info.save()
    
    return batch_info

@beartype
def process_batch(
    batch_info: OpenAIBatchInfo,
) -> PropRAGEval:
    """Process a batch of responses and create a PropRAGEval object."""
    # Get sources from batch_info metadata if not provided
    sources_by_entity = {entity_name: [RAGSource(**source) for source in sources]
                         for entity_name, sources in batch_info.metadata.get("sources_by_entity", {}).items()}
    values_by_entity_name: dict[str, list[RAGValue]] = {}
    
    # Check for existing results
    prop_id = batch_info.ds_params.prop_id
    output_path = DATA_DIR / "prop_rag_evals" / f"{prop_id}.yaml"
    if output_path.exists():
        existing_eval = PropRAGEval.load(output_path)
        values_by_entity_name = existing_eval.values_by_entity_name
    
    # Process the batch
    results = process_openai_batch_results(batch_info)

    # Group results by entity name
    for qr_id, response in results:
        entity_name, source_idx = qr_id.qid.rsplit("_", 1)  # Split entity name and source index
        source_idx = int(source_idx)
        source = sources_by_entity[entity_name][source_idx]

        # Remove content and relevant_snippet from source, since they take up a lot of space
        source.content = None
        source.relevant_snippet = None

        is_unknown = response and response.strip().lower() == "unknown"
        
        if not is_unknown:
            if entity_name not in values_by_entity_name:
                values_by_entity_name[entity_name] = []
            logging.info(f"Extracted value: `{response.strip()}` from source: `{source.url}`")
            values_by_entity_name[entity_name].append(RAGValue(value=response.strip(), source=source))
        else:
            logging.info(f"Skipping unknown value for entity {entity_name} from source {source.url}")

    # Create new PropRAGEval with merged results
    return PropRAGEval(
        values_by_entity_name=values_by_entity_name,
        prop_id=prop_id,
        model_id=batch_info.evaluator_model_id or batch_info.evaluated_model_id,
        sampling_params=batch_info.evaluated_sampling_params,
    )

@beartype
def perform_rag_eval_using_google_search(
    prop_id: str,
    props: Properties,
    entity_names: list[str],
    evaluator_model_id: str,
    sampling_params: SamplingParams,
) -> None:
    logging.info(f"Submitting batch for {len(entity_names)} unprocessed entities")

    # Get sources for each entity
    query_by_entity_name = {}
    sources_by_entity = {}
    for entity_name in tqdm(entity_names, desc=f"Getting RAG sources for props {prop_id}"):
        query = build_rag_query(entity_name, props)
        query_by_entity_name[entity_name] = query
        found_sources = get_rag_sources(query)
        sources = [s for s in found_sources if s.content is not None or s.relevant_snippet is not None]
        sources_by_entity[entity_name] = sources

    batch_info = submit_batch(
        prop_id=prop_id,
        entity_names=list(entity_names),
        query_by_entity_name=query_by_entity_name,
        sources_by_entity=sources_by_entity,
        evaluator_model_id=evaluator_model_id,
        sampling_params=sampling_params,
    )
    logging.info(f"Submitted batch {batch_info.batch_id} for {prop_id}")
    logging.info(f"Batch info saved to {batch_info.save()}")

@beartype
def perform_rag_eval_using_gpt_4o_web_search(
    prop_id: str,
    props: Properties,
    entity_names: list[str],
    sampling_params: SamplingParams,
    existing_rag_eval: PropRAGEval | None = None,
    save_every: int = 50,
) -> None:
    rag_eval = existing_rag_eval or PropRAGEval(
        values_by_entity_name={},
        prop_id=prop_id,
        model_id="gpt-4o-search-preview",
        sampling_params=sampling_params,
    )

    for i, entity_name in tqdm(enumerate(entity_names), desc=f"Fetching RAG values via GPT-4o web search for props {prop_id}"):
        if entity_name not in rag_eval.values_by_entity_name:
            rag_eval.values_by_entity_name[entity_name] = []

        query = build_rag_query(entity_name, props)
        values = get_openai_web_search_rag_values(query, model_id="gpt-4o-search-preview", search_context_size="medium")

        for value in values:
            same_url = next((v for v in rag_eval.values_by_entity_name[entity_name] if v.source.url == value.source.url), None)
            if same_url is None:
                rag_eval.values_by_entity_name[entity_name].append(value)
            else:
                logging.info(f"NOT replacing value for {entity_name} in {same_url.source.url} from {same_url.value} to {value.value}")
        
        # Save periodically
        if (i + 1) % save_every == 0:
            logging.info(f"Saving intermediate results after processing {i + 1} entities")
            rag_eval.save()
    
    logging.info(f"RAG eval for {prop_id} saved to {rag_eval.save()}")

@click.group()
def cli() -> None:
    """Evaluate properties using RAG and OpenAI's batch API."""
    pass

@cli.command()
@click.option("--evaluator-model-id", default="gpt-4o-2024-11-20")
@click.option("--rag-method", default="gpt-4o-web-search", type=click.Choice(["gpt-4o-web-search", "google-search-with-zyte"]))
@click.option("--temperature", default=0)
@click.option("--top-p", default=0.9)
@click.option("--max-tokens", default=1000)
@click.option("--min-popularity",
    type=int,
    default=None,
    help="Perform RAG eval only for entities with popularity >= this value (1-10)",
)
@click.option("--max-popularity",
    type=int,
    default=None,
    help="Perform RAG eval only for entities with popularity <= this value (1-10)",
)
@click.option(
    "--skip-processed-entities",
    is_flag=True,
    help="Skip entities that have already been processed",
)
@click.option(
    "--test",
    is_flag=True,
    help="Test mode: only process 10 properties from first dataset",
)
@click.option(
    "--save-every",
    default=50,
    help="Save RAG eval results every N entities (only for gpt-4o-web-search)",
)
@click.option("-v", "--verbose", is_flag=True)
def submit(
    evaluator_model_id: str,
    rag_method: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    min_popularity: int | None,
    max_popularity: int | None,
    skip_processed_entities: bool,
    test: bool,
    verbose: bool,
    save_every: int,
) -> None:
    """Submit batches of properties for RAG evaluation."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if min_popularity is not None:
        assert 1 <= min_popularity <= 10, "Min popularity must be between 1 and 10"
    if max_popularity is not None:
        assert 1 <= max_popularity <= 10, "Max popularity must be between 1 and 10"

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

    for property_file in tqdm(property_files, desc="Processing properties"):
        try:
            logging.info(f"Processing property file {property_file}")
            props = Properties.load_from_path(property_file)
            prop_id = property_file.stem

            # Check for existing processed results
            existing_rag_eval = None
            processed_entities = set()
            try:
                existing_rag_eval = PropRAGEval.load_id(prop_id, sampling_params)
                existing_rag_eval.sampling_params = sampling_params
                processed_entities = set(existing_rag_eval.values_by_entity_name.keys())
                logging.info(f"Found {len(processed_entities)} already processed entities in {existing_rag_eval.get_path()}")
            except FileNotFoundError:
                logging.info(f"No existing results found for {prop_id}")

            # Filter entities by popularity if needed
            if min_popularity is not None or max_popularity is not None:
                prop_eval = PropEval.load_id(prop_id)
                filtered_entities = [(entity, pop) for entity, pop in prop_eval.popularity_by_entity_name.items()]
                if min_popularity is not None:
                    filtered_entities = [
                        (entity, pop) for entity, pop in filtered_entities if pop >= min_popularity
                    ]
                if max_popularity is not None:
                    filtered_entities = [
                        (entity, pop) for entity, pop in filtered_entities if pop <= max_popularity
                    ]
                props.value_by_name = {k: pop for k, pop in filtered_entities}

            if test:
                test_properties = dict(
                    itertools.islice(props.value_by_name.items(), 3)
                )
                props.value_by_name = test_properties
                logging.info(f"Test mode: using {len(test_properties)} properties")

            # Filter out already processed entities
            if skip_processed_entities:
                entities_to_process = {
                    k: v for k, v in props.value_by_name.items() 
                    if k not in processed_entities
                }
            else:
                entities_to_process = props.value_by_name

            if not entities_to_process:
                logging.info(f"All entities for {prop_id} have already been processed, skipping")
                continue

            if rag_method == "gpt-4o-web-search":
                perform_rag_eval_using_gpt_4o_web_search(
                    prop_id=prop_id,
                    props=props,
                    entity_names=list(entities_to_process.keys()),
                    sampling_params=sampling_params,
                    existing_rag_eval=existing_rag_eval,
                    save_every=save_every,
                )
            elif rag_method == "google-search-with-zyte":
                perform_rag_eval_using_google_search(
                    prop_id=prop_id,
                    props=props,
                    entity_names=list(entities_to_process.keys()),
                    evaluator_model_id=evaluator_model_id,
                    sampling_params=sampling_params,
                )
            else:
                raise ValueError(f"Invalid RAG method: {rag_method}")

        except Exception as e:
            logging.error(f"Error processing {property_file}: {e}")
            # print stack trace
            traceback.print_exc()

@cli.command()
@click.option("-v", "--verbose", is_flag=True)
def process(verbose: bool) -> None:
    """Process all batches of property RAG evaluation responses."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Find all batch info files
    batch_files = list(DATA_DIR.glob("openai_batches/**/props_rag_eval*.yaml"))
    logging.info(f"Found {len(batch_files)} batch files to process")

    for batch_path in batch_files:
        try:
            logging.info(f"Processing batch {batch_path}")
            batch_info = OpenAIBatchInfo.load(batch_path)
            
            prop_rag_eval = process_batch(batch_info)
            saved_path = prop_rag_eval.save()
            logging.info(f"Processed batch {batch_info.batch_id}")
            logging.info(f"Results saved to {saved_path}")
        except Exception as e:
            logging.error(f"Error processing {batch_path}: {e}")

if __name__ == "__main__":
    cli() 