#!/usr/bin/env python3
"""E.g. run:

python3 -m dotenv run python3 scripts/check_cutoff.py \
    /workspace/COT/chainscope/chainscope/data/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/gemini-2.0-flash-thinking-exp-1219_correct_responses.yaml \
    /workspace/COT/chainscope/chainscope/data/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/gemini-2.0-flash-thinking-exp-1219_correct_responses_splitted.yaml \
    --verbose \
    --prefix=1
"""

import asyncio
import logging
from pathlib import Path
from typing import NamedTuple, Optional

import click

from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.typing import CotResponses, MathResponse, SplitCotResponses

_SUFFIX_EVAL_PROMPT = """Compare these two mathematical text snippets and determine if they express the same conclusion/final answer, even if worded differently.

First snippet:
{raw_suffix}

Second snippet:
{split_suffix}

First explain your comparison, focusing on the mathematical meaning rather than exact wording.
Then conclude with either EQUIVALENT or NOT_EQUIVALENT.
If the second snippet appears to be cut off mid-thought or is clearly shorter/incomplete compared to the first, conclude with NOT_EQUIVALENT.
"""


class SuffixPair(NamedTuple):
    qid: str
    uuid: str
    raw_suffix: str
    split_suffix: str


def process_or_response(
    or_response: str, suffix_pair: SuffixPair
) -> tuple[str, str, bool, str]:
    """Process OpenRouter response to get classification."""
    # Extract the classification from the response
    has_equivalent = or_response.count("EQUIVALENT") > or_response.count(
        "NOT_EQUIVALENT"
    )
    has_not_equivalent = "NOT_EQUIVALENT" in or_response

    match (has_equivalent, has_not_equivalent):
        case (True, False):
            is_equivalent = True
        case (False, True):
            is_equivalent = False
        case _:
            logging.warning(f"Ambiguous classification in response: {or_response}")
            is_equivalent = False

    return suffix_pair.qid, suffix_pair.uuid, not is_equivalent, or_response


async def check_cutoff(
    raw_responses: CotResponses,
    split_responses: SplitCotResponses,
    suffix_fraction: float = 0.1,  # Look at last 10% by default
    prefix: Optional[int] = None,  # Only process first N questions
    model_id: str = "anthropic/claude-3-sonnet",
    max_retries: int = 1,
    max_parallel: Optional[int] = None,
) -> dict[str, dict[str, bool]]:
    """Check if split responses were cutoff compared to raw responses.

    Returns:
        dict mapping qid -> {uuid -> was_cutoff}
    """
    if prefix is None:
        prefix = 1_000_000_000

    # Prepare all suffix pairs for batch processing
    suffix_pairs: list[SuffixPair] = []
    prompts: list[str] = []

    # Get list of QIDs and limit by prefix if specified
    qids = list(raw_responses.responses_by_qid.keys())
    remaining_prefix = prefix

    for qid in qids:
        raw_responses_dict = raw_responses.responses_by_qid[qid]

        # Skip if QID not in split responses
        if qid not in split_responses.split_responses_by_qid:
            logging.warning(f"QID {qid} not found in split responses")
            continue

        split_responses_dict = split_responses.split_responses_by_qid[qid]

        for uuid, raw_response in list(raw_responses_dict.items())[
            : min(remaining_prefix, len(raw_responses_dict))
        ]:
            remaining_prefix -= 1

            # Skip if UUID not in split responses
            if uuid not in split_responses_dict:
                logging.warning(
                    f"UUID {uuid} not found in split responses for QID {qid}"
                )
                continue

            split_response = split_responses_dict[uuid]

            # Get raw response text
            [raw_answer] = raw_response.model_answer
            assert isinstance(raw_answer, str)

            # Get split response text
            split_answer = "\n".join(split_response.model_answer)

            # Get suffix length
            suffix_len = max(1, int(len(raw_answer) * suffix_fraction))

            # Get suffixes
            raw_suffix = raw_answer[-suffix_len:]
            split_suffix = split_answer[-suffix_len:]

            # Add to batch
            suffix_pair = SuffixPair(qid, uuid, raw_suffix, split_suffix)
            suffix_pairs.append(suffix_pair)

            prompt = _SUFFIX_EVAL_PROMPT.format(
                raw_suffix=raw_suffix, split_suffix=split_suffix
            )
            prompts.append(prompt)

            if remaining_prefix <= 0:
                break

        if remaining_prefix <= 0:
            break

    # Set up rate limiter if needed
    or_rate_limiter = None
    if max_parallel is not None:
        or_rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    # Process all pairs in batch
    processor = ORBatchProcessor[SuffixPair, tuple[str, str, bool, str]](
        model_id=model_id,
        max_retries=max_retries,
        max_new_tokens=1000,
        temperature=0.0,
        process_response=process_or_response,
        rate_limiter=or_rate_limiter,
    )

    results = await processor.process_batch(
        items=list(zip(suffix_pairs, prompts, strict=True))
    )

    # Convert results to output format
    cutoff_by_qid: dict[str, dict[str, bool]] = {}

    for suffix_pair, result in results:
        if result is None:
            logging.warning(f"No result for {suffix_pair.qid} {suffix_pair.uuid}")
            continue

        qid, uuid, was_cutoff, explanation = result

        if qid not in cutoff_by_qid:
            cutoff_by_qid[qid] = {}

        cutoff_by_qid[qid][uuid] = was_cutoff

        if was_cutoff:
            logging.info(f"Found cutoff in {qid} {uuid}:")
            logging.info(f"Raw suffix: {suffix_pair.raw_suffix}")
            logging.info(f"Split suffix: {suffix_pair.split_suffix}")
            logging.info(f"Explanation: {explanation}")

    return cutoff_by_qid


def save_not_cutoff_responses(
    split_responses: SplitCotResponses,
    cutoff_by_qid: dict[str, dict[str, bool]],
    path: str | Path,
) -> Path:
    """Save responses that were not cutoff to a new file."""
    # Create new responses dict with only non-cutoff responses
    not_cutoff_responses: dict[str, dict[str, MathResponse]] = {"default_qid": {}}

    for qid, cutoff_dict in cutoff_by_qid.items():
        for uuid, was_cutoff in cutoff_dict.items():
            if not was_cutoff:
                if uuid in split_responses.split_responses_by_qid[qid]:
                    not_cutoff_responses["default_qid"][uuid] = (
                        split_responses.split_responses_by_qid[qid][uuid]
                    )

    # Create new SplitCotResponses object
    not_cutoff_split_responses = SplitCotResponses(
        split_responses_by_qid=not_cutoff_responses,
        successfully_split_count=len(not_cutoff_responses["default_qid"]),
        failed_to_split_count=0,  # These all split successfully
        model_id=split_responses.model_id,
        instr_id=split_responses.instr_id,
        ds_params=split_responses.ds_params,
        sampling_params=split_responses.sampling_params,
    )

    # Save to new path with _not_cutoff suffix
    path = str(path)
    path_split = path.split(".")
    path_split[-2] = path_split[-2] + "_not_cutoff"
    save_path = Path(".".join(path_split))

    return not_cutoff_split_responses.save(path=save_path)


@click.command()
@click.argument("raw_yaml", type=click.Path(exists=True))
@click.argument("split_yaml", type=click.Path(exists=True))
@click.option(
    "--suffix_fraction",
    "-f",
    type=float,
    default=0.1,
    help="Fraction of text to compare at the end (default: 0.1)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--prefix",
    "-prefix",
    type=int,
    default=None,
    help="Only process the first N questions",
)
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3-sonnet",
    help="Model to use for evaluation",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=1,
    help="Maximum retries for failed requests",
)
@click.option(
    "--max_parallel",
    "-p",
    type=int,
    default=None,
    help="Maximum number of parallel requests",
)
def main(
    raw_yaml: str,
    split_yaml: str,
    suffix_fraction: float,
    verbose: bool,
    prefix: Optional[int],
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
):
    """Check if split responses were cutoff compared to raw responses."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load responses
    raw_responses = CotResponses.load(Path(raw_yaml))
    split_responses = SplitCotResponses.load(Path(split_yaml))

    # Check for cutoffs
    cutoff_by_qid = asyncio.run(
        check_cutoff(
            raw_responses=raw_responses,
            split_responses=split_responses,
            suffix_fraction=suffix_fraction,
            prefix=prefix,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )

    # Save non-cutoff responses
    save_path = save_not_cutoff_responses(
        split_responses=split_responses, cutoff_by_qid=cutoff_by_qid, path=split_yaml
    )
    logging.info(f"Saved non-cutoff responses to {save_path}")

    # Print summary
    total_responses = 0
    total_cutoffs = 0

    for qid, cutoff_dict in cutoff_by_qid.items():
        qid_responses = len(cutoff_dict)
        qid_cutoffs = sum(cutoff_dict.values())

        total_responses += qid_responses
        total_cutoffs += qid_cutoffs

        if qid_cutoffs > 0:
            print(f"\nQID {qid}:")
            print(f"Cutoffs: {qid_cutoffs}/{qid_responses}")

            # Print details of cutoff responses
            for uuid, was_cutoff in cutoff_dict.items():
                if was_cutoff:
                    print(f"  UUID {uuid} was cutoff")

    print(f"\nTotal cutoffs: {total_cutoffs}/{total_responses}")
    print(f"Cutoff rate: {total_cutoffs/total_responses:.1%}")
    print(
        f"Saved {total_responses - total_cutoffs} non-cutoff responses to {save_path}"
    )


if __name__ == "__main__":
    main()
