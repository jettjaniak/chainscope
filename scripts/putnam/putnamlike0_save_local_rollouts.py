#!/usr/bin/env python3
"""Script for generating Putnam problem solutions using local TransformerLens models.

Example usage:
python3 scripts/putnam/putnamlike0_save_local_rollouts.py \
    d/putnam2/minimal_fork_of_putnambench_with_clear_answers.yaml \
    --model_id "Qwen/QwQ-32B-Preview" \
    --verbose
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import torch
import yaml
from beartype import beartype
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from chainscope.cot_generation import HookedTransformerWithGenerator
from chainscope.typing import (
    CotResponses,
    SamplingParams,
    MathDatasetParams,
    MathQsDataset,
    MathQuestion,
    MathResponse,
)


def load_putnam_results_as_df(yaml_path: Path) -> pd.DataFrame:
    """Load Putnam results from YAML into a pandas DataFrame."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def create_putnam_dataset(df: pd.DataFrame) -> MathQsDataset:
    """Create a MathQsDataset from a Putnam DataFrame."""
    # Sort problems by year and type
    df = df.sort_values(
        by="problem_name",
        key=lambda x: pd.Series(
            [
                # Extract year and problem type (e.g. 'a1', 'b2')
                (int(name.split("_")[1]), name.split("_")[2])
                for name in x
            ]
        ).map(
            lambda t: (
                {
                    "a1": 0,
                    "b1": 1,
                    "a2": 2,
                    "b2": 3,
                    "a3": 4,
                    "b3": 5,
                    "a4": 6,
                    "b4": 7,
                    "a5": 8,
                    "b5": 9,
                    "a6": 10,
                    "b6": 11,
                }[t[1]],
                -t[0],
            )
        ),
    )

    return MathQsDataset(
        questions=[
            MathQuestion(
                name=row["problem_name"],
                problem=row["informal_statement"],
                solution=row["informal_solution"],
            )
            for _, row in df.iterrows()
        ],
        params=MathDatasetParams(
            description="Putnam Competition Problems",
            id="filtered_putnambench",
            pre_id=None,
        ),
    )


@beartype
def generate_local_rollouts(
    dataset: MathQsDataset,
    model_id: str,
    prefix: Optional[int] = None,
    preamble: str = "",
    local_gen_seed: int = 42,
    max_new_tokens: int = 10_024,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> CotResponses:
    """Generate rollouts using a local TransformerLens model.

    Args:
        dataset: The Putnam dataset
        model_id: HuggingFace model ID
        prefix: Only process first N problems if specified
        preamble: Text to prepend to each problem
        local_gen_seed: Random seed for generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Generated responses
    """
    # Set seed for reproducibility
    HookedTransformerConfig.set_seed_everywhere(None, local_gen_seed)

    # Initialize model
    model = HookedTransformer.from_pretrained(
        model_name=model_id,
        device="cuda:2", #"cuda" if torch.cuda.is_available() else "cpu",
    )
    assert model.tokenizer is not None, "Tokenizer not initialized"
    #model = HookedTransformerWithGenerator(model)

    # Prepare questions
    questions = dataset.questions[:prefix] if prefix else dataset.questions
    responses_by_qid = {}
    model = HookedTransformerWithGenerator(model)
    model.hooked_transformer.to(model.cfg.device)

    # Process each question
    for q in questions:
        logging.info(f"Processing {q.name}")

        # Prepare input
        input_str = f"{preamble}{q.problem}"
        tokens = model.to_tokens(input_str, prepend_bos=True).to("cuda:2") # model.cfg.device)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1

        # Generate response
        with torch.inference_mode():
            generated = []
            try:
                for token in model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    return_type="tokens",
                    verbose=True,
                ):
                    generated.append(token)

                generated = torch.cat(generated, dim=1)
                generated = torch.cat((tokens, generated), dim=1)
                assert isinstance(generated, torch.Tensor)
                assert generated.ndim == 2

                # Convert output tokens to text
                generated_text = model.tokenizer.batch_decode(
                    generated[:, tokens.shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
                assert isinstance(generated_text, str)

                # Split into thinking and answer if possible
                if "**WORKING**" in generated_text and "**ANSWER**" in generated_text:
                    parts = generated_text.split("**ANSWER**")
                    thinking = parts[0].replace("**WORKING**", "").strip()
                    answer = parts[1].strip()
                else:
                    thinking = ""
                    answer = generated_text.strip()

                # Store response
                responses_by_qid[q.name] = {
                    str(uuid.uuid4())[:8]: MathResponse(
                        name=q.name,
                        problem=q.problem,
                        solution=q.solution,
                        model_thinking=thinking,
                        model_answer=[answer],
                    )
                }

            except Exception as e:
                logging.error(f"Failed to generate for {q.name}: {e}")
                import traceback
                print(traceback.format_exc())
                import pdb; pdb.set_trace()
                continue

    return CotResponses(
        responses_by_qid=responses_by_qid,
        model_id=model_id,
        instr_id="instr-v0",
        ds_params=dataset.params,
        sampling_params=SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        ),
    )


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option(
    "--model_id",
    "-s",
    type=str,
    required=True,
    help="HuggingFace model ID for local generation",
)
@click.option(
    "--prefix",
    "-prefix",
    type=int,
    default=None,
    help="Only process the first N problems",
)
@click.option(
    "--preamble",
    type=str,
    default="Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n",
    help="Preamble text to add before each problem",
)
@click.option(
    "--local_gen_seed",
    type=int,
    default=42,
    help="Random seed for generation",
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=1024,
    help="Maximum number of new tokens to generate",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature",
)
@click.option(
    "--top_p",
    type=float,
    default=1.0,
    help="Top-p sampling parameter",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(
    input_yaml: str,
    model_id: str,
    prefix: Optional[int],
    verbose: bool,
    preamble: str,
    local_gen_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Generate Putnam problem solutions using local TransformerLens models."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load and prepare dataset
    input_path = Path(input_yaml)
    df = load_putnam_results_as_df(input_path)
    dataset = create_putnam_dataset(df)

    # Generate rollouts
    results = generate_local_rollouts(
        dataset=dataset,
        model_id=model_id,
        prefix=prefix,
        preamble=preamble,
        local_gen_seed=local_gen_seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Save results
    for i in range(0, 100):
        output_path = results.get_path(
            f"_v{i}" + (f"_prefix_{prefix}" if prefix else "")
        )
        if not os.path.exists(output_path):
            break

    saved_path = results.save(path=output_path)
    logging.info(f"Saved rollouts to {saved_path}")


if __name__ == "__main__":
    main()

