#!/usr/bin/env python3

"""E.g. run:

python3 -m dotenv run python3 scripts/atcoder/atcoder1_are_rollouts_correct.py \
    d/cot_responses/instr-v0/default_sampling_params/atcoder/qwen__qwq-32b-preview_v0.yaml \
    --verbose \
    --prefix=40
"""

import asyncio
import dataclasses
import itertools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Final, Optional

import click
import requests
import yaml

from chainscope.typing import (
    AtCoderDatasetParams,
    AtCoderResponse,
    AtCoderStats,
    CotResponses,
    DefaultSamplingParams,
    MathResponse,
)

SAFE_MAX_LONG_LONG_VALUE: Final[int] = 9223372036854775807 // 1000  # / 1000 to be safe

FINAL_SUBMISSION_STATUSES: Final[set[str]] = {
    "AC",
    "WA",
    "RE",
    "RTE",
    "TLE",
    "MLE",
    "ME",
    "CE",
}  # , "IE", "OLE", "SE"}  # hopefully do not get anything more cursed than these lol

MAX_ACCOUNT_FINDING_ATTEMPTS: Final[int] = (
    40  # Maximum number of attempts to find an inactive account
)


@dataclasses.dataclass
class AtCoderAccount:
    username: str
    password: str

    # e.g. ("abc123", 654321321)
    current_submission: tuple[str, int] | None = None


ATCODER_ACCOUNTS: list[AtCoderAccount] = []


async def get_inactive_account_idx() -> int:
    """Get an inactive account."""

    for loop_idx, account_idx in enumerate(
        itertools.cycle(range(len(ATCODER_ACCOUNTS)))
    ):
        account = ATCODER_ACCOUNTS[account_idx]

        if account.current_submission is None:
            return account_idx

        # Mockup f"https://atcoder.jp/contests/{account.current_submission[0]}/submissions/{account.current_submission[1]}" ish
        elif (
            check_submission_status(
                f"https://atcoder.jp/contests/{account.current_submission[0]}/submissions/{account.current_submission[1]}"
            )[0]
            in FINAL_SUBMISSION_STATUSES
        ):
            ATCODER_ACCOUNTS[account_idx].current_submission = None
            return account_idx

        if (account_idx + 1) == len(ATCODER_ACCOUNTS):
            logging.info(
                f"Completed a full cycle of accounts, sleeping for {loop_idx + 1} seconds"
            )
            await asyncio.sleep(loop_idx + 1)

        if loop_idx > MAX_ACCOUNT_FINDING_ATTEMPTS:
            raise ValueError(
                f"Too many failures ({MAX_ACCOUNT_FINDING_ATTEMPTS}) to find an inactive account, stopping."
            )


def load_atcoder_model_responses(
    yaml_path: Path, prefix: Optional[int] = None
) -> tuple[list[AtCoderResponse], str]:
    """Load AtCoder dataset from CotResponses YAML format, and return the model_id."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    model_id = data["model_id"]
    logging.info(f"Loaded YAML data with keys: {list(data.keys())}")

    # Extract unique questions from the responses
    questions: list[AtCoderResponse] = []
    try:
        responses_by_qid = data["responses_by_qid"]
    except Exception as e:
        print(f"Error: {e}")
        print(f"Data: {data}")
        raise e
    logging.info(f"Found {len(responses_by_qid)} qids in responses_by_qid")

    for qid, responses_dict in responses_by_qid.items():
        logging.info(f"Processing qid {qid} with {len(responses_dict)} responses")
        for response_data in responses_dict.values():
            if not isinstance(response_data, dict):
                continue
            questions.append(
                AtCoderResponse(
                    name=qid,
                    problem=response_data["problem"],
                    cpp_solution=response_data.get("cpp_solution"),
                    model_answer=response_data.get("model_answer", []),
                    model_thinking=response_data.get("model_thinking"),
                )
            )
            break  # Only take first response for each question

    if prefix is not None:
        questions = questions[:prefix]

    logging.info(f"Loaded {len(questions)} questions total")
    return questions, model_id


def extract_cpp_code(response: AtCoderResponse) -> Optional[str]:
    """Extract C++ code from model answer."""
    if not response.model_answer:
        logging.error(f"No model answer found for {response.name}")
        return None

    # Join all parts of the model answer
    full_answer = "\n".join(response.model_answer)

    # Look for code between ```cpp and ```
    match = re.search(r"```cpp([^`]*?)```(?!.*```cpp)", full_answer, re.DOTALL)
    if not match:
        logging.error(
            f"Is this really C++ code? {str(full_answer)[:100]=}; {str(full_answer)[-100:]=}"
        )
        return None

    return (
        match.group(1).strip()
        + f"\n// This is code scraped at {int(time.time()*1_000)}ms\n"
    )


def create_metadata_json(response: AtCoderResponse) -> dict:
    """Create metadata.json for atcoder-tools."""
    # Parse contest_id and problem_id from name
    # Format is like 'abc361_abc361_a'
    parts = response.name.split("_")
    if len(parts) not in [3, 5]:
        raise ValueError(f"Invalid response name format: {response.name}")
    if len(parts) == 5:
        parts = [f"{parts[0]}_{parts[1]}", parts[2], f"{parts[3]}_{parts[4]}"]

    contest_id = parts[0]
    problem_id = f"{parts[1]}_{parts[2]}"
    problem_letter = problem_id.split("_")[-1]

    return {
        "code_filename": "main.cpp",
        "judge": {"judge_type": "normal"},
        "lang": "cpp",
        "problem": {
            "alphabet": problem_letter.upper(),
            "contest": {"contest_id": contest_id},
            "problem_id": problem_id,
        },
        "sample_in_pattern": "input_*.txt",
        "sample_out_pattern": "output_*.txt",
        "timeout_ms": 2000,
    }


def check_submission_status(
    submission_output: str,
) -> tuple[str | None, int | None]:
    """Check if the submission was successful by parsing the submission page.

    Args:
        submission_output: Output from atcoder-tools submit command

    Returns:
        Tuple of (status, submission_url)
        where status is the parsed response
        and submission_url is the URL of the submission if found, None otherwise
    """
    # Look for submission URL in output

    match = re.search(
        r"https://atcoder.jp/contests/[^/]+/submissions/(\d+)", submission_output
    )
    if not match:
        logging.warning("No submission URL found in output")
        return None, None

    submission_url = match.group(0)
    logging.info(f"Found submission URL: {submission_url}")
    # Get the submission ID from the URL
    submission_id = int(submission_url.split("/")[-1])
    logging.info(f"Submission ID: {submission_id=}")

    try:
        response = requests.get(submission_url)
        response.raise_for_status()

        # Look for the status cell
        status_match = re.search(
            r'<td id="judge-status"[^>]*><span[^>]*>([^<]+)</span></td>', response.text
        )
        if not status_match:
            logging.warning("Could not find status in submission page")
            return False, submission_url

        status = status_match.group(1)
        logging.info(f"Submission status: {status=} and we will strip and upper that")
        return status.strip().upper(), submission_url

    except Exception as e:
        logging.error(f"Error checking submission page: {e}")
        return None, submission_url


async def check_submission_status_with_retries(
    submission_output: str,
    max_retries: int = 10,
    initial_backoff: int = 10,
) -> tuple[str | None, str | None]:
    """Check submission status with retries and linear backoff.

    Args:
        submission_output: Output from atcoder-tools submit command
        max_retries: Maximum number of retries (default: 10)
        initial_backoff: Initial backoff time in seconds (default: 10)

    Returns:
        Tuple of (status, submission_url)
        where status is the parsed response
        and submission_url is the URL of the submission if found, None otherwise
    """
    for attempt in range(max_retries):
        status, submission_url = check_submission_status(submission_output)

        if status in FINAL_SUBMISSION_STATUSES or status is None:
            return status, submission_url

        if attempt < max_retries - 1:
            backoff = initial_backoff + (
                attempt * 10
            )  # Linear backoff: 10, 20, 30, ...
            logging.info(
                f"Status not final yet ({status}), waiting {backoff} seconds before retry {attempt + 1}/{max_retries}"
            )
            await asyncio.sleep(backoff)

    return status, submission_url


def save_correct_results(
    results: list[tuple[AtCoderResponse, str | None]],
    model_id: str,
    path: str | Path,
    suffix: str = "",
    stats: AtCoderStats | None = None,
) -> Path:
    """Save all evaluation results using CotResponses format."""
    responses: dict[str, dict[str, MathResponse | AtCoderResponse | str]] = {}

    for question, _ in results:  # _ is cpp_code
        if question.correctness_is_correct:
            if question.name not in responses:
                responses[question.name] = {}
            responses[question.name]["default"] = question

    ds_params = AtCoderDatasetParams(
        description=f"AtCoder Problems with Evaluations (Suffix: {suffix})",
        id="atcoder",
        pre_id=None,
    )

    cot_responses = CotResponses(
        responses_by_qid=responses,
        model_id=model_id,
        instr_id="evaluation",
        ds_params=ds_params,
        sampling_params=DefaultSamplingParams(),
        atcoder_stats=stats or AtCoderStats(),
    )

    # Make the new path the same as the old with suffix:
    suffix = "_just_correct_responses"
    path = str(path)
    path_split = path.split(".")
    path_split[-2] = path_split[-2] + suffix
    path = Path(".".join(path_split))
    return cot_responses.save(path=path)


async def evaluate_model_responses(
    model_responses: list[AtCoderResponse],
    max_retries: int = 3,
) -> tuple[list[tuple[AtCoderResponse, str | None]], AtCoderStats]:
    """Evaluate responses by submitting to AtCoder."""
    results: list[tuple[AtCoderResponse, bool, str | None]] = []
    submissions: list[
        tuple[AtCoderResponse, str, str]
    ] = []  # (response, stdout, stderr)

    stats = AtCoderStats.zeros()

    try:
        # Phase 1: Submit all problems
        for response in model_responses:
            logging.info(
                f"Here is the raw response: {str(response)[:100]=}; {str(response)[-100:]=}"
            )
            cpp_code = extract_cpp_code(response)

            if cpp_code:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Try compiling and submitting with retries:
                    success = False
                    compiled = False
                    last_stdout = ""
                    last_stderr = ""

                    for attempt in range(max_retries):
                        tmpdir_path = Path(tmpdir)

                        # Save C++ code
                        with open(tmpdir_path / "main.cpp", "w") as f:
                            f.write(
                                f"const long long AI_SUBMISSION_INTEGER = {random.randint(0, SAFE_MAX_LONG_LONG_VALUE)};\n{cpp_code}"
                            )

                        # Create metadata.json
                        metadata = create_metadata_json(response)
                        with open(tmpdir_path / "metadata.json", "w") as f:
                            json.dump(metadata, f, indent=4)

                        # Sleep a bit between any submission ever
                        logging.info(
                            "Sleeping for 0.5 seconds before submitting (as always done!) ..."
                        )
                        await asyncio.sleep(0.5)

                        try:
                            # Compile code
                            compiler = "clang++" if shutil.which("clang++") else "g++"
                            subprocess.run(
                                [compiler, "main.cpp", "-o", "main"],
                                cwd=tmpdir,
                                check=True,
                                capture_output=True,
                            )
                        except Exception as e:
                            stats.compilation_failed += 1
                            logging.error(f"Error compiling {response.name}: {e}")
                            break
                        compiled = True

                        try:
                            # Get an inactive account
                            account_idx = await get_inactive_account_idx()

                            # Submit to AtCoder
                            result = subprocess.run(
                                ["atcoder-tools", "submit", "-u", "-f"]
                                + (
                                    ["--credential", str(account_idx + 1)]
                                    if account_idx >= 1
                                    else []
                                ),
                                cwd=tmpdir,
                                check=True,
                                capture_output=True,
                                text=True,
                                env=os.environ.copy(),
                            )

                            stdout = result.stdout
                            if stdout:
                                logging.info(
                                    f"Stdout: {str(stdout)[:400]=}; {str(stdout)[-400:]=}"
                                )

                            stderr = result.stderr
                            if stderr:
                                logging.info(
                                    f"Stderr: {str(stderr)[:400]=}; {str(stderr)[-400:]=}"
                                )

                            logging.info(
                                f"Submitted {response.name} with account {account_idx + 1}, sleeping for 0.5 seconds before checking status"
                            )
                            await asyncio.sleep(0.5)
                            _, submission_url = check_submission_status(
                                f"{stdout}{stderr}"
                            )

                            try:
                                if "ttpc" in response.name.lower():
                                    competition_name = "_".join(
                                        response.name.split("_")[:2]
                                    )
                                else:
                                    competition_name = response.name.split("_")[0]
                                ATCODER_ACCOUNTS[account_idx].current_submission = (
                                    competition_name,
                                    int(submission_url.split("/")[-1]),
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error updating account {account_idx + 1} current submission: {e=}"
                                )

                            success = True
                            last_stdout = stdout
                            last_stderr = stderr
                            break

                        except subprocess.CalledProcessError as e:
                            logging.error(
                                f"Error processing {response.name} (attempt {attempt + 1}/{max_retries}): {e}"
                            )
                            logging.error(
                                f"Command output: stdout={e.stdout}, stderr={e.stderr}"
                            )
                            last_stdout = e.stdout
                            last_stderr = e.stderr
                            if attempt < max_retries - 1:
                                sleep_amount = 4 ** (attempt + 1)
                                logging.info(
                                    f"Waiting {sleep_amount} seconds before retrying"
                                )
                                await asyncio.sleep(
                                    sleep_amount
                                )  # Wait before retrying
                            continue

                        except Exception as e:
                            logging.error(
                                f"Error processing {response.name} (attempt {attempt + 1}/{max_retries}): {e}"
                            )
                            if attempt < max_retries - 1:
                                sleep_amount = 4 ** (attempt + 1)
                                logging.info(
                                    f"Waiting {sleep_amount} seconds before retrying"
                                )
                                await asyncio.sleep(
                                    sleep_amount
                                )  # Wait before retrying
                            continue

                    if compiled:
                        if success:
                            logging.info(
                                f"Submission successful after {attempt + 1} total attempts"
                            )
                        else:
                            logging.error(
                                f"Submission failed after {max_retries} attempts"
                            )
                            stats.atcodertools_cmd_failed += 1
                            continue

                    submissions.append((response, last_stdout, last_stderr))

            else:
                stats.finding_code_failed += 1
                logging.error(f"No C++ code found for {response.name}")
                continue

    except (
        BaseException
    ) as e:  # NOTE! Also catch KeyboardInterrupt, so we can early stop.
        logging.error(
            f"Error during phase 1, proceeding to phase 2 anyway. Error: {e=}"
        )

    # Phase 2: Check all submissions
    for response, stdout, stderr in submissions:
        status, submission_url = await check_submission_status_with_retries(
            f"{stdout}{stderr}"
        )

        if status == "AC":
            stats.solution_passed += 1
        else:
            stats.solution_failed += 1

        response.correctness_is_correct = status == "AC"

        if submission_url:
            logging.info(f"Submission URL: {submission_url} ({status=})")
        else:
            logging.warning("No submission URL found")

        results.append((response, extract_cpp_code(response)))

    return results, stats


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=1,
    help="Maximum retries for failed requests",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--prefix",
    "-prefix",
    type=int,
    default=None,
    help="Only process the first N answers",
)
def main(
    input_yaml: str,
    max_retries: int,
    verbose: bool,
    prefix: Optional[int],
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    env = os.environ.copy()
    if "ALWAYS_DELETE_ATCODER_CACHE_BEFORE_LOGIN" not in env:
        raise ValueError(
            "ALWAYS_DELETE_ATCODER_CACHE_BEFORE_LOGIN must be set, and 17th Feb 2025 https://github.com/arthurdupe/atcoder-tools must be used"
        )

    ATCODER_ACCOUNTS.append(
        AtCoderAccount(
            username=env["ATCODER_USERNAME"],
            password=env["ATCODER_PASSWORD"],
        )
    )
    while f"ATCODER_USERNAME_{len(ATCODER_ACCOUNTS) + 1}" in env:
        ATCODER_ACCOUNTS.append(
            AtCoderAccount(
                username=env[f"ATCODER_USERNAME_{len(ATCODER_ACCOUNTS) + 1}"],
                password=env[f"ATCODER_PASSWORD_{len(ATCODER_ACCOUNTS) + 1}"],
            )
        )

    input_path = Path(input_yaml)
    model_responses, model_id = load_atcoder_model_responses(input_path, prefix)
    logging.info(f"Loaded {len(model_responses)} model_responses to evaluate")

    results, stats = asyncio.run(
        evaluate_model_responses(
            model_responses=model_responses,
            max_retries=max_retries,
        )
    )

    path = save_correct_results(
        results, model_id=model_id, path=input_path, stats=stats
    )
    logging.info(f"Saved correct-only results to {path}")


if __name__ == "__main__":
    main()
