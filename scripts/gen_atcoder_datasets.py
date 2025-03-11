#!/usr/bin/env python3

import datetime
import json
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Tuple
from zoneinfo import ZoneInfo

import click
import urllib3
from atcodertools.client.atcoder import AtCoderClient
from atcodertools.client.models.contest import Contest
from atcodertools.client.models.problem import Problem
from atcodertools.client.models.problem_content import is_japanese
from atcodertools.client.models.sample import Sample
from beartype import beartype
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from chainscope.typing import DATA_DIR

load_dotenv()

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if verbose else logging.WARNING,
    )


@dataclass
class ContestWithStartTime:
    """Contest with its parsed start time."""

    contest: Contest
    start_time: datetime.datetime


def remove_jp_characters(content):
    return "".join([x for x in content if not is_japanese(x)])


def clean_math_text(text: str) -> str:
    """Clean up LaTeX-style backslashes and extra spaces."""
    # Replace LaTeX commands with their plain text equivalents
    replacements = {
        r"\ldots": "...",
        r"\cdots": "...",
        r"\dots": "...",
        r"\\": " ",  # Replace double backslash with space
        r"\,": " ",  # Replace LaTeX thin space with regular space
        r"\;": " ",  # Replace LaTeX medium space with regular space
        r"\:": " ",  # Replace LaTeX thick space with regular space
        r"\ ": " ",  # Replace LaTeX space with regular space
        "\r\n": "\n",
    }

    for latex, plain in replacements.items():
        text = text.replace(latex, plain)

    # Clean up multiple spaces
    return " ".join(text.split())


def get_text_with_math(element) -> str:
    """Extract text from HTML while preserving spaces around math expressions."""
    result = []

    # Handle h3 headers specially
    header = element.find("h3")
    if header:
        result.append(f"{header.get_text(strip=True)}: ")
        # Remove the header from further processing
        header.extract()

    # Handle lists specially
    li_items = []
    for li in element.find_all("li"):
        item_result = []
        for content in li.contents:
            if isinstance(content, str):
                item_result.append(content.strip())
            elif content.name == "var":
                if item_result and not item_result[-1].endswith(" "):
                    item_result.append(" ")
                item_result.append(content.get_text(strip=True))
                if content.next_sibling and (
                    not isinstance(content.next_sibling, str)
                    or (
                        isinstance(content.next_sibling, str)
                        and content.next_sibling.strip()
                        and content.next_sibling.strip()[0] not in ".,)}"
                    )
                ):
                    item_result.append(" ")
            else:
                item_result.append(content.get_text(strip=True))
        if item_result:
            li_items.append("".join(item_result))

    if li_items:
        result.append(" \n ".join(li_items))
        return "".join(result)

    # Process each paragraph and pre tag separately for better spacing
    paragraphs = []
    for tag in element.find_all(["p", "pre"]):
        if tag.name == "pre":
            # Handle pre tags specially to preserve format
            pre_result = []
            for content in tag.contents:
                if isinstance(content, str):
                    pre_result.append(content.rstrip())
                elif content.name == "var":
                    pre_content = content.get_text(strip=True)
                    pre_content = clean_math_text(pre_content)
                    pre_result.append(pre_content)
                    # Add newline after var if it's not followed by another var
                    next_sibling = content.next_sibling
                    if not (next_sibling and next_sibling.name == "var"):
                        pre_result.append("\n")
                else:
                    pre_result.append(content.get_text(strip=True))
            paragraphs.append(
                "\n".join(line.rstrip() for line in "".join(pre_result).splitlines())
            )
        else:
            # Normal paragraph handling
            p_result = []
            for content in tag.contents:
                if isinstance(content, str):
                    p_result.append(content.strip().replace("\r\n", "\n"))
                else:
                    if content.name == "var":
                        if p_result and not p_result[-1].endswith(" "):
                            p_result.append(" ")

                    p_content = content.get_text(strip=True)
                    p_content = clean_math_text(p_content)
                    p_result.append(p_content)

                    next_sibling = content.next_sibling
                    if next_sibling and (
                        not isinstance(next_sibling, str)
                        or (
                            isinstance(next_sibling, str)
                            and next_sibling.strip()
                            and next_sibling.strip()[0] not in ".,)}"
                        )
                    ):
                        p_result.append(" ")

            paragraphs.append("".join(p_result))

    if paragraphs:
        result.extend(paragraphs)
        return "\n\n".join(result)

    # Fallback for non-paragraph content
    for content in element.contents:
        if isinstance(content, str):
            result.append(content.strip().replace("\r\n", "\n"))
        elif content.name == "var":
            if result and not result[-1].endswith(" "):
                result.append(" ")

            var_content = content.get_text(strip=True).replace("\r\n", "\n")
            var_content = clean_math_text(var_content)
            result.append(var_content)

            next_sibling = content.next_sibling
            if next_sibling and (
                not isinstance(next_sibling, str)
                or (
                    isinstance(next_sibling, str)
                    and next_sibling.strip()
                    and next_sibling.strip()[0] not in ".,)}"
                )
            ):
                result.append(" ")
        else:
            result.append(content.get_text(strip=True))
    return "".join(result)


class ProblemEnglishContent:
    """Parse and store problem content in English."""

    def __init__(self) -> None:
        self.statement: str | None = None
        self.constraints: str | None = None
        self.io_format: str | None = None
        self.samples: list[
            tuple[Sample, str | None]
        ] = []  # List of (sample, explanation)

    @classmethod
    def from_html(cls, html: str) -> "ProblemEnglishContent":
        """Create ProblemEnglishContent from HTML."""
        content = cls()
        soup = BeautifulSoup(html, "html.parser")

        # Find English content
        lang_en = soup.find("span", class_="lang-en")
        if not lang_en:
            raise ValueError("No English content found")

        # Get all parts
        parts = lang_en.find_all("div", class_="part")
        if not parts:
            raise ValueError("No parts found in the problem")

        # First part is always the problem statement
        content.statement = get_text_with_math(parts[0].find("section"))

        # Second part is constraints
        if len(parts) > 1:
            content.constraints = get_text_with_math(parts[1].find("section"))

        # Find IO format section
        io_style = lang_en.find("div", class_="io-style")
        if io_style:
            # Process each part in io-style separately
            io_parts = []
            for part in io_style.find_all("div", class_="part"):
                section_text = get_text_with_math(part.find("section"))
                if section_text:
                    io_parts.append(section_text)
            content.io_format = "\n\n".join(io_parts)

        # Get samples (they come in pairs after io-style)
        samples_start = None
        for i, part in enumerate(parts):
            if "Sample Input 1" in part.get_text():
                samples_start = i
                break

        if samples_start is not None:
            # Process samples in pairs
            for i in range(samples_start, len(parts), 2):
                if i + 1 < len(parts):
                    input_text = parts[i].find("pre").get_text(strip=True)
                    input_text = input_text.replace("\r\n", "\n")
                    output_text = parts[i + 1].find("pre").get_text(strip=True)
                    output_text = output_text.replace("\r\n", "\n")
                    sample = Sample(input_text, output_text)

                    # Get explanation if it exists (all <p> tags and lists after the output pre)
                    explanation_parts = []

                    # Process paragraphs
                    for p in parts[i + 1].find_all("p"):
                        explanation_parts.append(get_text_with_math(p))

                    # Process lists
                    for ul in parts[i + 1].find_all("ul"):
                        list_text = get_text_with_math(ul)
                        if list_text:
                            explanation_parts.append(list_text)

                    explanation = None
                    if explanation_parts:
                        explanation = "\n\n".join(explanation_parts)

                    content.samples.append((sample, explanation))

        return content


def env_credential_supplier() -> Tuple[str, str]:
    """Get AtCoder credentials from environment variables."""
    username = os.environ.get("ATCODER_USERNAME")
    password = os.environ.get("ATCODER_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Please set ATCODER_USERNAME and ATCODER_PASSWORD environment variables"
        )

    return username, password


def filter_contests_by_date(
    contests: list[ContestWithStartTime], min_date: datetime.datetime
) -> list[ContestWithStartTime]:
    """Filter contests to keep only those after the minimum date."""
    filtered_contests = []

    for i, contest_with_time in enumerate(contests):
        if contest_with_time.start_time >= min_date:
            logger.info(
                f"Keeping contest {i+1}/{len(contests)} {contest_with_time.contest.contest_id} "
                f"with start time: {contest_with_time.start_time}"
            )
            filtered_contests.append(contest_with_time)
        else:
            logger.info(
                f"Skipping contest {i+1}/{len(contests)} {contest_with_time.contest.contest_id} "
                f"with start time: {contest_with_time.start_time}"
            )

    return filtered_contests


def get_contest_info(contest_with_time: ContestWithStartTime) -> dict[str, Any] | None:
    """Extract relevant information from a contest and its problems."""
    client = AtCoderClient()
    # Disable SSL verification for the session
    client._session.verify = False
    contest = contest_with_time.contest

    try:
        problems: list[Problem] = client.download_problem_list(contest)
        problems_details: list[dict[str, Any]] = []
        for problem in problems:
            # Get problem content in English
            resp = client._request(f"{problem.get_url()}?lang=en")
            problem_content = ProblemEnglishContent.from_html(resp.text)

            problems_detail = {
                "id": problem.problem_id,
                "problem_letter": problem.alphabet,
                "url": f"{problem.get_url()}?lang=en",
                "statement": problem_content.statement,
                "constraints": problem_content.constraints,
                "io_format": problem_content.io_format,
                "samples": [
                    {
                        "input": sample.get_input(),
                        "output": sample.get_output(),
                        "explanation": explanation,
                    }
                    for sample, explanation in problem_content.samples
                ],
            }

            problems_details.append(problems_detail)
            logger.warning(f"Processed problem {problem.problem_id}")

        return {
            "id": contest.contest_id,
            "url": f"{contest.get_url()}?lang=en",
            "start_time": contest_with_time.start_time.isoformat(),
            "problems": problems_details,
        }
    except Exception as e:
        logger.warning(f"Error processing contest {contest.contest_id}: {str(e)}")
        # Print stack trace
        logger.warning(traceback.format_exc())
        return None


def download_all_contests() -> list[ContestWithStartTime]:
    """Download all contests with their start times."""
    contests_with_time = []
    page_num = 1
    previous_contests = set()
    client = AtCoderClient()

    while True:
        logger.info(f"Downloading page {page_num}")

        # Request the page in English
        resp = client._request(
            f"https://atcoder.jp/contests/archive?page={page_num}&lang=en"
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find the contest table
        table = soup.find("table", class_="table")
        if not table:
            break

        # Process each row in the table
        current_contests = set()
        for row in table.find_all("tr")[1:]:  # Skip header row
            cols = row.find_all("td")
            if len(cols) < 2:  # Skip malformed rows
                continue

            # Get contest ID from the URL
            contest_link = cols[1].find("a")
            if not contest_link:
                continue
            contest_id = contest_link["href"].split("/")[-1]

            # Skip if we've seen this contest before
            if contest_id == "archive":
                continue
            current_contests.add(contest_id)

            # Get start time
            time_element = cols[0].find("time", class_="fixtime-full")
            if not time_element:
                logger.warning(f"No time element found for contest {contest_id}")
                continue

            try:
                # Parse the time (format: "2025-02-09 21:00:00+0900")
                start_time = datetime.datetime.strptime(
                    time_element.text,
                    "%Y-%m-%d %H:%M:%S%z",  # 2025-02-09 21:00:00+0900
                )
                # Convert to UTC
                utc_time = start_time.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

                contests_with_time.append(
                    ContestWithStartTime(
                        contest=Contest(contest_id), start_time=utc_time
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Could not parse time str {time_element.text} for contest {contest_id}: {str(e)}"
                )
                continue

        # Check if we've seen all these contests before
        if current_contests == previous_contests:
            break

        previous_contests = current_contests
        page_num += 1

    return sorted(contests_with_time, key=lambda x: x.contest.contest_id)


@click.command()
@click.option(
    "--min-date",
    "-d",
    type=click.DateTime(),
    default="2024-07-01",
    help="Only include contests after this date (inclusive)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress information",
)
@beartype
def main(min_date: datetime.datetime, verbose: bool) -> None:
    """Download AtCoder contests and problems data."""
    setup_logging(verbose)

    # Suppress SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    output_path = DATA_DIR / "atcoder"
    output_path.mkdir(parents=True, exist_ok=True)

    client = AtCoderClient()

    # Login to AtCoder
    logger.warning("Logging in to AtCoder...")
    client.login(credential_supplier=env_credential_supplier)

    # Get all contests with start times
    logger.warning("Downloading contest list...")
    all_contests: list[ContestWithStartTime] = download_all_contests()

    # Filter contests by date
    logger.warning("Filtering contests by date...")
    filtered_contests = filter_contests_by_date(all_contests, min_date)
    logger.warning(f"Found {len(filtered_contests)} contests after {min_date}")

    # Process filtered contests
    dataset = []
    for i, contest_with_time in enumerate(filtered_contests):
        logger.warning(
            f"Processing contest {i+1}/{len(filtered_contests)}: {contest_with_time.contest.contest_id}"
        )
        contest_info = get_contest_info(contest_with_time)
        if contest_info:
            dataset.append(contest_info)

    # Calculate statistics
    total_contests = len(dataset)
    abc_count = sum(1 for contest in dataset if contest["id"].startswith("abc"))
    arc_count = sum(1 for contest in dataset if contest["id"].startswith("arc"))
    agc_count = sum(1 for contest in dataset if contest["id"].startswith("agc"))
    other_count = total_contests - (abc_count + arc_count + agc_count)
    total_problems = sum(len(contest["problems"]) for contest in dataset)

    logger.warning("\nContest Statistics:")
    logger.warning(f"Total contests: {total_contests}")
    logger.warning(f"ABC contests: {abc_count}")
    logger.warning(f"ARC contests: {arc_count}")
    logger.warning(f"AGC contests: {agc_count}")
    logger.warning(f"Other contests: {other_count}")
    logger.warning(f"Total problems: {total_problems}")

    result = {
        "contests": dataset,
        "accessed_timestamp": datetime.datetime.now().isoformat(),
        "min_date": min_date.isoformat(),
        "statistics": {
            "total_contests": total_contests,
            "abc_contests": abc_count,
            "arc_contests": arc_count,
            "agc_contests": agc_count,
            "other_contests": other_count,
            "total_problems": total_problems,
        },
    }

    # Save the dataset
    output_file = output_path / "atcoder_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.warning(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    main()
