#!/usr/bin/env python3
import argparse
import json
import logging
from dataclasses import asdict

from chainscope import DATA_DIR
from chainscope.qs_generation import Question, gen_qs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate questions dataset")
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default="gt",
        choices=["gt", "lt"],
        help="Template name",
    )
    parser.add_argument(
        "--pick-count",
        type=int,
        default=1,
        help="Pick K values to fill in the template",
    )
    parser.add_argument(
        "--min-value-diff",
        type=int,
        default=1,
        help="Minimum value difference between X and Y when filling in the template",
    )
    parser.add_argument(
        "--max-value-diff",
        type=int,
        default=9999,
        help="Maximum value difference between X and Y when filling in the template",
    )
    parser.add_argument(
        "--expected-answer",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Expected answer to the built questions",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    output_dir = DATA_DIR / "qs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_name = f"qs_{args.template}-template_expected-{args.expected_answer}_pick-{args.pick_count}_min-value-diff-{args.min_value_diff}_max-value-diff-{args.max_value_diff}.json"

    qs: list[Question] = gen_qs(
        expected_answer=args.expected_answer,
        template=args.template,
        pick_count=args.pick_count,
        min_value_diff=args.min_value_diff,
        max_value_diff=args.max_value_diff,
        verbose=args.verbose,
    )
    if args.verbose:
        logging.info(f"Generated {len(qs)} questions")

    with open(output_dir / output_file_name, "w") as f:
        json.dump([asdict(q) for q in qs], f)


if __name__ == "__main__":
    main(parse_args())
