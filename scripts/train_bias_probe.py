#!/usr/bin/env python3
import argparse
import logging
import uuid
from pathlib import Path

import torch

from chainscope.bias_probing import train_bias_probe
from chainscope.typing import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train attn probes")
    parser.add_argument(
        "-r",
        "--resids-dir",
        type=Path,
        required=True,
        help="Path to the directory containing the activations",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=True,
        help="Name of the model",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        required=True,
        help="Layer of the model to probe",
    )
    parser.add_argument(
        "--loc",
        type=str,
        required=True,
        help="Location of the activations to probe",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=8192,
        help="Model dimension (should match the model being probed)",
    )
    parser.add_argument(
        "--weight-init-range",
        type=float,
        default=0.02,
        help="Range for weight initialization",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for Adam optimizer",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adam optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20 * 100,
        help="Number of steps to wait for improvement before early stopping",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30_000,
        help="Maximum number of steps (batches) to train for",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=100,
        help="Run validation every N steps (batches)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bias-probes",
        help="Wandb project name",
    )
    parser.add_argument(
        "--cv-n-folds",
        "--cvnf",
        type=int,
        default=10,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--cv-fold-nrs",
        type=str,
        default="all",
        help="Comma-separated list of folds to train on, defaults to all folds.",
    )
    parser.add_argument(
        "--cv-seed",
        "--cvs",
        type=int,
        default=0,
        help="Random seed for reproducibility of cross-validation (data folds splitting).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of training data to use as validation set",
    )
    parser.add_argument(
        "--train-seed",
        "--ts",
        type=int,
        default=0,
        help="Random seed for reproducibility of training (data shuffling and weight initialization)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.info("Running with arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    if args.cv_fold_nrs == "all":
        cv_fold_nrs = list(range(args.cv_n_folds))
    else:
        cv_fold_nrs = [int(f) for f in args.cv_fold_nrs.split(",")]

    experiment_uuid = uuid.uuid4().hex[:8]
    for cv_test_fold in cv_fold_nrs:
        train_bias_probe(args, experiment_uuid, cv_test_fold)


if __name__ == "__main__":
    main(parse_args())
