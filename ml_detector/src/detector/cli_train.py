"""Command-line interface for training anomaly detectors."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from .train import TrainConfigError, load_train_config, train

logger = logging.getLogger("detector")

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m detector.train",
        description=(
            "Train an unsupervised reconstruction-based anomaly detector on clean "
            "windows and evaluate it against labelled anomalous windows."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the training YAML file, e.g. configs/train.yaml",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only report warnings and errors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format=LOG_FORMAT,
        stream=sys.stderr,
    )

    try:
        config = load_train_config(args.config)
    except TrainConfigError as error:
        logger.error("%s", error)
        return 1

    logger.info(
        "run '%s': %s / %s, window %d stride %d, %d epochs",
        config.run_name,
        config.model.arch,
        config.input.mode,
        config.data.window,
        config.data.stride,
        config.train.epochs,
    )

    try:
        result = train(config)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    except ValueError as error:
        logger.error("training failed: %s", error)
        return 1

    logger.info("checkpoint %s", result.checkpoint)
    logger.info("metrics    %s", result.metrics_file)
    logger.info("config     %s", result.config_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

