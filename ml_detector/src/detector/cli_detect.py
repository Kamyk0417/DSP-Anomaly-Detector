"""Command-line interface for scoring a signal file with a trained model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .detect import detect_file, load_detector, write_detections

logger = logging.getLogger("detector")

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m detector.detect",
        description=(
            "Score one signal with a trained detector. The JSON report goes to "
            "stdout; use --write to also save <name>.detections.json."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="checkpoint written by detector.train, e.g. data/models/ae.pt",
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="signal samples (.bin) to score",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="metadata (.json) for the signal; derived from --file when omitted",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="also write <name>.detections.json next to the input",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="inference device (default: cpu)",
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
        detector = load_detector(args.model, device=args.device)
    except (OSError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1

    try:
        report = detect_file(detector, args.file, args.json)
    except (OSError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1

    if args.write:
        try:
            path = write_detections(args.file, report)
        except OSError as error:
            logger.error("cannot write detections: %s", error)
            return 1
        logger.info("wrote %s", path)

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

