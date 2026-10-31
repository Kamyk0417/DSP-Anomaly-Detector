"""Command-line interface for the anomaly injector."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from . import config as config_module
from .anomalies import apply_anomalies
from .io import SignalFile, list_signal_bases, read_signal, signal_base, write_signal
from .labels import build_label, write_label

logger = logging.getLogger("injector")

DEFAULT_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m injector",
        description=(
            "Apply YAML-configured anomalies to clean signals and write "
            ".bin/.json/.label.json outputs. Relative directories are resolved "
            "against the current working directory."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the anomalies YAML file, e.g. configs/anomalies.yaml",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="override the input_dir from the configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the output_dir from the configuration",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only report warnings and errors",
    )
    return parser


def inject_file(
    base: Path,
    output_dir: Path,
    config: config_module.InjectionConfig,
) -> Path:
    """Inject every configured anomaly into one signal and write the outputs."""
    source = read_signal(base)
    samples, label_records = apply_anomalies(
        source.samples,
        config.requests(),
        sample_rate_hz=source.sample_rate_hz,
        seed=config.seed,
    )

    metadata = dict(source.metadata)
    metadata["samples"] = int(samples.size)

    target_base = Path(output_dir) / signal_base(base).name
    write_signal(target_base, SignalFile(samples=samples, metadata=metadata))

    source_clean_file = f"{Path(config.input_dir).name}/{signal_base(base).name}"
    write_label(target_base, build_label(source_clean_file, label_records))
    return target_base


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format=DEFAULT_LOG_FORMAT,
        stream=sys.stderr,
    )

    try:
        config = config_module.load_config(args.config)
    except config_module.ConfigError as error:
        logger.error("%s", error)
        return 1

    input_dir = Path(args.input_dir) if args.input_dir is not None else config.input_dir
    output_dir = Path(args.output_dir) if args.output_dir is not None else config.output_dir

    if not input_dir.is_dir():
        logger.error("input directory %s does not exist", input_dir)
        return 1
    if input_dir.resolve() == output_dir.resolve():
        logger.error("output directory must differ from the input directory: %s", output_dir)
        return 1

    bases = list_signal_bases(input_dir)
    if not bases:
        logger.warning("no .bin files found in %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for base in bases:
        try:
            target = inject_file(base, output_dir, config)
        except (OSError, ValueError) as error:
            logger.error("%s: %s", base.name, error)
            failures += 1
            continue
        logger.info("wrote %s", target.name)

    if failures:
        logger.error("%d of %d signal(s) failed", failures, len(bases))
        return 1

    logger.info("injected %d anomaly type(s) into %d signal(s)", len(config.anomalies), len(bases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

