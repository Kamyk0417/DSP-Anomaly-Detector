"""Inference helpers for scoring signals and windows.

A checkpoint written by :mod:`detector.train` is self-contained: it carries the
architecture, the windowing parameters, the input mode, the normalisation
statistics and the detection threshold, so scoring never needs the training YAML.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    INPUT_MODES,
    InputMode,
    Normalizer,
    WindowDataset,
    extract_features,
    read_metadata,
)
from .models import BaseAutoencoder, load_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class Detector:
    """A trained model plus the pre-processing and threshold it was tuned with."""

    model: BaseAutoencoder
    config: dict[str, Any]
    normalizer: Normalizer
    threshold: float
    window: int
    stride: int
    mode: InputMode
    device: torch.device

    @property
    def run_name(self) -> str:
        return str(self.config.get("run_name", "detector"))

    @property
    def sequence_length(self) -> int:
        return int(self.config["sequence_length"])

    def score_features(self, features: np.ndarray, *, batch_size: int = DEFAULT_BATCH_SIZE,
                       normalise: bool = True) -> np.ndarray:
        """Reconstruction error for each row of ``features``."""
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        if features.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        prepared = self.normalizer.transform(features) if normalise else features
        dataset = WindowDataset(prepared)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        scores: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                scores.append(
                    self.model.reconstruction_error(batch.to(self.device)).cpu().numpy()
                )
        return np.concatenate(scores).astype(np.float64)

    def score_samples(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Window a record and return ``(scores, window_starts)``."""
        features, starts = extract_features(
            samples, window=self.window, stride=self.stride, mode=self.mode
        )
        return self.score_features(features), starts


def load_detector(path: Path, device: torch.device | str = "cpu") -> Detector:
    """Load a detector checkpoint."""
    model, config = load_checkpoint(path, device=device)

    mode = str(config.get("input_mode", "raw"))
    if mode not in INPUT_MODES:
        raise ValueError(f"checkpoint declares unknown input mode {mode!r}")
    if "threshold" not in config:
        raise ValueError("checkpoint is missing the detection threshold")
    if "normalizer" not in config:
        raise ValueError("checkpoint is missing the normalisation statistics")

    return Detector(
        model=model,
        config=config,
        normalizer=Normalizer.from_dict(config["normalizer"]),
        threshold=float(config["threshold"]),
        window=int(config["window"]),
        stride=int(config["stride"]),
        mode=mode,  # type: ignore[arg-type]
        device=torch.device(device),
    )


def merge_detections(
    starts: Sequence[int],
    scores: Sequence[float],
    flagged: Sequence[bool],
    window: int,
    stride: int,
) -> list[dict[str, Any]]:
    """Merge stride-adjacent flagged windows into sample-span detections."""
    detections: list[dict[str, Any]] = []
    index = 0
    count = len(starts)

    while index < count:
        if not flagged[index]:
            index += 1
            continue

        first = index
        peak = float(scores[index])
        while (
            index + 1 < count
            and flagged[index + 1]
            and int(starts[index + 1]) == int(starts[index]) + stride
        ):
            index += 1
            peak = max(peak, float(scores[index]))

        detections.append(
            {
                "start_sample": int(starts[first]),
                "end_sample": int(starts[index]) + window,
                "peak_score": peak,
                "windows": index - first + 1,
            }
        )
        index += 1

    return detections


def detect_samples(detector: Detector, samples: np.ndarray) -> dict[str, Any]:
    """Score a whole record and describe the flagged regions."""
    scores, starts = detector.score_samples(samples)
    if scores.size == 0:
        return {
            "window": detector.window,
            "stride": detector.stride,
            "threshold": detector.threshold,
            "windows": 0,
            "flagged_windows": 0,
            "max_score": None,
            "detections": [],
            "anomalous": False,
        }

    flagged = scores > detector.threshold
    detections = merge_detections(starts, scores, flagged, detector.window, detector.stride)
    return {
        "window": detector.window,
        "stride": detector.stride,
        "threshold": detector.threshold,
        "windows": int(scores.size),
        "flagged_windows": int(np.count_nonzero(flagged)),
        "max_score": float(scores.max()),
        "detections": detections,
        "anomalous": bool(detections),
    }


def detect_file(
    detector: Detector, bin_file: Path, json_file: Path | None = None
) -> dict[str, Any]:
    """Score one ``.bin``/``.json`` pair and return a JSON-ready report."""
    bin_file = Path(bin_file)
    base = bin_file.with_name(bin_file.name[: -len(".bin")]) if bin_file.name.endswith(
        ".bin"
    ) else bin_file
    metadata_file = Path(json_file) if json_file is not None else base.with_name(
        base.name + ".json"
    )
    samples_file = base.with_name(base.name + ".bin")
    if not samples_file.is_file():
        raise FileNotFoundError(f"signal file {samples_file} does not exist")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"metadata file {metadata_file} does not exist")

    metadata = read_metadata(metadata_file)
    samples = np.fromfile(samples_file, dtype="<f4")
    if samples.size != int(metadata["samples"]):
        raise ValueError(
            f"{base.name}: {samples.size} samples on disk but metadata declares "
            f"{metadata['samples']}"
        )

    report = detect_samples(detector, samples)
    report.update(
        {
            "file": base.name + ".bin",
            "json": metadata_file.name,
            "samples": int(samples.size),
            "sample_rate_hz": float(metadata["sample_rate_hz"]),
            "model": detector.run_name,
            "input_mode": detector.mode,
        }
    )
    return report


def detections_path(bin_file: Path) -> Path:
    """``<name>.detections.json`` next to the input."""
    bin_file = Path(bin_file)
    if bin_file.name.endswith(".bin"):
        bin_file = bin_file.with_name(bin_file.name[: -len(".bin")])
    return bin_file.with_name(bin_file.name + ".detections.json")


def write_detections(bin_file: Path, report: dict[str, Any]) -> Path:
    path = detections_path(bin_file)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m detector.detect`` entry point."""
    from .cli_detect import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

