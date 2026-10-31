"""Writer for ``.label.json`` anomaly label files.

Label files use interchange schema version 1 and exist only next to anomalous
signals; clean signals never carry one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from .io import label_path

LABEL_VERSION = 1

logger = logging.getLogger(__name__)


def build_label(source_clean_file: str, anomalies: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Assemble a label document for one anomalous signal."""
    label: dict[str, Any] = {
        "version": LABEL_VERSION,
        "source_clean_file": source_clean_file,
        "anomalies": [dict(anomaly) for anomaly in anomalies],
    }
    validate_label(label)
    return label


def validate_label(label: dict[str, Any]) -> None:
    """Check a label document before it is written."""
    if label.get("version") != LABEL_VERSION:
        raise ValueError(f"unsupported label version {label.get('version')!r}")
    if not isinstance(label.get("source_clean_file"), str):
        raise ValueError("label source_clean_file must be a string")

    anomalies = label.get("anomalies")
    if not isinstance(anomalies, list) or not anomalies:
        raise ValueError("label must describe at least one anomaly")

    previous_start = -1
    for anomaly in anomalies:
        for key in ("type", "start_sample", "end_sample", "params"):
            if key not in anomaly:
                raise ValueError(f"label anomaly is missing '{key}'")
        start = int(anomaly["start_sample"])
        end = int(anomaly["end_sample"])
        if start < 0 or end <= start:
            raise ValueError(f"invalid anomaly span {start}..{end}")
        if start < previous_start:
            raise ValueError("label anomalies must be ordered by start_sample")
        previous_start = start


def write_label(base: Path, label: dict[str, Any]) -> None:
    validate_label(label)
    path = label_path(base)
    path.write_text(json.dumps(label, indent=2), encoding="utf-8")
    logger.debug("wrote %s", path.name)

