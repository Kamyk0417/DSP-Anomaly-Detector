"""Read and write ``.bin`` / ``.json`` signal files (interchange format version 1).

A signal is a headerless little-endian ``float32`` sample file paired with a JSON
metadata file. The ``.bin`` and ``.json`` names share one base path, so callers
work with that base and the module derives the concrete file names.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
SAMPLE_DTYPE = np.dtype("<f4")

BIN_SUFFIX = ".bin"
JSON_SUFFIX = ".json"
LABEL_SUFFIX = ".label.json"

logger = logging.getLogger(__name__)


@dataclass
class SignalFile:
    samples: np.ndarray
    metadata: dict[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)

    @property
    def sample_rate_hz(self) -> float:
        return float(self.metadata["sample_rate_hz"])


def signal_base(path: Path) -> Path:
    """Return the shared base path, dropping ``.bin``, ``.json`` or ``.label.json``."""
    path = Path(path)
    name = path.name
    for suffix in (LABEL_SUFFIX, JSON_SUFFIX, BIN_SUFFIX):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path


def bin_path(base: Path) -> Path:
    base = signal_base(base)
    return base.with_name(base.name + BIN_SUFFIX)


def json_path(base: Path) -> Path:
    base = signal_base(base)
    return base.with_name(base.name + JSON_SUFFIX)


def label_path(base: Path) -> Path:
    base = signal_base(base)
    return base.with_name(base.name + LABEL_SUFFIX)


def list_signal_bases(directory: Path) -> list[Path]:
    """Return the base paths of all ``.bin`` files in ``directory``, sorted by name."""
    directory = Path(directory)
    return sorted((path for path in directory.glob("*" + BIN_SUFFIX) if path.is_file()))


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Check the metadata against interchange schema version 1."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")

    for key in ("version", "samples", "sample_rate_hz", "dtype", "endianness", "channels"):
        if key not in metadata:
            raise ValueError(f"metadata is missing required field '{key}'")

    version = metadata["version"]
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported metadata version {version!r}; expected {SCHEMA_VERSION}"
        )
    if metadata["dtype"] != "float32":
        raise ValueError(f"unsupported dtype {metadata['dtype']!r}; expected 'float32'")
    if metadata["endianness"] != "little":
        raise ValueError(
            f"unsupported endianness {metadata['endianness']!r}; expected 'little'"
        )
    if int(metadata["channels"]) != 1:
        raise ValueError(f"unsupported channel count {metadata['channels']!r}; expected 1")

    samples = int(metadata["samples"])
    if samples < 0:
        raise ValueError(f"metadata samples must not be negative, got {samples}")
    if not float(metadata["sample_rate_hz"]) > 0.0:
        raise ValueError("metadata sample_rate_hz must be positive")


def read_signal(base: Path) -> SignalFile:
    base = signal_base(base)
    samples = np.fromfile(bin_path(base), dtype=SAMPLE_DTYPE)
    metadata = json.loads(json_path(base).read_text(encoding="utf-8"))
    validate_metadata(metadata)
    if samples.size != int(metadata["samples"]):
        raise ValueError(
            f"{bin_path(base).name}: size mismatch: {samples.size} samples vs "
            f"{metadata['samples']} declared"
        )
    return SignalFile(samples=samples, metadata=metadata)


def write_signal(base: Path, sig: SignalFile) -> None:
    base = signal_base(base)
    declared = int(sig.metadata.get("samples", -1))
    if declared != sig.samples.size:
        raise ValueError(
            f"{base.name}: refusing to write {sig.samples.size} samples while metadata "
            f"declares {declared}"
        )
    validate_metadata(sig.metadata)

    bin_path(base).write_bytes(np.asarray(sig.samples, dtype=SAMPLE_DTYPE).tobytes())
    json_path(base).write_text(json.dumps(sig.metadata, indent=2), encoding="utf-8")
    logger.debug("wrote %s and %s", bin_path(base).name, json_path(base).name)

