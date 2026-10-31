"""Dataset construction and windowing helpers.

Signals are read from the interchange format (headerless little-endian
``float32`` samples plus a JSON metadata file). Clean records are windowed for
unsupervised training; anomalous records additionally carry ``.label.json``
spans, which are mapped onto windows to produce ground-truth labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SAMPLE_DTYPE = np.dtype("<f4")

InputMode = Literal["raw", "fft", "both"]
INPUT_MODES: tuple[str, ...] = ("raw", "fft", "both")

WINDOW_DEFAULT = 1024
STRIDE_DEFAULT = 512


@dataclass(frozen=True)
class AnomalySpan:
    """Sample span of one labelled anomaly; ``end_sample`` is exclusive."""

    type: str
    start_sample: int
    end_sample: int

    def overlaps(self, start: int, end: int) -> bool:
        return start < self.end_sample and end > self.start_sample


@dataclass(frozen=True)
class SignalRecord:
    """One signal plus its metadata and (optionally) its anomaly spans."""

    base: Path
    samples: np.ndarray
    metadata: dict[str, Any]
    anomalies: tuple[AnomalySpan, ...] = ()

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)

    @property
    def sample_rate_hz(self) -> float:
        return float(self.metadata["sample_rate_hz"])


@dataclass(frozen=True)
class Normalizer:
    """Per-feature standardisation statistics fitted on clean training windows."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Sequence[float]]) -> "Normalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )

    @classmethod
    def fit(cls, features: np.ndarray) -> "Normalizer":
        mean = features.mean(axis=0).astype(np.float32)
        std = features.std(axis=0).astype(np.float32)
        # Constant features would otherwise divide by zero.
        std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)


def read_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    if metadata.get("version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema version {metadata.get('version')!r}")
    if metadata.get("dtype") != "float32":
        raise ValueError(f"{path}: unsupported dtype {metadata.get('dtype')!r}")
    for key in ("samples", "sample_rate_hz"):
        if key not in metadata:
            raise ValueError(f"{path}: metadata is missing '{key}'")
    return metadata


def read_anomaly_spans(path: Path) -> tuple[AnomalySpan, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    spans = []
    for entry in document.get("anomalies", []):
        spans.append(
            AnomalySpan(
                type=str(entry["type"]),
                start_sample=int(entry["start_sample"]),
                end_sample=int(entry["end_sample"]),
            )
        )
    return tuple(spans)


def load_record(base: Path, *, with_labels: bool) -> SignalRecord:
    """Load one ``<name>.bin`` / ``<name>.json`` pair, plus its label file."""
    base = Path(base)
    if base.name.endswith(".bin"):
        base = base.with_name(base.name[: -len(".bin")])

    samples = np.fromfile(base.with_name(base.name + ".bin"), dtype=SAMPLE_DTYPE)
    metadata = read_metadata(base.with_name(base.name + ".json"))
    if samples.size != int(metadata["samples"]):
        raise ValueError(
            f"{base.name}: {samples.size} samples on disk but metadata declares "
            f"{metadata['samples']}"
        )

    anomalies: tuple[AnomalySpan, ...] = ()
    if with_labels:
        label_file = base.with_name(base.name + ".label.json")
        if not label_file.is_file():
            raise ValueError(f"{base.name}: missing label file {label_file.name}")
        anomalies = read_anomaly_spans(label_file)

    return SignalRecord(base=base, samples=samples, metadata=metadata, anomalies=anomalies)


def list_records(directory: Path, *, with_labels: bool) -> list[SignalRecord]:
    """Load every signal in ``directory``, sorted by name.

    With ``with_labels`` only labelled anomalous records are returned and
    unlabelled files are skipped; otherwise only unlabelled clean records are
    returned so that anomalous files never leak into training.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"directory {directory} does not exist")

    records: list[SignalRecord] = []
    for path in sorted(directory.glob("*.bin")):
        has_label = path.with_name(path.stem + ".label.json").is_file()
        if has_label != with_labels:
            continue
        try:
            records.append(load_record(path, with_labels=with_labels))
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"{path.name}: {error}") from error

    logger.debug("loaded %d record(s) from %s", len(records), directory)
    return records


def window_starts(count: int, window: int, stride: int) -> np.ndarray:
    """Start offsets of every full window covering ``count`` samples."""
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    if count < window:
        return np.empty(0, dtype=np.int64)
    last = count - window
    starts = np.arange(0, last + 1, stride, dtype=np.int64)
    if starts.size == 0 or starts[-1] != last:
        # Always cover the tail of the record with one final window.
        starts = np.append(starts, last)
    return starts


def window_array(samples: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Split ``samples`` into ``(window_count, window)`` float32 windows."""
    starts = window_starts(int(samples.size), window, stride)
    if starts.size == 0:
        return np.empty((0, window), dtype=np.float32)
    index = starts[:, None] + np.arange(window, dtype=np.int64)[None, :]
    return np.asarray(samples[index], dtype=np.float32)


def fft_features(windows: np.ndarray) -> np.ndarray:
    """Positive-half real-FFT magnitude of every window."""
    magnitudes = np.abs(np.fft.rfft(windows, axis=1))
    return np.asarray(magnitudes, dtype=np.float32)


def to_input_features(windows: np.ndarray, mode: InputMode) -> np.ndarray:
    """Convert raw windows to the model input layout for ``mode``."""
    if mode == "raw":
        return np.asarray(windows, dtype=np.float32)
    if mode == "fft":
        return fft_features(windows)
    if mode == "both":
        stacked = (np.asarray(windows, dtype=np.float32), fft_features(windows))
        return np.concatenate(stacked, axis=1)
    raise ValueError(f"unknown input mode {mode!r}; expected one of {INPUT_MODES}")


def extract_features(
    samples: np.ndarray, *, window: int, stride: int, mode: InputMode
) -> tuple[np.ndarray, np.ndarray]:
    """Window a record and return ``(features, window_starts)``."""
    windows = window_array(samples, window, stride)
    starts = window_starts(int(samples.size), window, stride)
    return to_input_features(windows, mode), starts


def window_labels(starts: np.ndarray, window: int, spans: Iterable[AnomalySpan]) -> np.ndarray:
    """Mark windows that overlap at least one anomaly span."""
    spans = tuple(spans)
    labels = np.zeros(starts.size, dtype=np.int64)
    for index, start in enumerate(starts):
        end = int(start) + window
        if any(span.overlaps(int(start), end) for span in spans):
            labels[index] = 1
    return labels


def window_anomaly_types(
    starts: np.ndarray, window: int, spans: Iterable[AnomalySpan]
) -> list[list[str]]:
    """Anomaly types touching each window, used for reporting."""
    spans = tuple(spans)
    result: list[list[str]] = []
    for start in starts:
        end = int(start) + window
        result.append(sorted({span.type for span in spans if span.overlaps(int(start), end)}))
    return result


def concatenate_features(records: Sequence[SignalRecord], *, window: int, stride: int,
                         mode: InputMode) -> tuple[np.ndarray, list[Path]]:
    """Window several records and stack the results."""
    blocks: list[np.ndarray] = []
    owners: list[Path] = []
    for record in records:
        features, _ = extract_features(
            record.samples, window=window, stride=stride, mode=mode
        )
        if features.shape[0] == 0:
            logger.debug("skipping %s: shorter than one window", record.name)
            continue
        blocks.append(features)
        owners.append(record.base)

    if not blocks:
        return np.empty((0, 0), dtype=np.float32), owners
    return np.concatenate(blocks, axis=0), owners


class WindowDataset(Dataset):
    """Tensor-backed dataset of (already normalised) feature windows."""

    def __init__(self, features: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels: torch.Tensor | None = None

    @classmethod
    def from_arrays(cls, features: np.ndarray, labels: np.ndarray | None = None) -> "WindowDataset":
        dataset = cls(features)
        if labels is not None:
            dataset.labels = torch.as_tensor(labels, dtype=torch.float32)
        return dataset

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0


@dataclass(frozen=True)
class WindowedData:
    """Training-ready window sets produced by :func:`build_datasets`."""

    train: WindowDataset
    validation: WindowDataset
    anomalous: WindowDataset
    anomalous_labels: np.ndarray
    anomalous_starts: list[np.ndarray]
    anomalous_types: list[list[list[str]]]
    anomalous_files: list[Path]
    normalizer: Normalizer
    input_dim: int

    @property
    def train_windows(self) -> int:
        return len(self.train)

    @property
    def validation_windows(self) -> int:
        return len(self.validation)

    @property
    def anomalous_windows(self) -> int:
        return len(self.anomalous)


def split_features(
    features: np.ndarray, *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle and split windows into train/validation sets (~80/20 by default)."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    count = int(features.shape[0])
    if count < 2:
        raise ValueError(f"need at least 2 windows to split, got {count}")

    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    boundary = int(round(count * (1.0 - validation_fraction)))
    boundary = min(max(boundary, 1), count - 1)
    shuffled = features[order]
    return shuffled[:boundary], shuffled[boundary:]


def build_datasets(
    clean_dir: Path,
    anomalous_dir: Path,
    *,
    window: int = WINDOW_DEFAULT,
    stride: int = STRIDE_DEFAULT,
    mode: InputMode = "raw",
    validation_fraction: float = 0.2,
    seed: int = 42,
    normalise: bool = True,
) -> WindowedData:
    """Load, window, normalise and split the clean and anomalous corpora."""
    if mode not in INPUT_MODES:
        raise ValueError(f"unknown input mode {mode!r}; expected one of {INPUT_MODES}")

    clean_records = list_records(clean_dir, with_labels=False)
    if not clean_records:
        raise ValueError(f"no clean signals found in {clean_dir}")

    clean_features, _ = concatenate_features(
        clean_records, window=window, stride=stride, mode=mode
    )
    if clean_features.shape[0] < 2:
        raise ValueError(
            f"clean corpus yields {clean_features.shape[0]} window(s); "
            f"need at least 2 for a train/validation split"
        )

    train_raw, validation_raw = split_features(
        clean_features, validation_fraction=validation_fraction, seed=seed
    )
    normalizer = Normalizer.fit(train_raw) if normalise else Normalizer(
        mean=np.zeros(clean_features.shape[1], dtype=np.float32),
        std=np.ones(clean_features.shape[1], dtype=np.float32),
    )
    train = WindowDataset(normalizer.transform(train_raw))
    validation = WindowDataset(normalizer.transform(validation_raw))

    anomalous_blocks: list[np.ndarray] = []
    anomalous_labels: list[np.ndarray] = []
    anomalous_starts: list[np.ndarray] = []
    anomalous_types: list[list[list[str]]] = []
    anomalous_files: list[Path] = []

    if Path(anomalous_dir).is_dir():
        for record in list_records(anomalous_dir, with_labels=True):
            features, starts = extract_features(
                record.samples, window=window, stride=stride, mode=mode
            )
            if features.shape[0] == 0:
                logger.debug("skipping %s: shorter than one window", record.name)
                continue
            anomalous_blocks.append(normalizer.transform(features))
            anomalous_labels.append(window_labels(starts, window, record.anomalies))
            anomalous_starts.append(starts)
            anomalous_types.append(window_anomaly_types(starts, window, record.anomalies))
            anomalous_files.append(record.base)
    else:
        logger.warning("anomalous directory %s does not exist", anomalous_dir)

    if anomalous_blocks:
        anomalous_features = np.concatenate(anomalous_blocks, axis=0)
        labels = np.concatenate(anomalous_labels)
    else:
        anomalous_features = np.empty((0, train.input_dim), dtype=np.float32)
        labels = np.empty(0, dtype=np.int64)

    return WindowedData(
        train=train,
        validation=validation,
        anomalous=WindowDataset.from_arrays(anomalous_features, labels),
        anomalous_labels=labels,
        anomalous_starts=anomalous_starts,
        anomalous_types=anomalous_types,
        anomalous_files=anomalous_files,
        normalizer=normalizer,
        input_dim=train.input_dim,
    )

