"""Unsupervised reconstruction-based anomaly detection for oscilloscope signals."""

from __future__ import annotations

from typing import Any

from .data import (
    AnomalySpan,
    InputMode,
    Normalizer,
    SignalRecord,
    WindowDataset,
    WindowedData,
    build_datasets,
    list_records,
    load_record,
    window_array,
    window_starts,
)
from .metrics import evaluate, threshold_from_mean_std, threshold_from_percentile
from .models import ARCHITECTURES, build_model, load_checkpoint, save_checkpoint

__version__ = "0.1.0"

# ``detector.train`` and ``detector.detect`` are also runnable as modules, so they
# are resolved lazily: importing them from the package __init__ would make
# ``python -m detector.train`` warn that the module is already in sys.modules.
_LAZY_EXPORTS: dict[str, str] = {
    "Detector": "detect",
    "TrainConfig": "train",
    "TrainConfigError": "train",
    "detect_file": "detect",
    "detect_samples": "detect",
    "fit": "train",
    "load_detector": "detect",
    "load_train_config": "train",
    "train": "train",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "ARCHITECTURES",
    "AnomalySpan",
    "Detector",
    "InputMode",
    "Normalizer",
    "SignalRecord",
    "TrainConfig",
    "TrainConfigError",
    "WindowDataset",
    "WindowedData",
    "__version__",
    "build_datasets",
    "build_model",
    "detect_file",
    "detect_samples",
    "evaluate",
    "fit",
    "list_records",
    "load_checkpoint",
    "load_detector",
    "load_record",
    "load_train_config",
    "save_checkpoint",
    "threshold_from_mean_std",
    "threshold_from_percentile",
    "train",
    "window_array",
    "window_starts",
]

