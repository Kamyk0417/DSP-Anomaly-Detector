"""YAML-driven anomaly injection for synthetic oscilloscope signals."""

from __future__ import annotations

from .anomalies import (
    ANOMALY_APPLIERS,
    ANOMALY_TYPES,
    AnomalyRequest,
    apply_anomalies,
    apply_anomaly,
)
from .config import AnomalySpec, ConfigError, InjectionConfig, load_config
from .io import SignalFile, read_signal, signal_base, write_signal
from .labels import build_label, validate_label, write_label

__version__ = "0.1.0"

__all__ = [
    "ANOMALY_APPLIERS",
    "ANOMALY_TYPES",
    "AnomalyRequest",
    "AnomalySpec",
    "ConfigError",
    "InjectionConfig",
    "SignalFile",
    "__version__",
    "apply_anomalies",
    "apply_anomaly",
    "build_label",
    "load_config",
    "read_signal",
    "signal_base",
    "validate_label",
    "write_label",
    "write_signal",
]
