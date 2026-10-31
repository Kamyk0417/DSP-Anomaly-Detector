"""Precision, recall, F1, AUC and confusion-matrix helpers."""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def threshold_from_percentile(scores: Sequence[float] | np.ndarray, percentile: float) -> float:
    """Score threshold at ``percentile`` of the validation scores."""
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"threshold percentile must be in (0, 100), got {percentile}")
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot pick a threshold from an empty score set")
    return float(np.percentile(values, percentile))


def threshold_from_mean_std(
    scores: Sequence[float] | np.ndarray, k: float = 3.0
) -> float:
    """Score threshold at ``mean + k * std`` of the validation scores."""
    if k < 0.0:
        raise ValueError(f"k must not be negative, got {k}")
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot pick a threshold from an empty score set")
    return float(values.mean() + k * values.std())


def confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(true_negative, false_positive, false_negative, true_positive)``."""
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return tn, fp, fn, tp


def evaluate(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Score a window set against ground truth at a fixed threshold."""
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    if score_values.shape != label_values.shape:
        raise ValueError(
            f"scores and labels must have the same length, got "
            f"{score_values.size} and {label_values.size}"
        )

    predictions = (score_values > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_counts(label_values, predictions)

    positives = int(np.count_nonzero(label_values))
    negatives = int(label_values.size - positives)
    zero_division = 0.0

    precision = float(precision_score(label_values, predictions, zero_division=zero_division))
    recall = float(recall_score(label_values, predictions, zero_division=zero_division))
    f1 = float(f1_score(label_values, predictions, zero_division=zero_division))

    auc: float | None = None
    if positives > 0 and negatives > 0:
        auc = float(roc_auc_score(label_values, score_values))
    else:
        logger.debug("AUC undefined: %d positive / %d negative windows", positives, negatives)

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "windows": int(label_values.size),
        "anomalous_windows": positives,
        "normal_windows": negatives,
        "flagged_windows": int(np.count_nonzero(predictions)),
    }


def summarise_scores(scores: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Descriptive statistics of a score distribution."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarise an empty score set")
    return {
        "count": float(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(values.max()),
    }


def json_safe(value: Any) -> Any:
    """Replace non-finite floats so the metrics file stays valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

