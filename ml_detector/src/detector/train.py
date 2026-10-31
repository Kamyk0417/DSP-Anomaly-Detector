"""Training loop for unsupervised reconstruction-based detectors.

Pipeline (section 5.4): split clean windows into train/validation, train on
reconstruction loss, score windows by reconstruction error, pick the threshold on
the validation split, then evaluate on the anomalous set. The run is saved as
``<output>/<run>.pt`` plus ``metrics.json`` and ``config.yaml``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from . import metrics
from .data import (
    INPUT_MODES,
    STRIDE_DEFAULT,
    WINDOW_DEFAULT,
    InputMode,
    Normalizer,
    WindowedData,
    build_datasets,
)
from .models import (
    ARCHITECTURES,
    ARCH_DEFAULT,
    BaseAutoencoder,
    build_model,
    parameter_count,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1
LOSS_NAME = "mse"
THRESHOLD_METHODS: tuple[str, ...] = ("percentile", "mean_std")


class TrainConfigError(RuntimeError):
    """Raised when the training configuration cannot be read or validated."""


@dataclass(frozen=True)
class DataConfig:
    clean_dir: Path = Path("data/clean")
    anomalous_dir: Path = Path("data/anomalous")
    window: int = WINDOW_DEFAULT
    stride: int = STRIDE_DEFAULT
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    arch: str = ARCH_DEFAULT
    latent_dim: int = 32
    hidden: int = 128


@dataclass(frozen=True)
class InputConfig:
    mode: InputMode = "raw"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1.0e-3
    device: str = "cuda"


@dataclass(frozen=True)
class EvalConfig:
    threshold_percentile: float = 99.0
    threshold_method: str = "percentile"
    threshold_k: float = 3.0


@dataclass(frozen=True)
class OutputConfig:
    dir: Path = Path("data/models")
    run_name: str | None = None


@dataclass(frozen=True)
class TrainConfig:
    """Validated contents of ``configs/train.yaml``."""

    version: int = CONFIG_VERSION
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    input: InputConfig = field(default_factory=InputConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def run_name(self) -> str:
        return self.output.run_name or f"{self.model.arch}_{self.input.mode}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "data": {
                "clean_dir": str(self.data.clean_dir),
                "anomalous_dir": str(self.data.anomalous_dir),
                "window": self.data.window,
                "stride": self.data.stride,
                "validation_fraction": self.data.validation_fraction,
            },
            "model": {
                "arch": self.model.arch,
                "latent_dim": self.model.latent_dim,
                "hidden": self.model.hidden,
            },
            "input": {"mode": self.input.mode},
            "train": {
                "batch_size": self.train.batch_size,
                "epochs": self.train.epochs,
                "lr": self.train.lr,
                "device": self.train.device,
            },
            "eval": {
                "threshold_percentile": self.eval.threshold_percentile,
                "threshold_method": self.eval.threshold_method,
                "threshold_k": self.eval.threshold_k,
            },
            "output": {"dir": str(self.output.dir), "run_name": self.output.run_name},
        }


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TrainConfigError(f"'{name}' must be a mapping")
    return value


def _reject_unknown(block: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise TrainConfigError(f"unknown key(s) in '{name}': {', '.join(unknown)}")


def _positive_int(block: dict[str, Any], key: str, name: str, default: int) -> int:
    value = block.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TrainConfigError(f"'{name}.{key}' must be a positive integer, got {value!r}")
    return value


def _positive_float(block: dict[str, Any], key: str, name: str, default: float) -> float:
    value = block.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0.0:
        raise TrainConfigError(f"'{name}.{key}' must be positive, got {value!r}")
    return float(value)


def parse_config(document: dict[str, Any]) -> TrainConfig:
    """Validate a raw YAML mapping into a :class:`TrainConfig`."""
    if not isinstance(document, dict):
        raise TrainConfigError("configuration must be a YAML mapping")

    _reject_unknown(
        document,
        {"version", "seed", "data", "model", "input", "train", "eval", "output"},
        "<root>",
    )

    version = document.get("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise TrainConfigError(f"unsupported config version {version!r}; expected 1")

    seed = document.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TrainConfigError(f"'seed' must be an integer, got {seed!r}")

    data_block = _require_mapping(document.get("data"), "data")
    _reject_unknown(
        data_block,
        {"clean_dir", "anomalous_dir", "window", "stride", "validation_fraction"},
        "data",
    )
    data_config = DataConfig(
        clean_dir=Path(data_block.get("clean_dir", "data/clean")),
        anomalous_dir=Path(data_block.get("anomalous_dir", "data/anomalous")),
        window=_positive_int(data_block, "window", "data", WINDOW_DEFAULT),
        stride=_positive_int(data_block, "stride", "data", STRIDE_DEFAULT),
        validation_fraction=float(data_block.get("validation_fraction", 0.2)),
    )
    if not 0.0 < data_config.validation_fraction < 1.0:
        raise TrainConfigError("'data.validation_fraction' must be in (0, 1)")

    model_block = _require_mapping(document.get("model"), "model")
    _reject_unknown(model_block, {"arch", "latent_dim", "hidden"}, "model")
    arch = str(model_block.get("arch", ARCH_DEFAULT))
    if arch not in ARCHITECTURES:
        raise TrainConfigError(
            f"'model.arch' must be one of {', '.join(ARCHITECTURES)}, got {arch!r}"
        )
    model_config = ModelConfig(
        arch=arch,
        latent_dim=_positive_int(model_block, "latent_dim", "model", 32),
        hidden=_positive_int(model_block, "hidden", "model", 128),
    )

    input_block = _require_mapping(document.get("input"), "input")
    _reject_unknown(input_block, {"mode"}, "input")
    mode = str(input_block.get("mode", "raw"))
    if mode not in INPUT_MODES:
        raise TrainConfigError(
            f"'input.mode' must be one of {', '.join(INPUT_MODES)}, got {mode!r}"
        )

    train_block = _require_mapping(document.get("train"), "train")
    _reject_unknown(train_block, {"batch_size", "epochs", "lr", "device"}, "train")
    device = str(train_block.get("device", "cuda"))
    if device not in {"cuda", "cpu"}:
        raise TrainConfigError(f"'train.device' must be 'cuda' or 'cpu', got {device!r}")
    training_config = TrainingConfig(
        batch_size=_positive_int(train_block, "batch_size", "train", 64),
        epochs=_positive_int(train_block, "epochs", "train", 50),
        lr=_positive_float(train_block, "lr", "train", 1.0e-3),
        device=device,
    )

    eval_block = _require_mapping(document.get("eval"), "eval")
    _reject_unknown(eval_block, {"threshold_percentile", "threshold_method", "threshold_k"}, "eval")
    method = str(eval_block.get("threshold_method", "percentile"))
    if method not in THRESHOLD_METHODS:
        raise TrainConfigError(
            f"'eval.threshold_method' must be one of {', '.join(THRESHOLD_METHODS)}, got {method!r}"
        )
    eval_config = EvalConfig(
        threshold_percentile=float(eval_block.get("threshold_percentile", 99.0)),
        threshold_method=method,
        threshold_k=float(eval_block.get("threshold_k", 3.0)),
    )
    if not 0.0 < eval_config.threshold_percentile < 100.0:
        raise TrainConfigError("'eval.threshold_percentile' must be in (0, 100)")

    output_block = _require_mapping(document.get("output"), "output")
    _reject_unknown(output_block, {"dir", "run_name"}, "output")
    run_name = output_block.get("run_name")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise TrainConfigError("'output.run_name' must be a non-empty string")
    output_config = OutputConfig(
        dir=Path(output_block.get("dir", "data/models")), run_name=run_name
    )

    return TrainConfig(
        version=CONFIG_VERSION,
        seed=seed,
        data=data_config,
        model=model_config,
        input=InputConfig(mode=mode),  # type: ignore[arg-type]
        train=training_config,
        eval=eval_config,
        output=output_config,
    )


def load_train_config(path: Path) -> TrainConfig:
    """Read and validate a training YAML file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise TrainConfigError(f"cannot read configuration {path}: {error}") from error

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise TrainConfigError(f"{path} is not valid YAML: {error}") from error

    return parse_config(document)


@dataclass(frozen=True)
class TrainResult:
    run_name: str
    checkpoint: Path
    metrics_file: Path
    config_file: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TrainedModel:
    """Everything inference needs, assembled during training."""

    model: BaseAutoencoder
    normalizer: Normalizer
    threshold: float
    window: int
    stride: int
    mode: InputMode
    sequence_length: int


def resolve_device(requested: str) -> torch.device:
    """Honour the configured device, falling back to CPU when CUDA is absent."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_loss(model: BaseAutoencoder, loader: DataLoader, criterion: nn.Module,
               device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            features = batch.to(device)
            reconstruction, _ = model(features)
            total += float(criterion(reconstruction, features)) * features.shape[0]
            count += features.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def score_dataset(model: BaseAutoencoder, dataset: Any, batch_size: int,
                  device: torch.device) -> np.ndarray:
    """Reconstruction error of every window in a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    for batch in loader:
        features = batch.to(device)
        scores.append(model.reconstruction_error(features).cpu().numpy())
    if not scores:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(scores).astype(np.float64)


def fit(
    config: TrainConfig,
    *,
    windowed: WindowedData | None = None,
) -> tuple[TrainedModel, dict[str, Any]]:
    """Train one detector and evaluate it; returns the model and run metrics."""
    seed_everything(config.seed)
    device = resolve_device(config.train.device)

    if windowed is None:
        windowed = build_datasets(
            config.data.clean_dir,
            config.data.anomalous_dir,
            window=config.data.window,
            stride=config.data.stride,
            mode=config.input.mode,
            validation_fraction=config.data.validation_fraction,
            seed=config.seed,
        )

    model = build_model(
        config.model.arch,
        windowed.input_dim,
        hidden=config.model.hidden,
        latent_dim=config.model.latent_dim,
    ).to(device)
    logger.info(
        "%s on %s: %d parameters, input dim %d, device %s",
        config.model.arch,
        config.input.mode,
        parameter_count(model),
        windowed.input_dim,
        device,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        windowed.train, batch_size=config.train.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        windowed.validation, batch_size=config.train.batch_size, shuffle=False
    )

    criterion: nn.Module = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            features = batch.to(device)
            reconstruction, _ = model(features)
            loss = criterion(reconstruction, features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.detach()) * features.shape[0]
            seen += features.shape[0]

        train_loss = running / max(seen, 1)
        validation_loss = _mean_loss(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss})
        logger.info(
            "epoch %d/%d  train_loss %.6f  val_loss %.6f",
            epoch,
            config.train.epochs,
            train_loss,
            validation_loss,
        )

    elapsed = time.perf_counter() - started

    validation_scores = score_dataset(
        model, windowed.validation, config.train.batch_size, device
    )
    if config.eval.threshold_method == "percentile":
        threshold = metrics.threshold_from_percentile(
            validation_scores, config.eval.threshold_percentile
        )
    else:
        threshold = metrics.threshold_from_mean_std(validation_scores, config.eval.threshold_k)
    logger.info("threshold %.6f from %d validation windows", threshold, validation_scores.size)

    if windowed.anomalous_windows:
        anomalous_scores = score_dataset(
            model, windowed.anomalous, config.train.batch_size, device
        )
        evaluation = metrics.evaluate(anomalous_scores, windowed.anomalous_labels, threshold)
    else:
        anomalous_scores = np.empty(0, dtype=np.float64)
        evaluation = {
            "threshold": threshold,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": None,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "windows": 0,
            "anomalous_windows": 0,
            "normal_windows": 0,
            "flagged_windows": 0,
        }
        logger.warning("no anomalous windows available; evaluation metrics are empty")

    run_metrics: dict[str, Any] = {
        "run_name": config.run_name,
        "arch": config.model.arch,
        "input_mode": config.input.mode,
        "device": str(device),
        "parameters": parameter_count(model),
        "sequence_length": windowed.input_dim,
        "window": config.data.window,
        "stride": config.data.stride,
        "seed": config.seed,
        "epochs": config.train.epochs,
        "batch_size": config.train.batch_size,
        "learning_rate": config.train.lr,
        "loss": LOSS_NAME,
        "duration_s": round(elapsed, 3),
        "window_counts": {
            "train": windowed.train_windows,
            "validation": windowed.validation_windows,
            "anomalous": windowed.anomalous_windows,
        },
        "validation_scores": metrics.summarise_scores(validation_scores),
        "anomaly_files": [path.name for path in windowed.anomalous_files],
        "threshold_method": config.eval.threshold_method,
        "threshold_percentile": config.eval.threshold_percentile,
        "evaluation": evaluation,
        "history": history,
    }

    trained = TrainedModel(
        model=model,
        normalizer=windowed.normalizer,
        threshold=threshold,
        window=config.data.window,
        stride=config.data.stride,
        mode=config.input.mode,
        sequence_length=windowed.input_dim,
    )
    return trained, metrics.json_safe(run_metrics)


def checkpoint_config(config: TrainConfig, trained: TrainedModel) -> dict[str, Any]:
    """Everything inference needs, stored inside the checkpoint."""
    return {
        "version": CONFIG_VERSION,
        "run_name": config.run_name,
        "arch": config.model.arch,
        "hidden": config.model.hidden,
        "latent_dim": config.model.latent_dim,
        "input_mode": trained.mode,
        "sequence_length": trained.sequence_length,
        "window": trained.window,
        "stride": trained.stride,
        "threshold": trained.threshold,
        "threshold_method": config.eval.threshold_method,
        "threshold_percentile": config.eval.threshold_percentile,
        "loss": LOSS_NAME,
        "seed": config.seed,
        "normalizer": trained.normalizer.to_dict(),
    }


def train(config: TrainConfig) -> TrainResult:
    """Run the full training pipeline and write the run artifacts."""
    trained, run_metrics = fit(config)

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{config.run_name}.pt"
    save_checkpoint(checkpoint_path, trained.model, checkpoint_config(config, trained))

    metrics_path = output_dir / f"{config.run_name}.metrics.json"
    metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    config_path = output_dir / f"{config.run_name}.config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")

    logger.info(
        "run %s finished: precision %.3f recall %.3f f1 %.3f auc %s",
        config.run_name,
        run_metrics["evaluation"]["precision"],
        run_metrics["evaluation"]["recall"],
        run_metrics["evaluation"]["f1"],
        f"{run_metrics['evaluation']['auc']:.4f}"
        if run_metrics["evaluation"]["auc"] is not None
        else "n/a",
    )
    return TrainResult(
        run_name=config.run_name,
        checkpoint=checkpoint_path,
        metrics_file=metrics_path,
        config_file=config_path,
        metrics=run_metrics,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m detector.train`` entry point."""
    from .cli_train import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

