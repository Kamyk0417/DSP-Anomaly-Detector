"""YAML configuration loading and Pydantic validation.

The schema mirrors section 4.3. Every anomaly type has its own parameter model
and the union is discriminated by ``type``, so a misspelled key, a missing
parameter or an unknown anomaly type is a configuration error instead of a
runtime surprise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .anomalies import ANOMALY_TYPES, AnomalyRequest

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1


class ConfigError(RuntimeError):
    """Raised when the YAML file cannot be read or fails validation."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpikeParams(_StrictModel):
    amplitude_multiplier: float = Field(gt=0.0)
    width_samples: int = Field(ge=1)


class DropoutParams(_StrictModel):
    width_samples: int = Field(ge=1)
    value: float = 0.0


class FreqDriftParams(_StrictModel):
    delta_hz: float
    duration_samples: int = Field(ge=1)


class AmpJumpParams(_StrictModel):
    gain_multiplier: float = Field(gt=0.0)
    duration_samples: int = Field(ge=1)


class NoiseBurstParams(_StrictModel):
    snr_db: float
    duration_samples: int = Field(ge=1)


class ClippingParams(_StrictModel):
    clip_level_v: float = Field(gt=0.0)
    duration_samples: int = Field(ge=1)


class PhaseShiftParams(_StrictModel):
    delta_deg: float


class HarmonicSpec(_StrictModel):
    order: int = Field(ge=2)
    amp_ratio: float = Field(gt=0.0)


class HarmonicDistortionParams(_StrictModel):
    harmonics: list[HarmonicSpec] = Field(min_length=1)


class DcOffsetParams(_StrictModel):
    offset_v: float
    duration_samples: int = Field(ge=1)


class _AnomalyBase(_StrictModel):
    start_sample: int = Field(ge=0)


class SpikeAnomaly(_AnomalyBase):
    type: Literal["spike"]
    params: SpikeParams


class DropoutAnomaly(_AnomalyBase):
    type: Literal["dropout"]
    params: DropoutParams


class FreqDriftAnomaly(_AnomalyBase):
    type: Literal["freq_drift"]
    params: FreqDriftParams


class AmpJumpAnomaly(_AnomalyBase):
    type: Literal["amp_jump"]
    params: AmpJumpParams


class NoiseBurstAnomaly(_AnomalyBase):
    type: Literal["noise_burst"]
    params: NoiseBurstParams


class ClippingAnomaly(_AnomalyBase):
    type: Literal["clipping"]
    params: ClippingParams


class PhaseShiftAnomaly(_AnomalyBase):
    type: Literal["phase_shift"]
    params: PhaseShiftParams


class HarmonicDistortionAnomaly(_AnomalyBase):
    type: Literal["harmonic_distortion"]
    params: HarmonicDistortionParams


class DcOffsetAnomaly(_AnomalyBase):
    type: Literal["dc_offset"]
    params: DcOffsetParams


AnomalySpec = Annotated[
    Union[
        SpikeAnomaly,
        DropoutAnomaly,
        FreqDriftAnomaly,
        AmpJumpAnomaly,
        NoiseBurstAnomaly,
        ClippingAnomaly,
        PhaseShiftAnomaly,
        HarmonicDistortionAnomaly,
        DcOffsetAnomaly,
    ],
    Field(discriminator="type"),
]


class InjectionConfig(_StrictModel):
    """Validated contents of ``configs/anomalies.yaml``."""

    version: Literal[1] = CONFIG_VERSION
    input_dir: Path = Path("data/clean")
    output_dir: Path = Path("data/anomalous")
    seed: int = 42
    anomalies: list[AnomalySpec] = Field(min_length=1)

    def requests(self) -> list[AnomalyRequest]:
        """Convert the validated entries into dispatcher requests."""
        return [
            AnomalyRequest(
                type=anomaly.type,
                start_sample=anomaly.start_sample,
                params=anomaly.params.model_dump(),
            )
            for anomaly in self.anomalies
        ]


def load_config_document(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is not valid YAML: {error}") from error

    if not isinstance(document, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return document


def load_config(path: Path) -> InjectionConfig:
    """Load and validate an injection configuration."""
    path = Path(path)
    document = load_config_document(path)

    try:
        config = InjectionConfig.model_validate(document)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        raise ConfigError(f"{path} is invalid: {details}") from error

    logger.debug(
        "loaded %s with %d anomaly entries", path, len(config.anomalies)
    )
    return config


__all__ = [
    "ANOMALY_TYPES",
    "CONFIG_VERSION",
    "AnomalySpec",
    "ConfigError",
    "InjectionConfig",
    "load_config",
    "load_config_document",
]

