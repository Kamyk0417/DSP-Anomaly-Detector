"""One ``apply_<type>`` function per anomaly type.

Every applier follows the section 4.2 contract::

    def apply_<type>(x: np.ndarray, start: int, params: dict) -> tuple[np.ndarray, dict]

It returns the modified signal and the label record for that anomaly. The input
array is never mutated; the returned array is a new buffer with the dtype of the
input.

Besides the documented parameters, :func:`apply_anomalies` injects two reserved
context keys into ``params``:

``sample_rate_hz``
    Sample rate of the record, needed to convert hertz and degrees to samples.
``rng``
    A :class:`numpy.random.Generator` shared by the whole injection run, so the
    result is reproducible for a given seed.

Appliers build their label records from their own documented parameters, so the
context keys never leak into ``.label.json``.

``end_sample`` is exclusive, like a Python slice bound. Anomalies that have no
duration parameter affect the record from ``start`` to the end of the signal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ANOMALY_TYPES = (
    "spike",
    "dropout",
    "freq_drift",
    "amp_jump",
    "noise_burst",
    "clipping",
    "phase_shift",
    "harmonic_distortion",
    "dc_offset",
)


@dataclass(frozen=True)
class AnomalyRequest:
    """One anomaly to apply, decoupled from the YAML/Pydantic configuration."""

    type: str
    start_sample: int
    params: dict[str, Any] = field(default_factory=dict)


def _working_copy(samples: np.ndarray) -> np.ndarray:
    return np.array(samples, dtype=np.float64, copy=True)


def _restore_dtype(work: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(work, dtype=reference.dtype)


def _require_start(samples: np.ndarray, start: int) -> None:
    count = int(samples.size)
    if start < 0 or start >= count:
        raise ValueError(f"start_sample {start} is outside a record of {count} samples")


def _segment_bounds(samples: np.ndarray, start: int, length: int) -> tuple[int, int]:
    _require_start(samples, start)
    if length < 1:
        raise ValueError(f"segment length must be positive, got {length}")
    end = start + length
    count = int(samples.size)
    if end > count:
        raise ValueError(f"segment {start}..{end} exceeds a record of {count} samples")
    return start, end


def _rng(params: dict[str, Any]) -> np.random.Generator:
    generator = params.get("rng")
    if isinstance(generator, np.random.Generator):
        return generator
    return np.random.default_rng(int(params.get("seed", 0)))


def _dominant_frequency(samples: np.ndarray, sample_rate_hz: float) -> float:
    """Estimate the dominant frequency of the record with a windowed FFT."""
    count = int(samples.size)
    if count < 8:
        raise ValueError(f"need at least 8 samples to estimate a frequency, got {count}")

    spectrum = np.abs(np.fft.rfft(np.asarray(samples, dtype=np.float64) * np.hanning(count)))
    spectrum[0] = 0.0
    peak = float(spectrum.max())
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("signal has no dominant frequency (it is silent)")

    index = int(np.argmax(spectrum))
    position = float(index)
    if 0 < index < spectrum.size - 1:
        left = float(spectrum[index - 1])
        centre = float(spectrum[index])
        right = float(spectrum[index + 1])
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            position += 0.5 * (left - right) / denominator
    return position * sample_rate_hz / count


def apply_spike(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add a smooth bump whose peak is ``amplitude_multiplier`` times the record peak."""
    multiplier = float(params["amplitude_multiplier"])
    width = int(params["width_samples"])
    start, end = _segment_bounds(samples, start, width)

    peak = float(np.max(np.abs(samples)))
    if not math.isfinite(peak) or peak <= 0.0:
        peak = 1.0

    positions = np.arange(width, dtype=np.float64)
    shape = 0.5 - 0.5 * np.cos(2.0 * math.pi * (positions + 0.5) / width)
    shape /= shape.max()

    work = _working_copy(samples)
    work[start:end] += multiplier * peak * shape

    label = {
        "type": "spike",
        "start_sample": start,
        "end_sample": end,
        "params": {"amplitude_multiplier": multiplier, "width_samples": width},
    }
    return _restore_dtype(work, samples), label


def apply_dropout(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hold the signal at a constant value for ``width_samples`` samples."""
    width = int(params["width_samples"])
    value = float(params.get("value", 0.0))
    start, end = _segment_bounds(samples, start, width)

    work = _working_copy(samples)
    work[start:end] = value

    label = {
        "type": "dropout",
        "start_sample": start,
        "end_sample": end,
        "params": {"width_samples": width, "value": value},
    }
    return _restore_dtype(work, samples), label


def apply_freq_drift(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drift the signal frequency by ``delta_hz`` over the segment.

    The drift is realised as a progressive time warp, so it works for any
    waveform: the local resampling factor grows linearly from 1 to
    ``1 + delta_hz / dominant_frequency`` across the segment.
    """
    delta_hz = float(params["delta_hz"])
    duration = int(params["duration_samples"])
    sample_rate_hz = float(params["sample_rate_hz"])
    start, end = _segment_bounds(samples, start, duration)

    dominant_hz = _dominant_frequency(samples, sample_rate_hz)
    scale = delta_hz / dominant_hz

    count = end - start
    ramp = np.linspace(0.0, 1.0, count, endpoint=False)
    local_scale = 1.0 + scale * ramp
    offsets = np.concatenate(([0.0], np.cumsum(local_scale)[:-1]))
    source = start + offsets

    work = _working_copy(samples)
    source = np.clip(source, 0.0, float(samples.size - 1))
    positions = np.arange(samples.size)
    work[start:end] = np.interp(source, positions, np.asarray(samples, dtype=np.float64))

    label = {
        "type": "freq_drift",
        "start_sample": start,
        "end_sample": end,
        "params": {"delta_hz": delta_hz, "duration_samples": duration},
    }
    return _restore_dtype(work, samples), label


def apply_amp_jump(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Multiply the segment by ``gain_multiplier``."""
    gain = float(params["gain_multiplier"])
    duration = int(params["duration_samples"])
    start, end = _segment_bounds(samples, start, duration)

    work = _working_copy(samples)
    work[start:end] *= gain

    label = {
        "type": "amp_jump",
        "start_sample": start,
        "end_sample": end,
        "params": {"gain_multiplier": gain, "duration_samples": duration},
    }
    return _restore_dtype(work, samples), label


def apply_noise_burst(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add white gaussian noise so the segment SNR becomes ``snr_db``."""
    snr_db = float(params["snr_db"])
    duration = int(params["duration_samples"])
    start, end = _segment_bounds(samples, start, duration)
    rng = _rng(params)

    work = _working_copy(samples)
    segment = work[start:end]
    signal_power = float(np.mean(np.square(segment)))
    if not math.isfinite(signal_power) or signal_power <= 0.0:
        signal_power = float(np.mean(np.square(work)))
    if not math.isfinite(signal_power) or signal_power <= 0.0:
        signal_power = 1.0

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    segment += rng.normal(0.0, math.sqrt(noise_power), size=end - start)

    label = {
        "type": "noise_burst",
        "start_sample": start,
        "end_sample": end,
        "params": {"snr_db": snr_db, "duration_samples": duration},
    }
    return _restore_dtype(work, samples), label


def apply_clipping(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hard clip the segment to the symmetric ADC rails ``+-clip_level_v``."""
    clip_level = float(params["clip_level_v"])
    duration = int(params["duration_samples"])
    start, end = _segment_bounds(samples, start, duration)

    work = _working_copy(samples)
    work[start:end] = np.clip(work[start:end], -clip_level, clip_level)

    label = {
        "type": "clipping",
        "start_sample": start,
        "end_sample": end,
        "params": {"clip_level_v": clip_level, "duration_samples": duration},
    }
    return _restore_dtype(work, samples), label


def apply_phase_shift(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Jump the phase of the remainder of the record by ``delta_deg``.

    A phase jump of a narrowband signal equals a time shift of
    ``delta_deg / 360 * fs / f_dominant`` samples, which is applied from
    ``start`` to the end of the record.
    """
    delta_deg = float(params["delta_deg"])
    sample_rate_hz = float(params["sample_rate_hz"])
    _require_start(samples, start)

    dominant_hz = _dominant_frequency(samples, sample_rate_hz)
    shift_samples = delta_deg / 360.0 * sample_rate_hz / dominant_hz

    work = _working_copy(samples)
    source = np.arange(start, samples.size, dtype=np.float64) - shift_samples
    source = np.clip(source, 0.0, float(samples.size - 1))
    work[start:] = np.interp(
        source, np.arange(samples.size), np.asarray(samples, dtype=np.float64)
    )

    label = {
        "type": "phase_shift",
        "start_sample": start,
        "end_sample": int(samples.size),
        "params": {"delta_deg": delta_deg},
    }
    return _restore_dtype(work, samples), label


def apply_harmonic_distortion(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add phase-locked harmonics of the fundamental to the rest of the record."""
    harmonics = params["harmonics"]
    sample_rate_hz = float(params["sample_rate_hz"])
    _require_start(samples, start)

    dominant_hz = _dominant_frequency(samples, sample_rate_hz)
    values = np.asarray(samples, dtype=np.float64)
    times = np.arange(samples.size, dtype=np.float64) / sample_rate_hz
    fundamental_phase = 2.0 * math.pi * dominant_hz * times

    previous = slice(start, samples.size)
    in_phase = 2.0 * float(np.mean(values[previous] * np.cos(fundamental_phase[previous])))
    quadrature = 2.0 * float(np.mean(values[previous] * np.sin(fundamental_phase[previous])))
    amplitude = math.hypot(in_phase, quadrature)
    phase0 = math.atan2(-quadrature, in_phase)
    if not math.isfinite(amplitude):
        raise ValueError("cannot estimate the fundamental amplitude")

    work = _working_copy(samples)
    for harmonic in harmonics:
        order = int(harmonic["order"])
        ratio = float(harmonic["amp_ratio"])
        work[previous] += ratio * amplitude * np.cos(
            order * (fundamental_phase[previous] + phase0)
        )

    label = {
        "type": "harmonic_distortion",
        "start_sample": start,
        "end_sample": int(samples.size),
        "params": {
            "harmonics": [
                {"order": int(harmonic["order"]), "amp_ratio": float(harmonic["amp_ratio"])}
                for harmonic in harmonics
            ]
        },
    }
    return _restore_dtype(work, samples), label


def apply_dc_offset(
    samples: np.ndarray, start: int, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add ``offset_v`` to the segment."""
    offset = float(params["offset_v"])
    duration = int(params["duration_samples"])
    start, end = _segment_bounds(samples, start, duration)

    work = _working_copy(samples)
    work[start:end] += offset

    label = {
        "type": "dc_offset",
        "start_sample": start,
        "end_sample": end,
        "params": {"offset_v": offset, "duration_samples": duration},
    }
    return _restore_dtype(work, samples), label


ANOMALY_APPLIERS: dict[
    str, Callable[[np.ndarray, int, dict[str, Any]], tuple[np.ndarray, dict[str, Any]]]
] = {
    "spike": apply_spike,
    "dropout": apply_dropout,
    "freq_drift": apply_freq_drift,
    "amp_jump": apply_amp_jump,
    "noise_burst": apply_noise_burst,
    "clipping": apply_clipping,
    "phase_shift": apply_phase_shift,
    "harmonic_distortion": apply_harmonic_distortion,
    "dc_offset": apply_dc_offset,
}


def apply_anomaly(
    samples: np.ndarray, request: AnomalyRequest, *, sample_rate_hz: float,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a single anomaly request."""
    applier = ANOMALY_APPLIERS.get(request.type)
    if applier is None:
        raise ValueError(
            f"unknown anomaly type {request.type!r}; expected one of {ANOMALY_TYPES}"
        )

    params = dict(request.params)
    params.setdefault("sample_rate_hz", sample_rate_hz)
    if rng is not None:
        params.setdefault("rng", rng)
    return applier(samples, request.start_sample, params)


def apply_anomalies(
    samples: np.ndarray,
    requests: Sequence[AnomalyRequest],
    *,
    sample_rate_hz: float,
    seed: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply every request in order and collect their label records."""
    rng = np.random.default_rng(seed)
    current = np.asarray(samples)
    labels: list[dict[str, Any]] = []

    for request in requests:
        try:
            current, label = apply_anomaly(
                current, request, sample_rate_hz=sample_rate_hz, rng=rng
            )
        except (ValueError, KeyError) as error:
            raise ValueError(f"{request.type} at sample {request.start_sample}: {error}") from error
        labels.append(label)
        logger.debug("applied %s at sample %d", request.type, request.start_sample)

    return current, labels

