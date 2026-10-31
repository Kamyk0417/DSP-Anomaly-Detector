# Copilot Instructions — Digital Oscilloscope & Anomaly Detection Pipeline

## 1. Project Overview

Three-part monorepo pipeline that generates synthetic oscilloscope signals,
injects labeled anomalies, and trains unsupervised anomaly detectors.

- **Platform:** Windows primary (code is cross-platform, Windows is the dev target).
- **Pipeline mode:** Offline. Files are the interchange medium.
- **Repo layout:** Monorepo with three independent subprojects.
- **Language of all code, comments, docstrings, commits, docs:** English.

### 1.1 Pipeline

```
[C: oscilloscope] --clean.bin + clean.json--> [Python: anomaly_injection]
                                                    |
                          augmented.bin + augmented.json + label.json
                                                    |
                                                    v
                                          [Python: ml_detector]
```

### 1.2 Directory Layout

```
/
├── .github/copilot-instructions.md
├── .clang-format
├── .gitignore
├── README.md
├── CMakeLists.txt
├── oscilloscope/          # Part 1 — C17 + SDL2 + CMake
├── anomaly_injection/     # Part 2 — Python + NumPy + PyYAML
├── ml_detector/           # Part 3 — Python + PyTorch
└── data/                  # generated artifacts (gitignored)
    ├── clean/
    ├── anomalous/
    └── models/
```

---

## 2. Data Interchange Format (all parts)

Signals are exchanged as `<name>.bin` (samples) + `<name>.json` (metadata).
Anomalous files additionally carry `<name>.label.json`. Clean files must NOT
have a label file.

### 2.1 `<name>.bin`
- Raw, headerless, little-endian `float32` samples.
- Mono, one channel.
- Sample count must equal `metadata.samples`.

### 2.2 `<name>.json` (schema v1)
```json
{
  "version": 1,
  "samples": 100000,
  "sample_rate_hz": 100000.0,
  "duration_s": 1.0,
  "dtype": "float32",
  "endianness": "little",
  "channels": 1,
  "generator": {
    "type": "sine|square|triangle|sawtooth|noise|multi_tone",
    "amplitude_v": 2.5,
    "frequency_hz": 1000.0,
    "phase_deg": 0.0,
    "offset_v": 0.0,
    "duty_cycle": 0.5,
    "snr_db": 30.0,
    "components": [
      {"freq_hz": 500.0, "amp_v": 1.0, "phase_deg": 0.0}
    ]
  },
  "adc": {"bits": 12, "v_min": -5.0, "v_max": 5.0},
  "filters_applied": [
    {"type": "chebyshev1|chebyshev2|butterworth|fir_hamming|fir_hann|fir_blackman", "params": {}}
  ],
  "features": {
    "rms": 0.0,
    "peak_to_peak": 0.0,
    "variance": 0.0,
    "skewness": 0.0,
    "kurtosis": 0.0,
    "dominant_freq_hz": 0.0,
    "band_energies": [],
    "fft_magnitude": []
  }
}
```

### 2.3 `<name>.label.json` (only for anomalous files)
```json
{
  "version": 1,
  "source_clean_file": "clean/sig_0001",
  "anomalies": [
    {
      "type": "spike|dropout|freq_drift|amp_jump|noise_burst|clipping|phase_shift|harmonic_distortion|dc_offset",
      "start_sample": 12000,
      "end_sample": 12050,
      "params": {"amplitude_multiplier": 3.0}
    }
  ]
}
```

**Rules:**
- Never change the schema without bumping `version`.
- Never silently change field names or semantics.
- All numeric fields in SI units (volts, hertz, seconds).

---

## 3. Part 1 — C Oscilloscope (`oscilloscope/`)

**Stack:** C17, SDL2 (display only), CMake >= 3.20, own FFT implementation.

### 3.1 Responsibilities
- Generate signals: sine, square, triangle, sawtooth, white noise, multi-tone.
- Simulate a 12-bit ADC with range -5 V..+5 V and configurable sample rate
  (1 kSa/s - 1 MSa/s, default 100 kSa/s).
- Quantize to ADC codes and dequantize back to volts.
- Own FFT: iterative radix-2 Cooley-Tukey, in-place.
- FIR filters: windowed-sinc (Hamming, Hann, Blackman), runtime design.
- IIR filters: Chebyshev I (default), Chebyshev II, Butterworth, biquad
  cascade (SOS).
- Extract features: RMS, peak-to-peak, variance, skewness, kurtosis,
  dominant frequency, band energies, FFT magnitude coefficients.
- Real-time SDL2 visualization: time-domain trace, amplitude spectrum,
  spectrogram (waterfall), parameter panel. Keyboard-driven; mouse optional.
- Export/import via `io.c` using the interchange format (section 2).

### 3.2 Module conventions (C)

**Public headers** in `include/<module>.h`, **implementation** in `src/<module>.c`.

**Naming:**
- Types: `PascalCase` — `SignalBuffer`, `AdcConfig`, `FilterCoeffs`.
- Functions: `snake_case`, prefixed by module — `fft_forward`,
  `filter_apply_iir_sos`, `adc_quantize`.
- Variables/parameters: `snake_case`.
- Macros/constants: `UPPER_SNAKE_CASE` — `ADC_BITS_DEFAULT`.
- Enums: type `PascalCase`, values `UPPER_SNAKE` —
  `typedef enum { GENERATOR_SINE, GENERATOR_SQUARE } GeneratorType;`.

**Structs:**
- Plain data structs: `typedef struct { ... } Name;`.
- Modules owning resources expose `*_create` / `*_destroy` pairs.
- Never return owning pointers from arbitrary functions.

**Error handling:**
- Functions return `int` (0 = success, negative = error) or `bool`.
- Results go through out-parameters.
- Never call `exit()` / `abort()` from library code; only `main.c` may exit.

**Memory:**
- Caller owns buffers; library functions never allocate on the caller's behalf
  unless the function name clearly says so (`*_create` returns owned object).
- Use `malloc` / `free` from `<stdlib.h>`. No custom allocators.

**Numerics:**
- `float` for signal samples, `double` for accumulators and twiddles.
- No implicit `float`/`double` mixing; use explicit casts.

### 3.3 ADC specification
- Bits: 12 (configurable 8/10/12/16).
- Range: -5.0 V .. +5.0 V (configurable).
- Sample rate: default 100 kSa/s; allowed 1 kSa/s - 1 MSa/s.
- Quantize: `code = round((v - v_min) / (v_max - v_min) * (2^bits - 1))`,
  clipped to [0, 2^bits - 1]. Expose `adc_quantize` and `adc_dequantize`.

### 3.4 FFT specification
- Own implementation: radix-2 Cooley-Tukey, iterative, in-place.
- `n` must be power of 2; pad with zeros otherwise.
- Real input -> complex spectrum; also a real-FFT wrapper returning only the
  positive half.
- Twiddles: `cos`/`sin` computed in `fft_plan_create` and cached in the plan.
- Windowing helpers (Hamming, Hann, Blackman) live in `fft.c` or `window.c`.

### 3.5 Filters specification
- FIR: windowed-sinc, order default 64, window selectable
  (Hamming / Hann / Blackman).
- IIR: Chebyshev I (default), Chebyshev II, Butterworth.
  - Biquad cascade (SOS).
  - Designed from cutoff, order, ripple.
  - `filter_apply_iir_sos` processes a whole buffer through the cascade.
- Filters do not allocate; caller provides output buffers.

### 3.6 Visualization (SDL2)
- SDL2 is used **only** for display, never for signal processing.
- Views: time-domain trace, amplitude spectrum, spectrogram, parameter panel.
- `visualize.c` must not depend on `main.c`; it may depend on `signal.h` and
  `io.h`.

### 3.7 Module dependency order (no cycles)
```
signal.h  (types)
  ^
  +-- generator
  +-- adc
  +-- fft
  +-- filter
  +-- features   (depends on fft)
  +-- io
        ^
        +-- visualize (signal + io)
              ^
              +-- main.c
```

### 3.8 Build (CMake)
- Top-level `CMakeLists.txt` calls `add_subdirectory(oscilloscope)`.
- `oscilloscope/CMakeLists.txt`: `find_package(SDL2 REQUIRED)`.
  If SDL2 is not found on Windows, fall back to `FetchContent` and document
  the fallback in the target.
- Target: `oscilloscope` executable; link `SDL2::SDL2` (+ `SDL2::SDL2main`
  on Windows).
- `set(CMAKE_C_STANDARD 17)` and `set(CMAKE_C_STANDARD_REQUIRED ON)`.
- Warnings: `-Wall -Wextra -Wpedantic -Werror` for GCC/Clang;
  `/W4 /WX` for MSVC (use generator expressions or `if(MSVC)`).
- Copy `configs/default.json` next to the executable at build time.

---

## 4. Part 2 — Python Anomaly Injection (`anomaly_injection/`)

**Stack:** Python 3.11+, NumPy, PyYAML, Pydantic. `venv` + `requirements.txt`.

### 4.1 Responsibilities
- Read clean `<name>.bin` + `<name>.json`.
- Apply anomalies to the sample buffer according to a YAML config.
- Write `<name>.bin` + `<name>.json` + `<name>.label.json`.
- Never modify input files; always produce new outputs.

### 4.2 Anomaly types (one Python function each)

| Type key | Effect | Params |
|---|---|---|
| `spike` | Single large-amplitude impulse | `amplitude_multiplier`, `width_samples` |
| `dropout` | Short zero/constant segment | `width_samples`, `value` (default 0) |
| `freq_drift` | Slow frequency modulation | `delta_hz`, `duration_samples` |
| `amp_jump` | Step change in gain | `gain_multiplier`, `duration_samples` |
| `noise_burst` | Local SNR drop | `snr_db`, `duration_samples` |
| `clipping` | Hard clip to ADC rails | `clip_level_v`, `duration_samples` |
| `phase_shift` | Sudden phase discontinuity | `delta_deg` |
| `harmonic_distortion` | Add harmonics | `harmonics: [{order, amp_ratio}]` |
| `dc_offset` | Add DC offset | `offset_v`, `duration_samples` |

Signature: `def apply_<type>(x: np.ndarray, start: int, params: dict) -> tuple[np.ndarray, dict]`
returning the modified signal and the label record.

### 4.3 YAML config schema (`configs/anomalies.yaml`)
```yaml
version: 1
input_dir: data/clean
output_dir: data/anomalous
seed: 42
anomalies:
  - type: spike
    start_sample: 12000
    params:
      amplitude_multiplier: 3.0
      width_samples: 5
  - type: noise_burst
    start_sample: 40000
    params:
      snr_db: -5
      duration_samples: 2000
```

### 4.4 CLI
```
python -m injector --config configs/anomalies.yaml
python -m injector --config configs/anomalies.yaml --input-dir data/clean --output-dir data/anomalous
```

### 4.5 Package layout
```
anomaly_injection/
├── requirements.txt
├── configs/anomalies.yaml
└── src/injector/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── config.py        # YAML load + Pydantic validation
    ├── io.py            # .bin/.json read/write (spec section 2)
    ├── anomalies.py     # one function per anomaly type
    └── labels.py        # label.json writer
```

### 4.6 Python conventions
- Type hints on all public functions (PEP 604: `list[str]`, `X | None`).
- Dataclasses or Pydantic for config objects.
- `pathlib.Path` instead of `os.path`.
- Vectorized NumPy; do not loop over samples in Python where a NumPy
  expression exists.
- No global state; pass a config object.
- Logging via `logging`; no `print` in library code.

---

## 5. Part 3 — PyTorch Anomaly Detection (`ml_detector/`)

**Stack:** Python 3.11+, PyTorch 2.x, NumPy, PyYAML, scikit-learn (metrics).
`venv` + `requirements.txt`.

### 5.1 Responsibilities
- Build a dataset from `data/anomalous/` (paired `.bin` + `.json` +
  `.label.json`) and `data/clean/` (for unsupervised training).
- Train unsupervised anomaly detectors on normal windows.
- Evaluate on anomalous windows using labels.
- Save `.pt` checkpoints + `metrics.json` + `config.yaml`.

### 5.2 Input modes (configurable)
- `raw` — windowed raw signal (default window 1024, stride 512).
- `fft` — real-FFT magnitude of each window (positive half).
- `both` — concatenated `[raw, fft]`.

### 5.3 Architectures (selectable)
All models implement `forward(x) -> (x_hat, latent)` and expose
`reconstruction_error(x) -> Tensor[float]`.
- `autoencoder` — 1D conv encoder/decoder (default, baseline).
- `lstm_ae` — LSTM autoencoder.
- `transformer_ae` — Transformer encoder + reconstruction head.

### 5.4 Training pipeline
1. Split clean windows into train/val (~80/20).
2. Train on reconstruction loss (MSE, default).
3. Score = reconstruction error per window.
4. Threshold chosen on val (percentile or mean + k*std).
5. Evaluate on anomalous set: precision, recall, F1, AUC, confusion matrix.
6. Save `data/models/<run>.pt` + `metrics.json` + `config.yaml`.

### 5.5 CLI
```
python -m detector.train --config configs/train.yaml
python -m detector.detect --model data/models/ae.pt --file data/anomalous/x.bin --json data/anomalous/x.json
```
`detect` prints JSON to stdout and optionally writes `<name>.detections.json`
next to the input.

### 5.6 Package layout
```
ml_detector/
├── requirements.txt
├── configs/train.yaml
└── src/detector/
    ├── __init__.py
    ├── cli_train.py
    ├── cli_detect.py
    ├── data.py         # dataset + windowing
    ├── models.py       # AE, LSTM-AE, Transformer-AE
    ├── train.py
    ├── detect.py
    └── metrics.py
```

### 5.7 Config schema (`configs/train.yaml`)
```yaml
version: 1
seed: 42
data:
  clean_dir: data/clean
  anomalous_dir: data/anomalous
  window: 1024
  stride: 512
model:
  arch: autoencoder      # autoencoder | lstm_ae | transformer_ae
  latent_dim: 32
  hidden: 128
input:
  mode: raw              # raw | fft | both
train:
  batch_size: 64
  epochs: 50
  lr: 1.0e-3
  device: cuda           # cuda | cpu
eval:
  threshold_percentile: 99.0
output:
  dir: data/models
```

### 5.8 Python conventions
Same as section 4.6. Additionally:
- No `print` for training progress; use `logging`.
- Device selection through config, not hardcoded.
- Save `torch.save({"model": state_dict, "config": cfg}, path)` so that
  inference is self-contained.

---

## 6. Coding Standards Summary

### 6.1 C
- Language: C17 (`CMAKE_C_STANDARD 17`, `CMAKE_C_STANDARD_REQUIRED ON`).
- Warnings: GCC/Clang `-Wall -Wextra -Wpedantic -Werror`;
  MSVC `/W4 /WX`.
- Formatter: `clang-format` with repo `.clang-format`
  (LLVM base, 4-space indent, 100 columns, `int *p`, braces on same line).
- Headers self-contained (include their own dependencies).
- No hidden globals; pass state through structs.
- No C++ files inside `oscilloscope/`.

### 6.2 Python
- Version 3.11+.
- Formatter + linter: `ruff` (config in `pyproject.toml`).
- Type hints on all public functions.
- Per-subproject `requirements.txt`; install into a local `venv`.
- No plotting in library code; logging only.

---

## 7. How Copilot Should Behave in This Repo

### 7.1 Proactive
- Suggest improvements as reviewable diffs.
- Implement features end-to-end (header + impl + CMake update + config
  default + doc), not just a snippet.
- If this document conflicts with the code, **flag it and ask before
  changing behavior**.

### 7.2 Avoid
- Do NOT add third-party dependencies without asking. Specifically:
  - C: no FFTW, no kissfft (FFT is ours), no GSL.
  - Python: no SciPy unless explicitly requested (NumPy + PyTorch suffice).
- Do NOT silently change the interchange format (section 2); bump `version`.
- Do NOT refactor across module boundaries without asking.
- Do NOT add tests — user has explicitly deferred them for now.
- Do NOT add CI/CD — user has explicitly deferred it.
- Do NOT use `print` in library code.
- Do NOT create C++ files.
- Do NOT add plotting libraries.

---

## 8. Style Reference Snippets

### 8.1 C header pattern
```c
/* include/fft.h */
#ifndef FFT_H
#define FFT_H

#include <stddef.h>

typedef struct {
    size_t n;
    double *twiddles;
} FftPlan;

int  fft_plan_create(FftPlan *plan, size_t n);
void fft_plan_destroy(FftPlan *plan);
int  fft_forward(const FftPlan *plan,
                 const float *in,
                 float *out_re,
                 float *out_im);

#endif
```

### 8.2 C implementation pattern
```c
/* src/fft.c */
#include "fft.h"

#include <math.h>
#include <stdlib.h>

int fft_plan_create(FftPlan *plan, size_t n) {
    if (plan == NULL || n == 0 || (n & (n - 1)) != 0) {
        return -1;
    }
    plan->n = n;
    plan->twiddles = malloc(n * sizeof(double));
    if (plan->twiddles == NULL) {
        return -2;
    }
    for (size_t k = 0; k < n; ++k) {
        plan->twiddles[k] = cos(2.0 * M_PI * (double)k / (double)n);
    }
    return 0;
}
```

### 8.3 Python I/O pattern
```python
# anomaly_injection/src/injector/io.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SignalFile:
    samples: np.ndarray
    metadata: dict


def read_signal(base: Path) -> SignalFile:
    samples = np.fromfile(base.with_suffix(".bin"), dtype="<f4")
    metadata = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    if samples.size != metadata["samples"]:
        raise ValueError(
            f"size mismatch: {samples.size} vs {metadata['samples']}"
        )
    return SignalFile(samples=samples, metadata=metadata)


def write_signal(base: Path, sig: SignalFile) -> None:
    base.with_suffix(".bin").write_bytes(sig.samples.astype("<f4").tobytes())
    base.with_suffix(".json").write_text(
        json.dumps(sig.metadata, indent=2), encoding="utf-8"
    )
```

---

## 9. Quick Decisions Reference

| Decision | Choice |
|---|---|
| Repo layout | Monorepo, three subprojects |
| Pipeline mode | Offline, files as interchange |
| OS primary | Windows |
| Interchange format | `.bin` float32 LE + `.json` + `.label.json` |
| ADC | 12-bit, -5..+5 V, 1 kSa/s - 1 MSa/s, default 100 kSa/s |
| FFT | Own radix-2 Cooley-Tukey, in-place |
| IIR default | Chebyshev I (SOS biquads) |
| FIR default | Windowed-sinc, Hamming, order 64 |
| Anomaly injection language | Python (YAML-driven) |
| ML approach | Unsupervised (reconstruction error) |
| ML primary architecture | 1D Conv Autoencoder |
| C standard | C17 |
| C warnings | `-Wall -Wextra -Wpedantic -Werror` / MSVC `/W4 /WX` |
| Python version | 3.11+ |
| Tests | Deferred |
| CI/CD | Deferred |
| Copilot mode | Proactive, asks before adding deps |
| Language | English |