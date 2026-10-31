# Signal Processing Pipeline

Offline monorepo pipeline that generates synthetic oscilloscope signals, injects
labeled anomalies and trains unsupervised anomaly detectors. Files are the only
interchange medium between the three subprojects.

```
oscilloscope (C17 + SDL2) --> anomaly_injection (Python) --> ml_detector (PyTorch)
```

## Layout

| Path | Description |
| --- | --- |
| `oscilloscope/` | Part 1 - C17, SDL2 display, CMake |
| `anomaly_injection/` | Part 2 - YAML-driven anomaly injection |
| `ml_detector/` | Part 3 - reconstruction-based anomaly detection |
| `data/` | Generated artifacts (`clean/`, `anomalous/`, `models/`), gitignored |

## Data interchange

Each signal is a `<name>.bin` file of headerless little-endian `float32` samples
plus a `<name>.json` metadata file (schema version 1). Anomalous signals add a
`<name>.label.json`. The authoritative schema lives in
`.github/copilot-instructions.md`.

## Anomaly injection

`anomaly_injection` reads clean signals, applies the anomalies described by a YAML
file and writes new `.bin`/`.json`/`.label.json` outputs; inputs are never modified.
Relative directories in the YAML resolve against the current working directory.

```powershell
cd anomaly_injection
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# from the repository root
$env:PYTHONPATH = "anomaly_injection/src"
python -m injector --config anomaly_injection/configs/anomalies.yaml
python -m injector --config anomaly_injection/configs/anomalies.yaml `
                   --input-dir data/clean --output-dir data/anomalous
```

Supported anomaly types: `spike`, `dropout`, `freq_drift`, `amp_jump`,
`noise_burst`, `clipping`, `phase_shift`, `harmonic_distortion`, `dc_offset`.
Every anomaly is validated against a per-type parameter schema, so unknown keys
and missing parameters are reported as configuration errors.

## Anomaly detection

`ml_detector` windows clean signals for unsupervised training, scores windows by
reconstruction error, and reports detections for anomalous files.

```powershell
# from the repository root
$env:PYTHONPATH = "ml_detector/src"
python -m detector.train  --config ml_detector/configs/train.yaml
python -m detector.detect --model data/models/autoencoder_raw.pt `
                          --file data/anomalous/sig_0001.bin `
                          --json data/anomalous/sig_0001.json --write
```

Training writes `<run>.pt` (weights plus the config inference needs, including
windowing, input mode, normalisation statistics and threshold),
`<run>.metrics.json` and `<run>.config.yaml` into `output.dir`. `detect` prints
its report as JSON to stdout and, with `--write`, saves
`<name>.detections.json` next to the input.

Input modes are `raw`, `fft` and `both`; architectures are `autoencoder`,
`lstm_ae` and `transformer_ae`. In every architecture `latent_dim` is the width
of the bottleneck code and `forward` also returns a pooled `(batch, latent_dim)`
embedding.

## Status

Part 1 has the signal, ADC, generator, FFT, filter and visualization modules;
`features` and `io` are still stubs. Part 2 (`anomaly_injection`) is
implemented. Part 3 (`ml_detector`) is implemented for the documented
architectures and input modes.


