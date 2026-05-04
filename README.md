# smolcde

A small library for conditional density estimation using masked
autoregressive flows (MAF) and mixture density networks (MDN), with
Python & C implementations and a small CLI.

## Quick start

### CLI (no Python needed)

Download the prebuilt binary for your platform from the
[GitHub Releases](https://github.com/maedoc/smolcde/releases) page.

```bash
# Train a conditional density model
./smolcde-linux-x86_64 train \
  --features features.csv \
  --params params.csv \
  --out model.maf \
  --epochs 200 --hidden 32 --blocks 5

# Draw samples conditioned on new features
./smolcde-linux-x86_64 infer \
  --model model.maf \
  --features test_features.csv \
  --out predictions.csv \
  --mode sample --samples 64
```

### Python library

```bash
pip install .
```

```python
import cde
import numpy as np

# Conditional density p(params | features)
C = cde.MAFEstimator(param_dim=3, feature_dim=10, hidden_size=32, n_flows=5)
C.train(params_data, features_data, n_epochs=200)

# Sample from the posterior for new features
samples = C.sample(features_new, n_samples=1000)
mean_est = np.mean(samples, axis=1)
```

## Setup

### Option 1: pip install (Python only)

```bash
pip install .          # or: pip install -e ".[dev]"
pytest                 # 17 tests, including C gradient checks
```

### Option 2: Build from source (C + Python)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cp build/libsmolmaf.so .   # or .dylib / .dll
cp build/smolcde .         # CLI binary
pytest
```

### Option 3: Download release binary

See [Releases](https://github.com/maedoc/smolcde/releases). Each
release bundles:
- `smolcde-*` — CLI binary for your platform
- `libsmolmaf-*.{so,dylib,dll}` — shared library for Python ctypes
- `cde.py` — drop-in Python module
- `README.md`

## Demos

```bash
bash demos/demo_banana.sh    # Synthetic banana-shaped density
bash demos/demo_mnist.sh      # MNIST 8×8 classification (one-hot MAF)
bash demos/demo_lorenz.sh     # Lorenz attractor regression
python demos/bench_c_vs_py.py # Speed comparison
```

## What's inside

| Component | Language | Description |
|-----------|----------|-------------|
| `cde.py` | Python | `MAFEstimator`, `MDNEstimator`, data generators |
| `maf.c` / `maf.h` | C | Fast MAF forward, backward, sampling, log-prob |
| `main.c` | C | CLI: `train` and `infer` commands |

**Key design choice**: the CLI binary statically links `maf.c` for
portability, while Python loads `libsmolmaf` via ctypes for speed.

## Performance

C training is ~24× faster than Python autograd (2000 samples, 100
eepochs: C 254 ms vs Python 6.16 s).

## Known limitations

- **D=1 MAF is degenerate** — the autoregressive structure is vacuous
  for scalar targets. For classification, use one-hot encoding (D≥2).
  For scalar regression, use MDN. A `RuntimeWarning` is emitted when
  `param_dim=1`.

## Tests

```bash
python -m pytest test_cde.py -v
```

17 tests covering gradient correctness, deterministic sampling, model
roundtrip, all inference modes, CLI coverage, resume training, and
MNIST pipeline.

## License

MIT
