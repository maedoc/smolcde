# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-04

### Added
- Cross-platform CI builds (linux-x86_64, linux-aarch64, macos-x86_64, macos-arm64, windows-x86_64)
- Auto-release GitHub workflow on `v*` tags
- Prebuilt binaries with shared libraries for all platforms
- `pyproject.toml` for `pip install`
- `Makefile` with `build`, `test`, `clean`, `install`, `dev`, `lint` targets
- `LICENSE` (MIT)
- `__version__` in `cde.py` and `--version` CLI flag
- `__all__` for clean Python public API
- `.editorconfig` for consistent formatting
- 8 new tests: gradient comparison, deterministic sampling, roundtrip, all infer modes, skip-header, resume training, C log_prob, C=0 backward/sampling
- 3 demo scripts: banana, MNIST, Lorenz
- Benchmark infrastructure: MNIST and Lorenz grid search
- `MNIST_FAILURE_ANALYSIS.md` documenting D=1 collapse

### Changed
- README rewritten with quick start, setup options, demos, performance numbers
- Notebook outputs cleared (stale NSF references)
- pyproject.toml: classifiers, license field, ruff dev dependency
- CI: ruff lint check on every push

### Fixed
- D=1 MADE mask generation (was `m_h=0`, now `m_h=1` with M2 override)
- C library workspace allocation for `feature_dim=0`
- C CLI: replaced `exit(1)` with return codes throughout
- C CLI: fixed "moel" typo → "model"
- C CLI: fixed global buffer memory leak (10 allocations)
- MNIST test: evaluates on held-out data, not training data
- pyproject.toml: `scikit_learn` → `scikit-learn`

### Removed
- ~314 lines of unfinished NSF code (`NSFEstimator`, `rational_quadratic_spline`)
- Stale `benchmarks/thorough_mnist_onehot.py`

## [0.1.0] - 2024

### Added
- Initial release: MAF and MDN conditional density estimation
- Python implementation with autograd
- C implementation with SIMD/AVX optimization
- CLI for training and inference

[0.2.0]: https://github.com/maedoc/smolcde/releases/tag/v0.2.0
[0.1.0]: https://github.com/maedoc/smolcde/releases/tag/v0.1
