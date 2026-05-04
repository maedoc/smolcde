## 🚀 Highlights

- **Dropped ~300 lines of unfinished code** — rational quadratic spline / NSF support removed
- **C CLI hardened** — deterministic `--seed`, clean error handling (no `exit(1)` from library paths), fixed memory leaks, typo fix
- **D=1 MAF posterior collapse diagnosed and fixed** — MADE masks now correct for `param_dim=1`; documented limitation and one-hot encoding workaround
- **17-test suite** — 7 new tests including gradient comparison against autograd, deterministic sampling, model roundtrip, all inference modes, and CLI coverage
- **4 demos** — banana toy problem, MNIST classification, Lorenz attractor regression, C-vs-Python speed comparison
- **Prebuilt binaries** — 5-platform CI artifacts + shared libraries, auto-release on push of `v*` tag
- **MNIST one-hot encoding** — MAF achieves 84–90% accuracy with D=10 one-hot (vs 57% MDN scalar)

## What's Changed

### Removals
- Drop unfinished `NSFEstimator` class and `rational_quadratic_spline` function (~314 lines)

### Fixes
- C MADE mask generation for `param_dim=1` (was `m_h=0`, now `m_h=1`; M2 overridden for D=1)
- C library workspace allocation for `feature_dim=0` (avoids `malloc(0)` returning NULL)
- C CLI: `exit(1)` replaced with clean NULL returns in `load_model_file`; CLI commands return error codes
- C CLI: "moel" typo → "model"
- C CLI: all global buffers freed after `maf_load_model` copies them (was leaking 10 allocations)
- CI: `scikit_learn` typo → `scikit-learn` (pip may silently install wrong package)
- MNIST test: now evaluates on held-out test data, not training data (no overfitting free pass)

### New Tests
- `test_c_gradient_matches_autograd` — C backward pass vs Python autograd for all 6 weight tensors
- `test_c_deterministic_sampling` — bit-identical results for same noise seed
- `test_model_serialization_roundtrip` — train → save → load → infer must match
- `test_infer_all_modes` — sample, stats, and quantiles modes
- `test_skip_header` — `--skip-header` flag for CSV with header row
- `test_resume_training` — `--load` checkpoint to continue training
- `test_c_backward_feature_dim_zero` — backward pass with no conditional features
- `test_cli_bad_model_file` — error handling on nonexistent/corrupt `.maf` file
- `test_c_log_prob` — log-prob consistency between forward train and standalone call
- `test_mnist_cli_workflow` — full MNIST pipeline with D=10 one-hot encoding

### Demos & Benchmarks
- `bash demos/demo_banana.sh` — synthetic conditional density (banana-shaped distribution)
- `bash demos/demo_mnist.sh` — MNIST 8×8 classification with one-hot MAF (D=10)
- `bash demos/demo_lorenz.sh` — Lorenz attractor regression with MAF (D=3)
- `python demos/bench_c_vs_py.py` — speed comparison: C training is ~24× faster than Python autograd
- `bash benchmarks/bench_mnist` — MNIST grid search (MDN vs MAF, 5–15 components)
- `bash benchmarks/bench_lorenz` — Lorenz hyperparameter sweep
- `benchmarks/MNIST_FAILURE_ANALYSIS.md` — full writeup of D=1 posterior collapse

### Infrastructure
- CMake builds CLI statically (no shared-lib dependency for the binary)
- CI matrix: linux-x86_64, linux-aarch64, macos-x86_64, macos-arm64, windows-x86_64
- Shared libraries bundled in artifacts for Python users (`libsmolmaf.{so,dylib}`, `smolmaf.dll`)
- Push `v*` tag → auto-creates GitHub Release with all binaries + `cde.py` + README
- `.gitignore` excludes model files, temp CSVs, experiment output

### Key Performance Numbers
| Task | Result |
|------|--------|
| C vs Py training (2000 samples, 100 epochs) | C: 254ms — Python: 6.16s (**24× faster**) |
| MNIST 8×8, 500 train, MAF D=10 one-hot | **84% accuracy** |
| MNIST 8×8, 1500 train, MAF D=10 one-hot | **90% accuracy** |
| Lorenz D=3, best config (1000ep, 64h, 8f) | MAE ~0.94, correlation ~0.916 |

### Known Limitations
- **D=1 MAF is degenerate** — the auto-regressive structure is vacuous for scalar targets. MADE masks become all-ones, the model learns input-independent parameters, and the posterior collapses to a delta. A `RuntimeWarning` is emitted. For classification, use one-hot encoding (D≥2). For scalar regression, use MDN.
