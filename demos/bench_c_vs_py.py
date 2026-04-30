#!/usr/bin/env python3
"""bench_c_vs_py.py — Performance comparison between C and Python MAF implementations."""

import time
import sys
import os
import subprocess
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cde import MAFEstimator, generate_test_data
import autograd.numpy as anp


def find_cli():
    for p in [REPO_ROOT / 'smolcde', REPO_ROOT / 'build' / 'smolcde']:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{seconds/60:.1f} min"


def bench_py_training(params, features, n_epochs, batch_size, hidden, n_flows):
    """Time Python MAF training."""
    model = MAFEstimator(param_dim=params.shape[1], feature_dim=features.shape[1],
                         n_flows=n_flows, hidden_units=hidden)
    start = time.time()
    model.train(params, features, n_epochs=n_epochs, batch_size=batch_size,
                learning_rate=0.001, use_tqdm=False, seed=42)
    elapsed = time.time() - start
    return elapsed, model


def bench_py_sampling(model, features, n_samples, n_repeats=5):
    """Time Python MAF sampling."""
    rng = anp.random.RandomState(42)
    start = time.time()
    for _ in range(n_repeats):
        _ = model.sample(features, n_samples, rng)
    elapsed = time.time() - start
    return elapsed / n_repeats


def bench_c(subcmd, args, timeout=300):
    """Run CLI command and time it."""
    cli = find_cli()
    if not cli:
        return None, "CLI not found"
    cmd = [cli] + subcmd.split() + args
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        if result.returncode != 0:
            return None, result.stderr[:200]
        return elapsed, None
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        print("=" * 60)
        print("SmolCDE: C vs Python Performance Comparison")
        print("=" * 60)
        print()

        # Generate synthetic data
        N, D, C = 2000, 3, 5
        print(f"Data: {N} samples, param_dim={D}, feature_dim={C}")

        np.random.seed(42)
        features_np = np.random.randn(N, C).astype(np.float32) * 2
        params_np = np.random.randn(N, D).astype(np.float32)
        # Make params somewhat predictable from features
        params_np += np.sin(features_np[:, :D]) * 3.0

        # Save CSVs for CLI
        feat_file = tmp / "feat.csv"
        param_file = tmp / "param.csv"
        model_file = tmp / "model.maf"
        np.savetxt(feat_file, features_np, delimiter=',', fmt='%.6f')
        np.savetxt(param_file, params_np, delimiter=',', fmt='%.6f')

        print()

        # --- Training ---
        print("--- Training ---")
        n_epochs = 100
        batch_size = 32
        hidden = 32
        n_flows = 3

        print(f"  Config: {n_epochs} epochs, batch={batch_size}, hidden={hidden}, flows={n_flows}")

        # Python
        print("  Python...", end=" ", flush=True)
        py_time, py_model = bench_py_training(
            params_np, features_np, n_epochs, batch_size, hidden, n_flows)
        print(format_time(py_time))

        # C
        print("  C......", end=" ", flush=True)
        args = [
            "--features", str(feat_file), "--params", str(param_file),
            "--out", str(model_file), "--epochs", str(n_epochs),
            "--hidden", str(hidden), "--blocks", str(n_flows),
            "--lr", "0.001", "--batch", str(batch_size), "--seed", "42"
        ]
        c_time, c_err = bench_c("train", args)
        if c_err:
            print(f"SKIP: {c_err}")
        else:
            print(format_time(c_time))

        # --- Sampling ---
        print()
        print("--- Sampling (1000 samples x 5 repeats) ---")

        # Python
        test_feat = features_np[:3].copy()
        n_samples = 1000
        print("  Python...", end=" ", flush=True)
        py_samp_time = bench_py_sampling(py_model, anp.array(test_feat), n_samples)
        print(f"{py_samp_time*1000:.1f} ms/sample")

        # C
        out_file = tmp / "samples.csv"
        print("  C......", end=" ", flush=True)
        args = [
            "--model", str(model_file), "--features", str(feat_file),
            "--out", str(out_file), "--mode", "sample",
            "--samples", str(n_samples)
        ]
        c_samp_time, c_err = bench_c("infer", args)
        if c_err:
            print(f"SKIP: {c_err}")
        else:
            print(f"  {c_samp_time*1000:.1f} ms (3 conditions)")

        # --- Summary ---
        print()
        print("--- Summary ---")
        cli_found = find_cli() is not None
        if cli_found:
            print(f"  Training:  Python {format_time(py_time):>10} | C {format_time(c_time):>10}")
            speedup = py_time / c_time if c_time and c_time > 0 else 0
            print(f"  C/Python speedup:  {speedup:.1f}x")
        else:
            print("  Python training: " + format_time(py_time))
            print("  C: binary not found — build with cmake first")

        print()


if __name__ == '__main__':
    main()
