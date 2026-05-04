#!/usr/bin/env python3
"""diagnose_mnist.py — Compare Python vs C MAF on MNIST and simpler tasks.

Exploratory diagnostic tool that helped isolate the D=1 MAF posterior collapse.
Core findings now documented in benchmarks/MNIST_FAILURE_ANALYSIS.md.

Keep this script as infrastructure for future regression diagnosis; it
is not a production benchmark and may contain exploratory/debug code.
"""

import sys, os, csv, tempfile, subprocess, json
from pathlib import Path
import numpy as np
np.random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cde import MAFEstimator, generate_test_data
import autograd.numpy as anp


def find_cli():
    for p in [REPO_ROOT / 'smolcde', REPO_ROOT / 'build' / 'smolcde']:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def save_csv(data, filename, header=None):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)


def load_csv(filename):
    rows = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. MNIST: Python vs C comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mnist_data(n_train=496, n_test=200):
    """n_train rounded to multiple of 8 for C compatibility."""
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target.astype(float)
    rng = np.random.RandomState(42)
    # Add small noise to labels so the posterior is not a delta
    y_noisy = y + rng.normal(0, 0.1, size=y.shape)
    Xtr = X[:n_train]
    Ytr = y_noisy[:n_train].reshape(-1, 1)
    Xts = X[n_train:n_train+n_test]
    Yts_true = y[n_train:n_train+n_test].reshape(-1, 1)
    Yts_noisy = y_noisy[n_train:n_train+n_test].reshape(-1, 1)
    return Xtr, Ytr, Xts, Yts_true, Yts_noisy


def train_python_maf(Xtr, Ytr, n_flows=5, hidden=32, epochs=300, lr=0.001, seed=42):
    """Train a Python MAF and return the model."""
    D = Ytr.shape[1]
    C = Xtr.shape[1]
    model = MAFEstimator(param_dim=D, feature_dim=C, n_flows=n_flows, hidden_units=hidden)
    model.train(Ytr, Xtr, n_epochs=epochs, batch_size=8, learning_rate=lr, use_tqdm=False, seed=seed)
    return model


def evaluate_python(model, Xts, Yts_true, n_samples=200, seed=99):
    """Evaluate Python model: sample posterior, compute accuracy, MAE, and check for explosion."""
    rng = anp.random.RandomState(seed)
    means = []
    stds = []
    n_valid = 0
    for i in range(len(Xts)):
        x_cond = Xts[i:i+1]
        try:
            samples = model.sample(x_cond, n_samples, rng)
            s = np.array(samples[0])  # (n_samples, D)
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                continue  # skip degenerate
            means.append(np.mean(s, axis=0))
            stds.append(np.std(s, axis=0))
            n_valid += 1
        except Exception:
            continue
    means = np.array(means)   # (N, D)
    stds = np.array(stds)      # (N, D)

    y_true = Yts_true.flatten()[:len(means)]
    y_pred = np.round(means.flatten()).astype(int)
    
    # Handle degenerate case where predictions blew up
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'acc': 0.0, 'mae': float('inf'), 'max_abs_mean': float('inf'),
            'any_nan': True, 'any_inf': True, 'mean_std': float('inf'),
            'n_valid': n_valid, 'n_total': len(Xts),
            'means': means, 'stds': stds,
        }
    if len(y_true) != len(y_pred):
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]
        means = means[:n]
        stds = stds[:n]
    
    acc = np.mean(y_true == y_pred)
    mae = np.mean(np.abs(y_true - means.flatten()))
    
    # Explosion checks
    max_abs_mean = np.max(np.abs(means))
    max_abs_sample = None  # would need to store all samples
    any_nan = np.any(np.isnan(means))
    any_inf = np.any(np.isinf(means))
    mean_std = np.mean(stds)
    
    return {
        'acc': acc, 'mae': mae, 'max_abs_mean': max_abs_mean,
        'any_nan': any_nan, 'any_inf': any_inf,
        'mean_std': float(mean_std),
        'n_valid': n_valid, 'n_total': len(Xts),
        'means': means, 'stds': stds,
    }


def train_c_maf(Xtr, Ytr, n_flows=5, hidden=32, epochs=300, lr=0.001, batch=32, seed=42, tmpdir=None):
    """Train C MAF via CLI and return model path."""
    cli = find_cli()
    if not cli:
        return None
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    feat_file = os.path.join(tmpdir, 'feat_train.csv')
    param_file = os.path.join(tmpdir, 'param_train.csv')
    model_file = os.path.join(tmpdir, 'model.maf')
    save_csv(Xtr, feat_file)
    save_csv(Ytr, param_file)
    result = subprocess.run([
        cli, 'train', '--features', feat_file, '--params', param_file,
        '--out', model_file, '--epochs', str(epochs),
        '--hidden', str(hidden), '--blocks', str(n_flows),
        '--lr', str(lr), '--batch', str(batch), '--seed', str(seed)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"C training failed: {result.stderr}")
        return None
    return model_file


def evaluate_c(model_file, Xts, Yts_true, n_samples=64, tmpdir=None):
    """Evaluate C model via CLI inference."""
    cli = find_cli()
    if not cli:
        return None
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    feat_file = os.path.join(tmpdir, 'feat_test.csv')
    pred_file = os.path.join(tmpdir, 'pred_stats.csv')
    save_csv(Xts, feat_file)
    subprocess.run([
        cli, 'infer', '--model', model_file, '--features', feat_file,
        '--out', pred_file, '--mode', 'stats', '--samples', str(n_samples)
    ], capture_output=True, text=True)
    
    means = []
    stds = []
    mean_keys_found = 0
    with open(pred_file) as f:
        for r in csv.DictReader(f):
            if r['stat'] == 'mean':
                mean_keys_found += 1
                means.append(float(r['p0']))
            elif r['stat'] == 'std':
                stds.append(float(r['p0']))
    means = np.array(means)
    stds_arr = np.array(stds)
    
    if mean_keys_found == 0:
        return {
            'acc': 0.0, 'mae': float('inf'), 'max_abs_mean': float('inf'),
            'any_nan': True, 'any_inf': True, 'mean_std': float('inf'),
            'n_valid': 0, 'n_total': len(Yts_true),
            'means': means, 'stds': stds_arr,
        }
    
    y_true = Yts_true.flatten()[:len(means)]
    y_pred = np.round(means).astype(int)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'acc': 0.0, 'mae': float('inf'), 'max_abs_mean': float('inf'),
            'any_nan': True, 'any_inf': True, 'mean_std': float('inf'),
            'n_valid': 0, 'n_total': len(Yts_true),
            'means': means, 'stds': stds_arr,
        }
    
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    means = means[:n]
    stds_arr = stds_arr[:n]
    
    acc = np.mean(y_true == y_pred)
    mae = np.mean(np.abs(y_true - means))
    
    return {
        'acc': acc, 'mae': mae, 'max_abs_mean': float(np.max(np.abs(means))),
        'any_nan': bool(np.any(np.isnan(means))),
        'any_inf': bool(np.any(np.isinf(means))),
        'mean_std': float(np.mean(stds_arr)),
        'n_valid': len(means), 'n_total': len(Yts_true),
        'means': means, 'stds': stds_arr,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Simpler discrete-target problems
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_binary_classification(n=400, seed=42):
    """2-class problem: features are 2D, label is 0 or 1 with noise."""
    rng = np.random.RandomState(seed)
    # Two Gaussian clusters
    X0 = rng.randn(n//2, 4) + np.array([[-1, -1, 0, 0]])
    X1 = rng.randn(n//2, 4) + np.array([[1, 1, 0, 0]])
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.concatenate([np.zeros(n//2), np.ones(n//2)])
    # Add noise to labels
    y_noisy = y + rng.normal(0, 0.1, size=y.shape)
    return X, y_noisy.reshape(-1, 1), y.reshape(-1, 1)


def make_multiclass_3(n=600, seed=42):
    """3-class problem: features are 4D, label is 0, 1, or 2 with noise."""
    rng = np.random.RandomState(seed)
    n_per = n // 3
    X0 = rng.randn(n_per, 4) + np.array([[-2, 0, 0, 0]])
    X1 = rng.randn(n_per, 4) + np.array([[0, 2, 0, 0]])
    X2 = rng.randn(n_per, 4) + np.array([[2, -2, 0, 0]])
    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per), 2*np.ones(n_per)])
    y_noisy = y + rng.normal(0, 0.1, size=y.shape)
    return X, y_noisy.reshape(-1, 1), y.reshape(-1, 1)


def make_continuous_regression(n=500, seed=42):
    """Smooth continuous problem for comparison. y = sin(x) + noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y.reshape(-1, 1), y.reshape(-1, 1)  # true = noisy here since it's continuous


def run_comparison(name, X, y_noisy, y_true, n_flows=5, hidden=32, epochs=300, lr=0.001,
                   n_flows_c=5, hidden_c=32, epochs_c=300, lr_c=0.001, batch_c=32):
    """Run Python vs C comparison on a given dataset."""
    n = len(X)
    n_train = (n - 100) // 8 * 8  # round to multiple of 8 for C
    n_test = min(100, n - n_train)
    
    Xtr, Ytr = X[:n_train], y_noisy[:n_train]
    Xts = X[n_train:n_train+n_test]
    Yts_true = y_true[n_train:n_train+n_test]
    
    D = Ytr.shape[1]
    C = Xtr.shape[1]
    
    print(f"\n{'='*70}")
    print(f"  {name}  (N={n}, n_train={n_train}, n_test={n_test}, D={D}, C={C})")
    print(f"{'='*70}")
    
    # ── Python ──
    print(f"  Python MAF ({n_flows}f, {hidden}h, {epochs}ep, lr={lr})...", end=" ", flush=True)
    py_model = train_python_maf(Xtr, Ytr, n_flows=n_flows, hidden=hidden, epochs=epochs, lr=lr)
    py_eval = evaluate_python(py_model, Xts, Yts_true)
    print(f"acc={py_eval['acc']:.3f}, MAE={py_eval['mae']:.4f}, max|μ|={py_eval['max_abs_mean']:.2f}, "
          f"mean(σ)={py_eval['mean_std']:.4f}, valid={py_eval['n_valid']}/{py_eval['n_total']}")
    if py_eval['any_nan']:
        print(f"  ⚠️ Python produced NaN means!")
    if py_eval['max_abs_mean'] > 100:
        print(f"  ⚠️ Python means exploding: max|μ| = {py_eval['max_abs_mean']:.1f}")
    
    # Sample a single condition and print distribution summary
    rng = anp.random.RandomState(42)
    sample0 = np.array(py_model.sample(Xts[0:1], 500, rng)[0])
    print(f"  Python sample[0]: mean={sample0.mean(axis=0)}, std={sample0.std(axis=0)}, "
          f"min={sample0.min():.2f}, max={sample0.max():.2f}")
    
    # ── C ──
    cli = find_cli()
    if not cli:
        print("  C CLI not found, skipping.")
        return
    
    tmpdir = tempfile.mkdtemp()
    print(f"  C MAF ({n_flows_c}f, {hidden_c}h, {epochs_c}ep, lr={lr_c})...", end=" ", flush=True)
    model_file = train_c_maf(Xtr, Ytr, n_flows=n_flows_c, hidden=hidden_c, epochs=epochs_c, 
                              lr=lr_c, batch=batch_c, tmpdir=tmpdir)
    if not model_file:
        print("FAILED")
        return
    c_eval = evaluate_c(model_file, Xts, Yts_true, tmpdir=tmpdir)
    print(f"acc={c_eval['acc']:.3f}, MAE={c_eval['mae']:.4f}, max|μ|={c_eval['max_abs_mean']:.2f}, "
          f"mean(σ)={c_eval['mean_std']:.4f}, valid={c_eval['n_valid']}/{c_eval['n_total']}")
    if c_eval['any_nan']:
        print(f"  ⚠️ C produced NaN means!")
    if c_eval['max_abs_mean'] > 100:
        print(f"  ⚠️ C means exploding: max|μ| = {c_eval['max_abs_mean']:.1f}")
    
    # ── Direct comparison on same test samples ──
    # For the first 5 test points, compare Python vs C means
    n_compare = min(5, len(py_eval['means']), len(c_eval['means']))
    print(f"\n  {'idx':>3} | {'true':>8} | {'Py μ':>10} {'Py σ':>8} | {'C μ':>10} {'C σ':>8}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*10} {'-'*8}-+-{'-'*10} {'-'*8}")
    for i in range(n_compare):
        yt = Yts_true[i, 0] if Yts_true.ndim > 1 else Yts_true[i]
        py_m = py_eval['means'][i, 0] if py_eval['means'].ndim > 1 else py_eval['means'][i]
        py_s = py_eval['stds'][i, 0] if py_eval['stds'].ndim > 1 else py_eval['stds'][i]
        c_m = c_eval['means'][i]
        c_s = c_eval['stds'][i]
        print(f"  {i:>3} | {yt:>8.2f} | {py_m:>10.4f} {py_s:>8.4f} | {c_m:>10.4f} {c_s:>8.4f}")
    
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    return {'python': py_eval, 'c': c_eval}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Weight inspection: compare Python vs C weight magnitudes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def inspect_weights(model, label=""):
    """Print weight statistics for a Python MAF model."""
    print(f"\n  Weight statistics ({label}):")
    for k, v in model.weights.items():
        v_np = np.array(v)
        print(f"    {k:>12}: shape={str(v_np.shape):>12}  "
              f"min={v_np.min():>10.4f}  max={v_np.max():>10.4f}  "
              f"mean={np.abs(v_np).mean():.4f}  |max|={np.abs(v_np).max():.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    print("=" * 70)
    print("  MNIST & Classification Diagnostic: Python vs C MAF")
    print("=" * 70)
    
    # ── A. MNIST: Python (baseline config) ──
    print("\n" + "─" * 70)
    print("  A. MNIST — Python MAF (5 flows, 32 hidden, 300 epochs)")
    print("─" * 70)
    Xtr_m, Ytr_m, Xts_m, Yts_true_m, Yts_noisy_m = mnist_data(n_train=496, n_test=200)
    
    py_model_mnist = train_python_maf(Xtr_m, Ytr_m, n_flows=5, hidden=32, epochs=300, lr=0.001)
    py_mnist = evaluate_python(py_model_mnist, Xts_m, Yts_true_m)
    print(f"  Python MNIST: acc={py_mnist['acc']:.3f}, MAE={py_mnist['mae']:.4f}, "
          f"max|μ|={py_mnist['max_abs_mean']:.2f}, mean(σ)={py_mnist['mean_std']:.4f}")
    inspect_weights(py_model_mnist, "Py-MNIST-5f-32h")
    
    # ── B. MNIST: Python (larger model that failed in C) ──
    print("\n" + "─" * 70)
    print("  B. MNIST — Python MAF (8 flows, 64 hidden, 600 epochs)")
    print("─" * 70)
    py_model_mnist_big = train_python_maf(Xtr_m, Ytr_m, n_flows=8, hidden=64, epochs=600, lr=0.001)
    py_mnist_big = evaluate_python(py_model_mnist_big, Xts_m, Yts_true_m)
    print(f"  Python MNIST (big): acc={py_mnist_big['acc']:.3f}, MAE={py_mnist_big['mae']:.4f}, "
          f"max|μ|={py_mnist_big['max_abs_mean']:.2f}, mean(σ)={py_mnist_big['mean_std']:.4f}")
    inspect_weights(py_model_mnist_big, "Py-MNIST-8f-64h")
    
    # ── C. MNIST: C MAF (same configs) ──
    print("\n" + "─" * 70)
    print("  C. MNIST — C MAF (5 flows, 32 hidden, 300 epochs)")
    print("─" * 70)
    run_comparison("MNIST baseline C", Xtr_m, Ytr_m, Yts_true_m,
                   n_flows=5, hidden=32, epochs=300, lr=0.001,
                   n_flows_c=5, hidden_c=32, epochs_c=300, lr_c=0.001, batch_c=32)
    
    print("\n" + "─" * 70)
    print("  D. MNIST — C MAF (8 flows, 64 hidden, 600 epochs)")
    print("─" * 70)
    run_comparison("MNIST big C", Xtr_m, Ytr_m, Yts_true_m,
                   n_flows=8, hidden=64, epochs=600, lr=0.001,
                   n_flows_c=8, hidden_c=64, epochs_c=600, lr_c=0.001, batch_c=32)
    
    # ── E. Binary classification ──
    print("\n" + "─" * 70)
    print("  E. Binary (2-class) classification — Python vs C")
    print("─" * 70)
    X_b, y_b_noisy, y_b_true = make_binary_classification(n=400)
    run_comparison("Binary (2-class)", X_b, y_b_noisy, y_b_true,
                   n_flows=3, hidden=16, epochs=200, lr=0.005,
                   n_flows_c=3, hidden_c=16, epochs_c=200, lr_c=0.005, batch_c=8)
    
    # ── F. 3-class classification ──
    print("\n" + "─" * 70)
    print("  F. 3-class classification — Python vs C")
    print("─" * 70)
    X_3, y_3_noisy, y_3_true = make_multiclass_3(n=600)
    run_comparison("Multiclass (3-class)", X_3, y_3_noisy, y_3_true,
                   n_flows=4, hidden=32, epochs=300, lr=0.003,
                   n_flows_c=4, hidden_c=32, epochs_c=300, lr_c=0.003, batch_c=8)
    
    # ── G. Continuous regression (should work well) ──
    print("\n" + "─" * 70)
    print("  G. Continuous regression (sin(x)+cos(y)+noise) — Python vs C")
    print("─" * 70)
    X_r, y_r_noisy, y_r_true = make_continuous_regression(n=500)
    run_comparison("Continuous regression", X_r, y_r_noisy, y_r_true,
                   n_flows=4, hidden=32, epochs=300, lr=0.003,
                   n_flows_c=4, hidden_c=32, epochs_c=300, lr_c=0.003, batch_c=8)
    
    # ── H. Banana (the simplest CDE benchmark) ──
    print("\n" + "─" * 70)
    print("  H. Banana (2D params, 1D feature) — Python vs C")
    print("─" * 70)
    params_ban, feat_ban = generate_test_data('banana', n_samples=496, seed=42)
    n_ban = len(params_ban)
    n_tr = 392  # multiple of 8
    X_ban_tr = np.array(feat_ban[:n_tr])
    Y_ban_tr = np.array(params_ban[:n_tr])
    X_ban_ts = np.array(feat_ban[n_tr:n_tr+100])
    Y_ban_ts = np.array(params_ban[n_tr:n_tr+100])
    
    print(f"  Python MAF (3 flows, 32 hidden, 300 epochs)...", end=" ", flush=True)
    py_ban = MAFEstimator(param_dim=2, feature_dim=1, n_flows=3, hidden_units=32)
    py_ban.train(Y_ban_tr, X_ban_tr, n_epochs=300, batch_size=8, learning_rate=0.005, use_tqdm=False, seed=42)
    ban_eval = evaluate_python(py_ban, X_ban_ts, Y_ban_ts)
    print(f"MAE={ban_eval['mae']:.4f}, max|μ|={ban_eval['max_abs_mean']:.2f}, mean(σ)={ban_eval['mean_std']:.4f}")
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)