#!/usr/bin/env python3
"""thorough_mnist_onehot.py — Systematic test of MAF on MNIST with one-hot encoding.

Hypothesis: MAF D=10 (one-hot) should work if trained carefully.
Previous failure was likely due to: learning rate too high, too many flows, or
other hyperparameter issues. This script systematically tests configurations.
"""
import sys, os, csv, tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import autograd.numpy as anp
from cde import MAFEstimator, MDNEstimator
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


def one_hot(labels, n_classes=10, noise_std=0.01, rng=None):
    """Convert labels to one-hot with optional label smoothing."""
    oh = np.zeros((len(labels), n_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    if noise_std > 0 and rng is not None:
        oh += rng.normal(0, noise_std, size=oh.shape).astype(np.float32)
    return oh


def evaluate_maf(model, X_test, y_test_int, n_samples=500, seed=99):
    """Evaluate MAF: sample posterior, argmax → predicted digit, report accuracy."""
    rng = anp.random.RandomState(seed)
    accs, maes, maxes = [], [], []
    sample_ok = 0
    n_cond = min(len(X_test), 50)
    for i in range(n_cond):
        try:
            s = np.array(model.sample(X_test[i:i+1], n_samples, rng)[0])
            if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.max(np.abs(s)) > 1e4:
                continue
            sample_ok += 1
            mean = s.mean(axis=0)
            pred = np.argmax(mean)
            accs.append(int(pred == y_test_int[i]))
            maes.append(np.mean(np.abs(s.mean(axis=0) - one_hot(np.array([y_test_int[i]]), 10, noise_std=0)[0])))
            maxes.append(np.max(np.abs(s)))
        except Exception:
            continue
    if not accs:
        return {'acc': 0.0, 'mae': float('inf'), 'max_abs': float('inf'),
                'n_ok': 0, 'n_total': n_cond, 'sample_ok_ratio': 0.0}
    return {
        'acc': np.mean(accs),
        'mae': np.mean(maes),
        'max_abs': np.max(maxes),
        'n_ok': sample_ok,
        'n_total': n_cond,
        'sample_ok_ratio': sample_ok / n_cond,
    }


def evaluate_mdn(model, X_test, y_test_int, n_samples=500, seed=99, D=1):
    """Evaluate MDN on scalar or one-hot labels."""
    rng = anp.random.RandomState(seed)
    samples = np.array(model.sample(X_test[:50], n_samples, rng))
    if D == 1:
        means = samples.mean(axis=1)[:, 0]
        preds = np.round(means).astype(int)
        acc = np.mean(preds == y_test_int[:50])
        return {'acc': acc, 'mae': np.mean(np.abs(y_test_int[:50] - means)),
                'max_abs': np.max(np.abs(samples)), 'sample_ok_ratio': 1.0}
    else:
        means = samples.mean(axis=1)
        preds = np.argmax(means, axis=1)
        acc = np.mean(preds == y_test_int[:50])
        return {'acc': acc, 'max_abs': np.max(np.abs(samples)),
                'sample_ok_ratio': float(np.all(np.isfinite(samples)) & (np.max(np.abs(samples)) < 1e4))}


def train_and_eval(config, X_train, y_oh_train, X_test, y_test_int, label=""):
    """Train MAF with given config and evaluate."""
    D = y_oh_train.shape[1]
    C = X_train.shape[1]
    n_f = config.get('n_flows', 4)
    h = config.get('hidden', 32)
    ep = config.get('epochs', 300)
    lr = config.get('lr', 0.001)
    bs = config.get('batch', 8)
    seed = config.get('seed', 42)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Config: flows={n_f}, hidden={h}, epochs={ep}, lr={lr}, batch={bs}")
    print(f"{'='*60}")

    try:
        model = MAFEstimator(param_dim=D, feature_dim=C, n_flows=n_f, hidden_units=h)
        model.train(y_oh_train, X_train, n_epochs=ep, batch_size=bs,
                    learning_rate=lr, use_tqdm=False, seed=seed)
        # Check final loss
        if hasattr(model, 'loss_history') and len(model.loss_history) > 0:
            print(f"  Final loss: {model.loss_history[-1]:.4f}")
        ev = evaluate_maf(model, X_test, y_test_int)
        print(f"  Accuracy: {ev['acc']:.3f}, MAE: {ev['mae']:.4f}")
        print(f"  Max |sample|: {ev['max_abs']:.2f}")
        print(f"  Valid samples: {ev['n_ok']}/{ev['n_total']} ({ev['sample_ok_ratio']:.0%})")
        # Check weight magnitudes
        max_w = max(float(np.max(np.abs(v))) for v in model.weights.values())
        print(f"  Max |weight|: {max_w:.4f}")
        return ev
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


if __name__ == '__main__':
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target

    N_train = 500
    N_test = 200
    X_train, X_test = X[:N_train], X[N_train:N_train+N_test]
    y_train_int, y_test_int = y[:N_train], y[N_train:N_train+N_test]

    rng = np.random.RandomState(42)

    # ── A. One-hot D=10 with various hyperparams ──
    print("\n" + "#"*60)
    print("# A. MAF D=10 (one-hot) — hyperparameter sweep")
    print("#"*60)

    configs_oh = [
        {"n_flows": 2, "hidden": 32, "epochs": 100, "lr": 0.001, "batch": 8,  "label": "A1: conservative (2f, 32h, 100ep)"},
        {"n_flows": 2, "hidden": 32, "epochs": 300, "lr": 0.001, "batch": 8,  "label": "A2: more epochs (2f, 32h, 300ep)"},
        {"n_flows": 4, "hidden": 32, "epochs": 100, "lr": 0.001, "batch": 8,  "label": "A3: more flows (4f, 32h, 100ep)"},
        {"n_flows": 2, "hidden": 64, "epochs": 100, "lr": 0.001, "batch": 8,  "label": "A4: wider (2f, 64h, 100ep)"},
        {"n_flows": 2, "hidden": 32, "epochs": 100, "lr": 0.0003, "batch": 8, "label": "A5: lower LR (2f, 32h, lr=0.0003)"},
        {"n_flows": 2, "hidden": 32, "epochs": 300, "lr": 0.0003, "batch": 8, "label": "A6: lower LR + more epochs"},
        {"n_flows": 2, "hidden": 32, "epochs": 100, "lr": 0.0001, "batch": 8, "label": "A7: very low LR (0.0001)"},
        {"n_flows": 2, "hidden": 16, "epochs": 100, "lr": 0.001, "batch": 8,  "label": "A8: small model (2f, 16h, 100ep)"},
        {"n_flows": 2, "hidden": 16, "epochs": 100, "lr": 0.0003, "batch": 8, "label": "A9: small + low LR (2f, 16h, lr=0.0003)"},
        {"n_flows": 2, "hidden": 32, "epochs": 100, "lr": 0.001, "batch": 32, "label": "A10: larger batch (32)"},
    ]

    results_oh = []
    for cfg in configs_oh:
        y_oh_train = one_hot(y_train_int, noise_std=0.01, rng=rng)
        # Try label smoothing instead of noise
        if cfg.get('label_smooth', False):
            y_oh_train = one_hot(y_train_int, noise_std=0, rng=rng)
            y_oh_train = y_oh_train * 0.9 + 0.01  # soft label smoothing
        ev = train_and_eval(cfg, X_train, y_oh_train, X_test, y_test_int, label=cfg['label'])
        if ev:
            results_oh.append((cfg['label'], ev))

    # ── B. Label smoothing (instead of Gaussian noise) ──
    print("\n" + "#"*60)
    print("# B. Label smoothing (0.9 confidence, 0.01 uniform)")
    print("#"*60)

    for n_f, h, ep, lr in [(2, 32, 100, 0.001), (2, 32, 300, 0.001), (4, 32, 100, 0.0003)]:
        y_smooth = one_hot(y_train_int, noise_std=0, rng=rng) * 0.9 + 0.01
        ev = train_and_eval(
            {"n_flows": n_f, "hidden": h, "epochs": ep, "lr": lr, "batch": 8},
            X_train, y_smooth, X_test, y_test_int,
            label=f"B: smooth(0.9) {n_f}f/{h}h/{ep}ep/lr={lr}")
        if ev:
            results_oh.append((f"smooth {n_f}f/{h}h/{ep}ep", ev))

    # ── C. More training data ──
    print("\n" + "#"*60)
    print("# C. Full dataset (1797 samples)")
    print("#"*60)

    X_full_train = X[:1500]
    y_full_int = y[:1500]
    X_full_test = X[1500:]
    y_full_test = y[1500:]
    y_oh_full = one_hot(y_full_int, noise_std=0.01, rng=rng)

    for n_f, h, ep, lr in [(2, 32, 200, 0.001), (2, 32, 200, 0.0003), (4, 64, 200, 0.0003)]:
        ev = train_and_eval(
            {"n_flows": n_f, "hidden": h, "epochs": ep, "lr": lr, "batch": 16},
            X_full_train, y_oh_full, X_full_test, y_full_test,
            label=f"C: full data {n_f}f/{h}h/{ep}ep/lr={lr}")

    # ── D. MDN baselines ──
    print("\n" + "#"*60)
    print("# D. MDN baselines (for comparison)")
    print("#"*60)

    # D=1 MDN
    y_scalar = y_train_int.reshape(-1, 1).astype(np.float32) + rng.normal(0, 0.1, size=(N_train, 1)).astype(np.float32)
    mdn1 = MDNEstimator(param_dim=1, feature_dim=64, n_components=5, hidden_sizes=(64, 32))
    mdn1.train(y_scalar, X_train, n_epochs=300, learning_rate=0.005, use_tqdm=False, seed=42)
    ev_mdn1 = evaluate_mdn(mdn1, X_test, y_test_int, D=1)
    print(f"\n  MDN D=1 (scalar): acc={ev_mdn1['acc']:.3f}")

    # D=10 MDN
    y_oh10 = one_hot(y_train_int, noise_std=0.01, rng=rng)
    mdn10 = MDNEstimator(param_dim=10, feature_dim=64, n_components=5, hidden_sizes=(64, 32))
    mdn10.train(y_oh10, X_train, n_epochs=300, learning_rate=0.005, use_tqdm=False, seed=42)
    ev_mdn10 = evaluate_mdn(mdn10, X_test, y_test_int, D=10)
    print(f"  MDN D=10 (one-hot): acc={ev_mdn10['acc']:.3f}")

    # ── Summary ──
    print("\n" + "="*60)
    print("  SUMMARY — MAF D=10 (one-hot) results")
    print("="*60)
    print(f"  {'Config':<40} {'Acc':>6} {'MAE':>8} {'Max|s|':>10} {'OK%':>5}")
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*10} {'-'*5}")
    for label, ev in sorted(results_oh, key=lambda x: -x[1]['acc']):
        print(f"  {label:<40} {ev['acc']:>6.3f} {ev['mae']:>8.4f} {ev['max_abs']:>10.2f} {ev['sample_ok_ratio']:>5.0%}")
    print()
    print(f"  MDN D=1 baseline:   {ev_mdn1['acc']:.3f}")
    print(f"  MDN D=10 baseline:  {ev_mdn10['acc']:.3f}")