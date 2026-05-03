#!/usr/bin/env bash
# demo_mnist.sh — SmolCDE trained on MNIST 8x8 digit images → predict digit label.
#
# IMPORTANT: MAF with scalar (D=1) targets produces collapsed posteriors.
# For classification tasks, use one-hot encoding (D=10) with MAF, or use MDN.
# This demo uses one-hot encoding with the MAF estimator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="${SMOLCDE_BIN:-$REPO_ROOT/smolcde}"
PYTHON="${PYTHON:-$REPO_ROOT/env/bin/python3}"
OUTDIR="${TMPDIR:-/tmp}/smolcde-mnist"
mkdir -p "$OUTDIR"

N_TRAIN=496   # multiple of 8 for C batching
N_TEST=200
EPOCHS=300   # gives ~84% accuracy; bump to 500+ for ~90%
HIDDEN=32
BLOCKS=2
BATCH=8
SEED=42
SAMPLES=64

echo "=== SmolCDE MNIST Demo (one-hot MAF, D=10) ==="
echo "CLI:           $CLI"
echo "Output dir:    $OUTDIR"
echo "Train samples: $N_TRAIN"
echo "Test samples:  $N_TEST"
echo "Epochs:        $EPOCHS"
echo ""

# ── 1. Generate & split data (one-hot encoded) ────────────────────────────────
echo "[1/5] Generating MNIST data (one-hot D=10)..."

$PYTHON -c "
import numpy as np, csv
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

N_train = $N_TRAIN
N_test = $N_TEST
Xtr, Xts = X[:N_train], X[N_train:N_train+N_test]
ytr, yts = y[:N_train], y[N_train:N_train+N_test]

# One-hot encode labels with slight noise for numerical stability
rng = np.random.RandomState($SEED)
ytr_oh = np.zeros((N_train, 10), dtype=np.float32)
ytr_oh[np.arange(N_train), ytr] = 1.0
ytr_oh += rng.normal(0, 0.01, size=ytr_oh.shape).astype(np.float32)

np.savetxt('$OUTDIR/feat_train.csv', Xtr, delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/feat_test.csv',  Xts, delimiter=',', fmt='%.8f')

# Save one-hot params (10 columns)
with open('$OUTDIR/param_train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(ytr_oh)

# Save true labels for evaluation
np.savetxt('$OUTDIR/true_test_labels.csv', yts.reshape(-1, 1), delimiter=',', fmt='%.0f')

print(f'  → {N_train} train rows, {N_test} test rows, param_dim=10')
"
echo "  → Data generated"

# ── 2. Train ───────────────────────────────────────────────────────────────────
echo "[2/5] Training (${EPOCHS} epochs, ${HIDDEN}h, ${BLOCKS} flows) — this may take a minute..."

$CLI train \
    --features "$OUTDIR/feat_train.csv" \
    --params   "$OUTDIR/param_train.csv" \
    --out      "$OUTDIR/mnist.maf" \
    --epochs   "$EPOCHS" \
    --hidden   "$HIDDEN" \
    --blocks   "$BLOCKS" \
    --lr       0.001 \
    --batch    "$BATCH" \
    --seed     "$SEED"

MODEL_KB=$(du -k "$OUTDIR/mnist.maf" | cut -f1)
echo "  → Model saved ($MODEL_KB KB)"

# ── 3. Infer (stats mode) ──────────────────────────────────────────────────────
echo "[3/5] Inferring posterior on test set (stats mode, $SAMPLES samples)..."
$CLI infer \
    --model    "$OUTDIR/mnist.maf" \
    --features "$OUTDIR/feat_test.csv" \
    --out      "$OUTDIR/pred_stats.csv" \
    --mode     stats \
    --samples  "$SAMPLES"
echo "  → Done"

# ── 4. Infer (quantiles) ───────────────────────────────────────────────────────
echo "[4/5] Inferring posterior quantiles..."
$CLI infer \
    --model          "$OUTDIR/mnist.maf" \
    --features       "$OUTDIR/feat_test.csv" \
    --out            "$OUTDIR/pred_quant.csv" \
    --mode           quantiles \
    --samples        "$SAMPLES" \
    --quantiles-list "0.05,0.50,0.95"
echo "  → Done"

# ── 5. Evaluate ─────────────────────────────────────────────────────────────────
echo "[5/5] Evaluating accuracy..."

$PYTHON -c "
import numpy as np, csv

# Read posterior means (10-dimensional)
means = []
with open('$OUTDIR/pred_stats.csv') as f:
    for row in csv.DictReader(f):
        if row['stat'] == 'mean':
            means.append([float(row[f'p{i}']) for i in range(10)])
means = np.array(means)

y_true = np.loadtxt('$OUTDIR/true_test_labels.csv', delimiter=',').astype(int)
N = min(len(means), len(y_true))
means = means[:N]
y_true = y_true[:N]

# Argmax of mean → predicted digit
y_pred = np.argmax(means, axis=1)
acc = np.mean(y_true == y_pred)

print('')
print('  ┌───────────────────────────────────────────────────┐')
print(f'  │  Accuracy (one-hot MAF, D=10):   {acc:.3f}            │')
print(f'  │  Correct: {np.sum(y_true==y_pred):>3d} / {N}                       │')
print('  ├───────────────────────────────────────────────────┤')
print(f'  │  Baseline (chance):             0.100 (10 classes)  │')
print(f'  │  Baseline (predict mean class): 0.104               │')
print('  └───────────────────────────────────────────────────┘')
print('')
if acc < 0.70:
    print('  ⚠  Accuracy below 70% — try more epochs (--epochs 500+)')
    print('     or a bigger model (--hidden 64 --blocks 4).')
"
echo ""
echo "=== Done ==="