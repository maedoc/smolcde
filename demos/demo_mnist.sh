#!/usr/bin/env bash
# demo_mnist.sh — SmolCDE trained on MNIST 8×8 digit images → predict digit label.
# Demonstrates high-dimensional features (64 pixels) → 1D parameter (label).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="${SMOLCDE_BIN:-$REPO_ROOT/smolcde}"
PYTHON="${PYTHON:-$REPO_ROOT/env/bin/python3}"
OUTDIR="${TMPDIR:-/tmp}/smolcde-mnist"
mkdir -p "$OUTDIR"

N_TRAIN=500
N_TEST=200
EPOCHS=300   # fewer for a quick demo; bump to 600+ for ~85% accuracy
HIDDEN=32
BLOCKS=5
BATCH=32
SEED=42
SAMPLES=64

echo "=== SmolCDE MNIST Demo ==="
echo "CLI:           $CLI"
echo "Output dir:    $OUTDIR"
echo "Train samples: $N_TRAIN"
echo "Test samples:  $N_TEST"
echo "Epochs:        $EPOCHS"
echo ""

# ── 1. Generate & split data ──────────────────────────────────────────────────
echo "[1/5] Generating MNIST data..."

$PYTHON -c "
import numpy as np, csv
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target.astype(float)

# Train split
N = $N_TRAIN
Xtr, Xts = X[:N], X[N:N+$N_TEST]
ytr, yts = y[:N], y[N:N+$N_TEST]

# Add noise to labels so the model learns a distribution, not a delta
rng = np.random.RandomState($SEED)
ytr_noisy = ytr + rng.normal(0, 0.1, size=ytr.shape)
yts_noisy = yts + rng.normal(0, 0.1, size=yts.shape)

# Save CSVs
np.savetxt('$OUTDIR/feat_train.csv', Xtr, delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/feat_test.csv',  Xts, delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/param_train.csv', ytr_noisy.reshape(-1,1), delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/param_test.csv',  yts_noisy.reshape(-1,1), delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/true_test_labels.csv', yts.reshape(-1,1), delimiter=',', fmt='%.0f')
"
echo "  → $N_TRAIN train rows, $N_TEST test rows"

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

# ── 5. Evaluate ────────────────────────────────────────────────────────────────
echo "[5/5] Evaluating accuracy..."

$PYTHON -c "
import numpy as np, csv

# Read posterior means
means = []
with open('$OUTDIR/pred_stats.csv') as f:
    for row in csv.DictReader(f):
        if row['stat'] == 'mean':
            means.append(float(row['p0']))
means = np.array(means)

y_pred = np.round(means).astype(int)
y_true = np.loadtxt('$OUTDIR/true_test_labels.csv', delimiter=',').astype(int)

acc   = np.mean(y_true == y_pred)
mae   = np.mean(np.abs(y_true - means))

print('')
print('  ┌─────────────────────────────────────┐')
print(f'  │  Accuracy:        {acc:.3f}               │')
print(f'  │  Mean Abs Error:  {mae:.3f}               │')
print('  ├─────────────────────────────────────┤')
print(f'  │  Perfect guesses:  {np.sum(y_true==y_pred):>3d} / {len(y_true)}         │')
print('  └─────────────────────────────────────┘')
print('')
print(f'  Baseline (chance):  0.100  (10 classes)')
print(f'  Baseline (mean):    {np.mean(np.abs(y_true - np.mean(y_true))):.3f}  (predict mean label)')
if acc < 0.15:
    print('')
    print('  ⚠  Accuracy is low — try more epochs (--epochs 600+)')
    print('     or train a bigger model (--hidden 64 --blocks 8).')
"
echo ""
echo "=== Done ==="
