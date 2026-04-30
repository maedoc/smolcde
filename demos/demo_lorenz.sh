#!/usr/bin/env bash
# demo_lorenz.sh — Estimate Lorenz attractor parameters (σ,ρ,β) from
# log-power-spectrum features using SmolCDE.
#
# Concept: the Lorenz system x'=σ(y-x), y'=x(ρ-z)-y, z'=xy-βz produces
# a characteristic power spectrum.  We vary (σ,ρ,β), simulate the ODE,
# compute Welch PSD, and train smolcde to recover params from the spectrum.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="${SMOLCDE_BIN:-$REPO_ROOT/smolcde}"
PYTHON="${PYTHON:-$REPO_ROOT/env/bin/python3}"
OUTDIR="${TMPDIR:-/tmp}/smolcde-lorenz"
mkdir -p "$OUTDIR"

N_TRAIN=800
N_TEST=200
EPOCHS=500
HIDDEN=64
BLOCKS=4
BATCH=32
LR=0.0002
SEED=42
SAMPLES=200

echo "=== SmolCDE Lorenz Demo ==="
echo "CLI:           $CLI"
echo "Output dir:    $OUTDIR"
echo "Train samples: $N_TRAIN"
echo "Test samples:  $N_TEST"
echo "Epochs:        $EPOCHS"
echo "Hidden units:  $HIDDEN"
echo "Flows:         $BLOCKS"
echo ""

# ── 1. Generate data ────────────────────────────────────────────────────────────
echo "[1/4] Generating Lorenz data (simulating ODE + Welch PSD) — this may take a minute..."

$PYTHON -c "
import numpy as np
from scipy.integrate import odeint
from scipy.signal import welch

def lorenz_ode(state, t, sigma, rho, beta):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]

t = np.linspace(0, 5, 500)
fs = 100
N = $N_TRAIN + $N_TEST

rng = np.random.RandomState($SEED)
params_list, features_list = [], []

for i in range(N):
    sigma = rng.uniform(5, 15)
    rho   = rng.uniform(20, 40)
    beta  = rng.uniform(1, 5)

    states = odeint(lorenz_ode, [1.0, 1.0, 1.0], t, args=(sigma, rho, beta))
    x_ts = states[:, 0]
    f, Pxx = welch(x_ts, fs=fs, nperseg=128)
    log_Pxx = np.log10(Pxx + 1e-10)

    params_list.append([sigma, rho, beta])
    features_list.append(log_Pxx)

    if (i+1) % 200 == 0:
        print(f'    {i+1}/{N} samples generated...')

features = np.array(features_list, dtype=np.float32)
params   = np.array(params_list,   dtype=np.float32)

# Save train/test splits
np.savetxt('$OUTDIR/feat_train.csv', features[:$N_TRAIN],     delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/feat_test.csv',  features[$N_TRAIN:],     delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/param_train.csv', params[:$N_TRAIN],      delimiter=',', fmt='%.8f')
np.savetxt('$OUTDIR/param_test.csv',  params[$N_TRAIN:],      delimiter=',', fmt='%.8f')

# Print feature dimension
print(f'')
print(f'  Feature dim: {features.shape[1]}  (PSD frequency bins)')
print(f'  Param dim:   {params.shape[1]}    (σ, ρ, β)')
print(f'  Param ranges: σ∈[5,15], ρ∈[20,40], β∈[1,5]')
"
echo "  → $N_TRAIN train + $N_TEST test rows"

# ── 2. Train ───────────────────────────────────────────────────────────────────
echo "[2/4] Training (${EPOCHS} epochs, ${HIDDEN}h, ${BLOCKS} flows, lr=${LR})..."

$CLI train \
    --features "$OUTDIR/feat_train.csv" \
    --params   "$OUTDIR/param_train.csv" \
    --out      "$OUTDIR/lorenz.maf" \
    --epochs   "$EPOCHS" \
    --hidden   "$HIDDEN" \
    --blocks   "$BLOCKS" \
    --lr       "$LR" \
    --batch    "$BATCH" \
    --seed     "$SEED"

MODEL_KB=$(du -k "$OUTDIR/lorenz.maf" | cut -f1)
echo "  → Model saved ($MODEL_KB KB)"

# ── 3. Infer (stats + quantiles) ────────────────────────────────────────────────
echo "[3/4] Inferring posterior on test set..."

$CLI infer \
    --model    "$OUTDIR/lorenz.maf" \
    --features "$OUTDIR/feat_test.csv" \
    --out      "$OUTDIR/pred_stats.csv" \
    --mode     stats \
    --samples  "$SAMPLES"

$CLI infer \
    --model          "$OUTDIR/lorenz.maf" \
    --features       "$OUTDIR/feat_test.csv" \
    --out            "$OUTDIR/pred_quant.csv" \
    --mode           quantiles \
    --samples        "$SAMPLES" \
    --quantiles-list "0.05,0.50,0.95"

echo "  → Done"

# ── 4. Evaluate ─────────────────────────────────────────────────────────────────
echo "[4/4] Evaluating parameter recovery..."

$PYTHON -c "
import numpy as np, csv

# Load true params and predicted means
y_true = np.loadtxt('$OUTDIR/param_test.csv', delimiter=',')  # (N, 3)
assert y_true.shape[1] == 3, f'Expected (N,3), got {y_true.shape}'

means = []
stds  = []
with open('$OUTDIR/pred_stats.csv') as f:
    for row in csv.DictReader(f):
        means.append([float(row['p0']), float(row['p1']), float(row['p2'])])
        stds.append([float(row['std0']), float(row['std1']), float(row['std2'])])

# Stats file has mean/std rows interleaved per feature
# Extract means (every other row starting at 0)
means_arr = np.array(means[::2])   # even rows = mean
stds_arr  = np.array(means[1::2])  # odd rows = std

param_names = ['σ (sigma)', 'ρ (rho)', 'β (beta)']
ranges = [(5,15), (20,40), (1,5)]

print('')
print('  ┌─────────────────────────────────────────────────────────┐')
print('  │  Parameter Recovery Results                             │')
print('  ├────────────────┬────────────┬────────────┬──────────────┤')
print('  │  Parameter     │  MAE       │  Corr(r)   │  Avg σ_pred  │')
print('  ├────────────────┼────────────┼────────────┼──────────────┤')
for j, name in enumerate(param_names):
    mae  = np.mean(np.abs(y_true[:,j] - means_arr[:,j]))
    corr = np.corrcoef(y_true[:,j], means_arr[:,j])[0,1]
    avg_std = np.mean(stds_arr[:,j])
    print(f'  │ {name:<14} │ {mae:>8.4f}   │ {corr:>8.4f}   │ {avg_std:>10.4f}   │')
print('  └────────────────┴────────────┴────────────┴──────────────┘')
print('')
print('  Interpretation:')
print('    MAE       — mean absolute error (lower = better recovery)')
print('    Corr(r)   — Pearson correlation true vs predicted (higher = tracks truth)')
print('    Avg σ_pred— mean posterior std-dev across test set')
print('')

# Check if the model learned anything useful
max_corr = max(np.corrcoef(y_true[:,j], means_arr[:,j])[0,1] for j in range(3))
if max_corr > 0.5:
    print('  ✓ Model recovers Lorenz parameters from PSD features.')
else:
    print('  ⚠ Correlation is low — try more epochs or larger model:')
    print('    --epochs 1000 --hidden 128 --blocks 6')
"
echo ""
echo "=== Done ==="
