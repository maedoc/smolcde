#!/usr/bin/env bash
# demo_banana.sh — End-to-end smolcde workflow on the banana dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="${SMOLCDE_BIN:-$REPO_ROOT/smolcde}"
PYTHON="${PYTHON:-$REPO_ROOT/env/bin/python3}"

echo "=== SmolCDE Banana Demo ==="
echo "CLI: $CLI"

cd "$REPO_ROOT"

# 1. Generate dataset
echo "[1/4] Generating banana dataset..."
$PYTHON -c "
from cde import generate_test_data
import csv
params, features = generate_test_data('banana', n_samples=500, seed=42)

with open('/tmp/demo_params.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(params)

with open('/tmp/demo_features.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(features)
"
echo "      Generated 500 samples to /tmp/demo_{params,features}.csv"

# 2. Train
echo "[2/4] Training MAF model (50 epochs)..."
$CLI train \
    --features /tmp/demo_features.csv \
    --params /tmp/demo_params.csv \
    --out /tmp/demo_model.maf \
    --epochs 50 \
    --hidden 32 \
    --blocks 4 \
    --lr 0.001 \
    --batch 16 \
    --seed 42
echo "      Model saved to /tmp/demo_model.maf"

# 3. Inference — sample mode
echo "[3/4] Running inference (sample mode)..."
$CLI infer \
    --model /tmp/demo_model.maf \
    --features /tmp/demo_features.csv \
    --out /tmp/demo_samples.csv \
    --mode sample \
    --samples 8
echo "      Samples saved to /tmp/demo_samples.csv"
echo "      First 3 rows:"
head -4 /tmp/demo_samples.csv

# 4. Inference — stats and quantiles
echo "[4/4] Running inference (stats + quantiles)..."
$CLI infer \
    --model /tmp/demo_model.maf \
    --features /tmp/demo_features.csv \
    --out /tmp/demo_stats.csv \
    --mode stats \
    --samples 64

$CLI infer \
    --model /tmp/demo_model.maf \
    --features /tmp/demo_features.csv \
    --out /tmp/demo_quantiles.csv \
    --mode quantiles \
    --samples 64 \
    --quantiles-list "0.05,0.50,0.95"

echo "      Stats: $(wc -l < /tmp/demo_stats.csv) rows"
echo "      Quantiles: $(wc -l < /tmp/demo_quantiles.csv) rows"
echo ""
echo "=== Demo complete ==="
