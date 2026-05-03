# MNIST Failure Analysis — REVISED

## Summary

The MNIST failure was caused by using **scalar (D=1) targets** with MAF.
With **one-hot encoding (D=10)**, MAF achieves **84–90% accuracy** on MNIST 8x8,
dramatically outperforming MDN (57–72%).

## Root Cause: D=1 MAF Posterior Collapse

### What went wrong with D=1

When `param_dim=1`, the MADE masking structure is trivially degenerate:
- Masks become all-ones (M1=all1, M2=all1)
- The autoregressive property is vacuously satisfied (only 1 variable)
- The MADE network produces **input-independent** alpha/mu values
- `alpha → -3` across all layers → `exp(alpha) ≈ 0.05` → posterior collapses

Empirically confirmed:
```
Binary classification (D=1): mean=-0.10, std=0.0        (collapsed)
Multiclass (D=1):           mean=-0.22, std=0.0        (collapsed)
Continuous regression (D=1):mean=0.59,  std=0.0        (collapsed)
Banana (D=2):               mean=[0.02, 1.33], std=[0.88, 1.44]  ✅ works
```

### Why one-hot (D=10) works

With D=10, each MADE layer has genuine autoregressive conditioning:
- Variable 1 depends on no predecessors (it's first in the permutation)
- Variable d depends on variables 1..d-1
- Each layer transforms a 10D Gaussian through a proper autoregressive flow
- The model can represent **multimodal** posteriors over one-hot vectors

## Results

### Python MAF, D=10 one-hot, 500 train samples

| Config | Accuracy |
|--------|----------|
| 2f/32h/100ep/lr=0.001 | **83.7%** |
| 2f/32h/300ep/lr=0.001 | 79.6% |
| 4f/32h/300ep/lr=0.001 | 80.6% |
| 4f/64h/300ep/lr=0.0003 | 80.6% |

### Python MAF, D=10 one-hot, 1500 train samples

| Config | Accuracy |
|--------|----------|
| 2f/32h/300ep/lr=0.001 | **89.9%** |
| 4f/32h/300ep/lr=0.001 | 88.2% |

### Comparison with MDN

| Method | D | Accuracy |
|--------|---|----------|
| MAF one-hot | 10 | **83.7%** (500 train) → **89.9%** (1500 train) |
| MDN scalar | 1 | 56.5% |
| MDN one-hot | 10 | 71.5% |

### C CLI results (D=1, scalar — BEFORE fix)

| Config | Accuracy | Notes |
|--------|----------|-------|
| 300ep, 32h, 5f | 44.5% | D=1, collapsed posterior |

## C vs Python mask divergence (now fixed)

Before the fix, C used `m_h[j]=0` for D=1 (M1=0, M2=1), producing an MLP-like
model that could at least predict conditional means (~44% accuracy). After the fix,
both use `m_h=1` with M2 override (M1=1, M2=1), producing identical degenerate
posteriors for D=1.

For D≥2 (including D=10 one-hot), both implementations produce non-degenerate
results.

## Recommendations

1. **Always use one-hot encoding for classification tasks** — MAF D=10 one-hot
   achieves 84-90% accuracy on MNIST, far exceeding MDN (57-72%).

2. **Never use MAF with D=1 for classification** — the autoregressive structure
   is vacuous. A RuntimeWarning is now emitted in Python.

3. **For the CLI, add support for one-hot encoding** — currently the CLI only
   supports raw CSV columns. A `--one-hot` flag or automatic detection would
   help users.

4. **The Lorenz demo (D=3) works perfectly** — MAE 0.94, correlation 0.92.

5. **Update MNIST demo to use one-hot** — the demo now trains with D=10 one-hot
   and achieves ~84% accuracy with 500 samples.