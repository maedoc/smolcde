# MNIST Failure Analysis

## Summary

The MNIST (D=1) failure is **NOT a C implementation bug** — it's an architectural
limitation of MAF for univariate (param_dim=1) targets. Both Python and C produce
the same degenerate behavior.

## Root Cause

### MADE degeneracy for D=1

With `param_dim=1`, the MADE masking is trivially degenerate:
- `m_in = [1]`, `m_hidden ∈ {1}` → M1 = all ones, M2 = all ones
- The autoregressive property is trivially satisfied (only 1 variable)
- The MADE network can **ignore the noise input entirely** and just condition on features

### Empirical evidence

```
Binary classification (D=1, 2-class):
  Python: mean=-0.10, std=0.0  (collapsed!)

Multiclass (D=1, 3-class):
  Python: mean=-0.22, std=0.0  (collapsed!)

Continuous regression (D=1, sin+cos+noise):
  Python: mean=0.59, std=0.0  (collapsed!)

Banana (D=2):
  Python: mean=[0.02, 1.33], std=[0.88, 1.44]  (works!)
```

### Alpha collapse

Sampled alpha values across all layers are **constant** (input-independent):

```
Layer 2: alpha=-3.02, mu=0.04, exp(alpha)=0.049  → scale factor is ~5%
Layer 1: alpha=-2.91, mu=-0.09, exp(alpha)=0.055  → scale factor is ~5%
Layer 0: alpha=-2.66, mu=-0.21, exp(alpha)=0.070   → scale factor is ~7%
```

The MADE network learns `mu(f)` and `alpha(f)` — both are input-independent
functions of the features only. The noise `z` is effectively ignored because
the model sets alpha so negative that `exp(alpha) ≈ 0`, making the output
nearly deterministic.

### Why the C model appeared to work (44%)

The original C code had `m_h[j] = 0` for D=1, producing M1=all-zeros and
M2=all-ones. This meant the hidden layer received **no input from y** (only
context from features). The model was effectively a 2-layer MLP conditioned on
features, producing shift and scale parameters for a simple affine transform of
the base Gaussian. This MLP could learn the conditional mean, giving ~44% accuracy.

After fixing both Python and C to use M1=all-ones, M2=all-ones, the model has
the capacity to use the input — but it doesn't learn to, because ignoring it
is a local minimum that still reduces NLL.

## Why it works for D≥2

When `param_dim ≥ 2`, each MADE layer conditions on the already-sampled
dimensions. The autoregressive structure creates a genuine sequential dependency:
`y[1]` depends on `y[0]`, `y[2]` depends on `y[0]` and `y[1]`, etc. This
prevents the model from collapsing because different noise realizations `z`
produced genuinely different outputs through the chain of dependencies.

For D=1, there's only one dimension — no autoregressive conditioning is possible.
The flow reduces to a composition of `D` affine transforms of a single variable,
which can only shift and scale the base distribution, not reshape it into
multiple modes.

## C vs Python divergence

Before the mask fix, the C and Python implementations had **different masks**
for D=1:

| Implementation | m_hidden | M1 | M2 | Result |
|---|---|---|---|---|
| C (original) | 0 | all-zeros | all-ones | MLP-like, ~44% acc |
| C (fixed) | 1 | all-ones | all-ones | Same as Python, collapsed |
| Python (fixed) | 1 | all-ones | all-ones | Collapsed posterior |

After fix, both implementations agree and both produce collapsed posteriors.

## Impact on Lorenz and other D≥2 tasks

Lorenz (D=3) is unaffected — the autoregressive conditioning works correctly
when `param_dim ≥ 2`. The best Lorenz result is:

```
MAE avg = 0.94, correlation avg = 0.916
```

## Recommendations

1. **Document limitation**: MAF is suitable for D≥2 parameter estimation.
   For D=1 (scalar regression), MAF degenerates and should not be used.

2. **Add MDN for D=1**: The `MDNEstimator` class already handles D=1 correctly
   by learning a mixture of Gaussians, which can represent multimodal posteriors.
   For classification-like tasks with D=1, MDN is the right tool.

3. **Add runtime warning**: When param_dim=1, print a warning suggesting MDN
   instead of MAF.

4. **Keep the mask fix**: The (M1=all1, M2=all1) fix is correct per the MADE
   paper. It doesn't make things worse — both versions produce collapsed
   posteriors, just through different mechanisms.

5. **Update demos**: The MNIST demo should use MDN, not MAF. The Lorenz demo
   (D=3) can continue using MAF.