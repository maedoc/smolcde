# Code Review Fixes TODO

## Blocking
- [x] Fix `test_mnist_cli_workflow`: evaluate on held-out test data, not training data
- [x] Add `results_*.jsonl` to `.gitignore`, remove from git tracking

## Non-blocking
- [x] Clean dead code in `test_c_gradient_matches_autograd`
- [x] Add test for `maf_backward` with `feature_dim=0`
- [x] Add test for `load_model_file` returning NULL on bad file (also fixed main.c to return non-zero)
- [x] Remove `benchmarks/thorough_mnist_onehot.py` (stale WIP)
- [x] Clean up `benchmarks/diagnose_mnist.py` header comment

## Bonus (found during fixes)
- [x] Fix main.c cmd_train/cmd_infer to return int instead of void, propagate error exit codes
