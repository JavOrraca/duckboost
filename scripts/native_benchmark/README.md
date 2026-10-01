# Native vs LightGBM medium-scale benchmark

Reproducible holdout comparison of duckboost's built-in histogram trainer
(`backend='native'`) against in-process LightGBM (`backend='lightgbm'`) on a
deterministic synthetic tabular problem.

## Requirements

- Release build with LightGBM linked, e.g.
  `EXT_FLAGS='-DDUCKBOOST_WITH_LIGHTGBM=ON' make release`
- `LD_LIBRARY_PATH` including the LightGBM shared library (the runner adds the
  common pip-wheel path under `~/.local` when present)

## Run

From the repository root:

```bash
python3 scripts/native_benchmark/run.py
python3 scripts/native_benchmark/run.py --n-rows 200000 --seed 7
```

Writes `scripts/native_benchmark/results.json`. The GitHub Pages
[Native vs LightGBM benchmark](https://javorraca.github.io/duckboost/benchmarks.html)
page renders that file.

## Design

- **Data:** `range(n)` with 4 low-cardinality categoricals + 8 continuous
  features; regression target with additive and interaction terms plus small
  noise; binary label from a related score threshold.
- **Split:** `hash(i, seed) % 5 = 0` → test (~20%), else train.
- **Grid:** shared MAP of `n_estimators`, `max_depth`, `learning_rate`,
  `min_samples_leaf`, `max_bins`, `reg_lambda`, `subsample`,
  `colsample_bytree`, `seed`, plus `categorical_features` for `cat_a`–`cat_d`.
- **Metrics:** holdout RMSE (regression) or logloss + accuracy (binary);
  train wall time is `epoch_ms(current_timestamp)` around `duckboost_train` only.

This is a smoke-level quality/speed check for the SQL-native path, not a claim
that native replaces LightGBM on every dataset.
