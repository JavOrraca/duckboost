# Native trainer benchmarks

The runner fits regression and binary models on a synthetic 12-feature table, evaluates untouched holdouts, and records training time and peak process memory. The optional wide workload uses 32 numeric features dominated by repeated zeros and ones. The release measurements and an earlier comparison with linked LightGBM appear on the [benchmark page](https://javorraca.github.io/duckboost/benchmarks.html).

## Native release comparison

Build the release, then run from the repository root. Use a distinct output for every revision and thread budget.

```bash
make release
python3 scripts/native_benchmark/run.py --n-rows 200000 --seed 7 --skip-lightgbm \
  --threads 4 --repeats 3 --output standard-4.json
python3 scripts/native_benchmark/run.py --wide --n-rows 50000 --seed 7 --skip-lightgbm \
  --threads 4 --repeats 3 --output wide-4.json
```

Repeat at 1 and 8 threads. Supply `--cli /path/to/duckdb` and `--extension /path/to/duckboost.duckdb_extension` to compare multiple builds on the same engine. The CLI and extension must target the same DuckDB release.

Each repeat starts a fresh CLI process. Model fitting uses 80 depth-5 trees, 64 numeric cutpoints, learning rate 0.1, row and column sampling of 0.8, and minimum leaf size 20. The standard workload marks the four low-cardinality codes as categorical. Its regression target combines additive terms, a categorical interaction and noise. Wide regression depends on three indicators of positive values and a noise term. The split is `hash(i, seed) % 5 = 0` for test rows.

The timer surrounds the training statement, including row ordering, binning and serialization. Data preparation and evaluation run outside the timer. The runner reports the median elapsed time and the highest whole-process peak RSS across repeats. Peak RSS is reported in MiB on macOS and Linux and as null when the platform lacks per-child resource measurements. Holdout metrics include RMSE for regression and log loss and accuracy for binary classification. The release JSON contains the build settings and the results for each revision and thread budget. [0.1.0 measurements](results-v0.1.0.json)

## Linked LightGBM

```bash
EXT_FLAGS='-DDUCKBOOST_WITH_LIGHTGBM=ON' make release
python3 scripts/native_benchmark/run.py --n-rows 200000 --seed 7 --output linked-results.json
```

The linked build requires the LightGBM wheel or C library. On Linux, the library directory must be on the loader's search path. Without `--output`, the runner writes `scripts/native_benchmark/results.json`, the file used for the historical comparison.
