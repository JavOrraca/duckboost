# Changelog

duckboost uses [semantic versioning](https://semver.org/). Releases are git tags (`v0.0.1`, `v0.0.2`, ...) with matching [GitHub releases](https://github.com/JavOrraca/duckboost/releases). Targets DuckDB **2.0.1 LTS** (“Cinnamon Teal”).

A build from a release tag reports that version, and any other commit reports its short commit hash:

```sql
SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'duckboost';
```

## 0.1.0 (2026-11-21)

### Saved models and SQL scoring

- Fit a named STRUCT with `duckboost_train_raw` to save numeric inputs, optional training medians, string dictionaries and trees together. `duckboost_predict_raw` matches fields by name when scoring reordered rows. Missing and unseen categories take the model’s missing branch. `duckboost_to_sql_raw` includes the fitted mapping in the exported query. [Raw-column example](https://javorraca.github.io/duckboost/vignettes/split-and-preprocess.html)
- `duckboost_calibrate_interval` packages three quantile fits with a correction learned on a separate calibration fold. `duckboost_predict_interval` returns lower, median and upper fields, and `duckboost_interval_to_sql` exports all three. Predictions are sorted before calibration and scoring. At 90% target coverage the exact correction for scores 1 through 10 is 10. On 300 untouched vignette rows, the calibrated interval covered 264 (88.0%) with mean width 1.4968. Insufficient calibration rows raise an error. [Calibration and coverage example](https://javorraca.github.io/duckboost/vignettes/prediction-intervals.html)
- Raw and interval artifacts use version 3 JSON. Existing positional `DOUBLE[]` APIs still read v1/v2 models and write the lowest ensemble format they need. Use the raw or interval functions for their respective artifacts.

### Training and export

- Balanced class weights now use the sample-weight totals of training rows. Validation loss reuses the fitted multipliers. In the balanced binary regression fixture, changing 100 validation labels used to move the fixed-round probability between 0.25 and 0.75. Both fits now start at a probability of 0.5 (log-odds zero). Early stopping can still select a different iteration.
- Every SQL tree split emits its left and right children once. Compact binary probability exports reuse the score across all outputs. In a complete depth-eight tree, leaf literal occurrences fell from 6,561 to 256. Compact SQL shrank from 292,757 to 19,843 bytes. Median prepare time on an Apple M2 fell from 254 to 21 ms, and scoring 100,000 rows fell from 188 to 29 ms (three runs, one thread).
- Numeric cuts now follow training observation ranks while retaining all midpoints on low-cardinality features. In the skew fixture the first split moved from 3.5 to 0.5, separating its 10,000 zeros from its 10,000 ones. Numeric bins use packed columns. Sibling histograms use parent subtraction and large feature scans use DuckDB’s thread budget with deterministic tie-breaking.

### DuckDB 2.0.1 compatibility

These fixes ship in [DuckDB 2.0.1 LTS](https://github.com/duckdb/duckdb/releases/tag/v2.0.1). Both sides of the duckboost benchmark use this engine.

- DuckDB retains qualifying NULL rows when filtering some data written by DuckDB 1.x. This preserves observations entering training or scoring. [DuckDB #28355](https://github.com/duckdb/duckdb/pull/28355)
- DuckDB corrected Parquet `IS NULL` pruning for pages with constant string statistics. Missing categories can reach preprocessing correctly. [DuckDB #27514](https://github.com/duckdb/duckdb/pull/27514)
- Window functions now propagate certain build failures under memory pressure, preventing a hang in queries such as window-based split and fold assignments. [DuckDB #27596](https://github.com/duckdb/duckdb/pull/27596)
- DuckDB reduced repeated memory pressure in grouped aggregates with non-spillable states, relevant to surrounding `quantile_cont` or `arg_max` analytics. Interval calibration uses an exact finite-sample rank. [DuckDB #28646](https://github.com/duckdb/duckdb/pull/28646)

## 0.0.5 (2026-10-24)

### Fixed

- `duckboost_to_sql` now writes leaf values, thresholds, and the learning rate as `DOUBLE` literals (`0.1e0` instead of `0.10000000000000001`). DuckDB 2.0 reads long decimal literals as `DECIMAL`, so a 200-tree regression export in v0.0.4 could fail with `DECIMAL(38,37)` overflow. Exported SQL now matches `duckboost_predict` without a `regexp_replace` workaround.

### Documentation

- Getting Started / README now lead with `INSTALL duckboost FROM community` (live after [community-extensions#3236](https://github.com/duckdb/community-extensions/pull/3236)).
- Added `docs/community_extensions_function_descriptions.csv` so the community “Added Functions” table can be populated.

## 0.0.4 (2026-10-21)

### Changed

- Pinned `duckdb` to the official `v2.0.0` release (`d54cbbe`) and `extension-ci-tools` to current `v2.0-cyanoptera` (`78bd1e4`). Local CI / `Makefile` label builds as `v2.0.0` to match community-extensions.
- Kept the `duckdb` and `extension-ci-tools` git submodules (required by the extension-template layout for local `make`); documented why in `PACKAGING.md`.
- Added `make check-extension-metadata` to assert the loadable extension footer carries the expected DuckDB version before a community submission.
- Updated README, site, and publish docs for DuckDB 2.0.0 GA; community submission checklist is in `docs/COMMUNITY_PUBLISH.md`.

## 0.0.3 (2026-10-05)

### Added

- Classification evaluate metrics `brier_score` (alias `brier`), `roc_auc_ovr` (alias `auc_ovr`), and `roc_auc_ovo` (alias `auc_ovo`) on `duckboost_evaluate_agg`. Binary and multiclass are both supported. Scalar `duckboost_evaluate` returns the per-row Brier contribution; ROC AUC is aggregate-only.

## 0.0.2 (2026-10-04)

### Fixed

- `LOAD duckboost` works again in `./build/release/duckdb` on a fresh clone. Since the DuckDB 2.0 alpha bump on 2026-09-30, the build compiled duckboost but no longer linked it into the shell, so `LOAD duckboost` failed with `Extension ".../duckboost.duckdb_extension" not found`. duckboost is now statically linked into the shell, `libduckdb` and `make test` ([#34](https://github.com/JavOrraca/duckboost/pull/34)).

### Changed

- Bumped the `duckdb` submodule to `80e17fc` (`v2.0-cyanoptera`) and `extension-ci-tools` to `969bc76`. Macro registration follows DuckDB's `ParserOptions` API change ([#35](https://github.com/JavOrraca/duckboost/pull/35)).

### Documentation

- The site is re-rendered from scratch on every publish. The docs workflow builds DuckDB with duckboost and runs every example, so the published output matches `main`. Committed `site/_freeze` output is gone.
- New Changelog page on the site, and versioned releases: git tags `vX.Y.Z` with GitHub pre-releases, starting from `v0.0.1`.
- Clarified that the `.duckdb_extension` file is for a separate DuckDB 2.0 shell built from the same `duckdb` commit. This repo's shell already has duckboost linked in.

## 0.0.1 (2026-10-02)

Soft launch: the first public pre-release, built against the DuckDB 2.0 alpha (`v2.0-cyanoptera`). Try it by cloning and building this repo. PyPI and CRAN DuckDB 1.x cannot load it, and the community extension store submission waits for DuckDB 2.0.

- Native histogram GBDT trainer with no external dependencies (`backend = 'native'`, alias `reference`): regression and classification, quantile / MAE / expectile objectives, categorical features, leaf-wise and oblivious tree growth, sample weights and early stopping on a validation set.
- Training and scoring in SQL: `duckboost_train`, `duckboost_fit`, `duckboost_predict`, `duckboost_predict_proba`, `duckboost_score`, `duckboost_evaluate`, `duckboost_evaluate_agg` and `duckboost_importance`.
- `duckboost_to_sql` exports a model as plain DuckDB SQL (`CASE WHEN` ensembles), so scoring needs no extension.
- `duckboost_import` reads XGBoost, LightGBM and CatBoost model dumps.
- Split and preprocessing macros: `duckboost_initial_split`, `duckboost_initial_validation_split`, `duckboost_vfold`, `duckboost_dummy`, `duckboost_levels`, `duckboost_other` and more.
- Optional in-process XGBoost / LightGBM training behind `DUCKBOOST_WITH_*` build flags, and `duckboost_build_info()` / `duckboost_backends()` to inspect a build.
- Documentation site with Getting Started, vignettes, a native vs LightGBM benchmark and a SQL reference.

**Known issue:** on a fresh clone, `LOAD duckboost` fails in `./build/release/duckdb`. Load the file by path instead, `LOAD 'build/release/extension/duckboost/duckboost.duckdb_extension';`, or update to 0.0.2.
