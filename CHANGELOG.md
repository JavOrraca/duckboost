# Changelog

duckboost uses [semantic versioning](https://semver.org/). Releases are git tags (`v0.0.1`, `v0.0.2`, ...) with matching [GitHub releases](https://github.com/JavOrraca/duckboost/releases). Until DuckDB 2.0 ships, every release is a pre-release against the DuckDB 2.0 alpha pinned in the `duckdb` submodule.

A build from a release tag reports that version, and any other commit reports its short commit hash:

```sql
SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'duckboost';
```

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
