# duckboost

Experimental DuckDB extension for **in-database gradient boosting**: train and evaluate tree ensembles inside DuckDB, then export pure SQL for orbital-style in-database inference.

**Documentation:** <https://javorraca.github.io/duckboost/>

Standalone [extension-template](https://github.com/duckdb/extension-template) repository targeting DuckDB 2.0. The `duckdb` submodule is pinned to `v2.0-cyanoptera`.

## Motivation

[Orbital](https://posit-dev.github.io/orbital/) converts trained sklearn / tidymodels pipelines into SQL so scoring needs no Python runtime. `duckboost` brings a similar loop fully into DuckDB:

1. **Train** with a selectable boosting backend
2. **Evaluate** fit quality in SQL
3. **Export** model trees to DuckDB SQL (`CASE WHEN` ensembles, optional `separate_trees`)
4. **Score** with either `duckboost_predict` or the exported SQL (no extension required at inference time)

## Build

Initialize the submodules, then build the release shell and loadable extension:

```bash
git submodule update --init --recursive
make
# or
GEN=ninja make
```

The shell is `./build/release/duckdb`. Tests:

```bash
make test
```

Load a local unsigned build:

```bash
./build/release/duckdb -unsigned
```

```sql
LOAD '<path>/build/release/extension/duckboost/duckboost.duckdb_extension';
```

Optional native trainer flags (XGBoost / LightGBM train via vendor C API → dump → import):

```bash
# Link real libraries (pip wheels work; set ROOT or rely on auto-detect under ~/.local)
EXTRA_CMAKE_VARIABLES='-DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_WITH_LIGHTGBM=ON' make

# Compile #ifdef paths without linking vendor libraries
EXTRA_CMAKE_VARIABLES='-DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_NATIVE_STUB_ONLY=ON' make
```

CatBoost has no public in-process training C API — use `duckboost_import('catboost', ...)`.

Inspect the active build:

```sql
SELECT * FROM duckboost_build_info();
SELECT * FROM duckboost_backends();
```

Native train tests (linked builds only):

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/xgboost/lib:$HOME/.local/lib/python3.12/site-packages/lightgbm/lib:${LD_LIBRARY_PATH}"
DUCKBOOST_NATIVE_TRAIN_TEST=1 make test T=test/sql/duckboost/native_train.test
```

See [`PACKAGING.md`](PACKAGING.md) for the community-extension descriptor and native-trainer flags.

## SQL API

```sql
LOAD duckboost;

-- Available backends
SELECT * FROM duckboost_backends();

-- Train (reference GBDT backend)
CREATE TABLE models AS
SELECT duckboost_train(
	y,
	[x1, x2],
	MAP {
		'backend': 'reference',
		'task': 'regression',
		'n_estimators': '20',
		'max_depth': '3',
		'learning_rate': '0.1',
		'feature_names': 'x1,x2'
	}
) AS model
FROM train;

-- In-process predict
SELECT duckboost_predict(model, [x1, x2]) FROM models, test;

-- Dataset-level metrics
SELECT duckboost_evaluate_agg(model, y, [x1, x2], MAP {'metric': 'rmse'})
FROM models, test;

-- Orbital-style SQL export (trees as separate columns for DuckDB parallelism)
SELECT duckboost_to_sql(
	model,
	'test',
	['x1', 'x2'],
	MAP {'separate_trees': 'true', 'prediction_alias': 'pred'}
) FROM models;

-- Import a vendor dump trained outside DuckDB
SELECT duckboost_import('xgboost', xgb_dump_json, MAP {'task': 'binary', 'base_score': '0.0'});
SELECT duckboost_import('lightgbm', lgb_model_txt);
SELECT duckboost_import('catboost', catboost_model_json);

-- Table macros (query_table wrappers; pass table names as VARCHAR literals)
CREATE TABLE models AS
FROM duckboost_fit('train', y, [x1, x2], options := MAP {'n_estimators': '20', 'feature_names': 'x1,x2'});

SELECT * FROM duckboost_score((SELECT model FROM models), 'test', [x1, x2]);

-- Multiclass: labels are class indices 0 .. n_classes - 1 (n_classes inferred, or set 'n_classes')
CREATE TABLE mc_models AS
SELECT duckboost_train(label, features, MAP {'task': 'multiclass', 'n_estimators': '20'}) AS model
FROM mc_train;

-- predict returns argmax class index; predict_proba returns softmax LIST
SELECT duckboost_predict(model, features), duckboost_predict_proba(model, features) FROM mc_models, mc_test;
```

### Example datasets

[`data/`](data/) ships two small datasets to learn with. Run from the repository root:

```sql
LOAD duckboost;
CREATE TABLE iris AS FROM read_csv('data/iris.csv');                          -- 150 rows
CREATE TABLE penguins AS FROM read_csv('data/penguins.csv', nullstr = 'NA');  -- 344 rows

-- Binary: is a (non-setosa) iris virginica?
CREATE TABLE iris_binary AS
SELECT (species = 'virginica')::DOUBLE AS y,
       [sepal_length, sepal_width, petal_length, petal_width] AS features
FROM iris WHERE species <> 'setosa';

SELECT duckboost_evaluate_agg(model, y, features, MAP {'metric': 'accuracy'})
FROM (SELECT duckboost_train(y, features, MAP {'task': 'binary'}) AS model FROM iris_binary), iris_binary;
```

The [example datasets vignette](https://javorraca.github.io/duckboost/vignettes/datasets.html) walks through train/validation/test splits, one-hot encoding, regression, binary classification, and a SQL grid search on both files.

### Backends

| Backend | Train in this build | Dump import | Predict / evaluate / `to_sql` |
| --- | --- | --- | --- |
| `reference` | Yes (built-in GBDT) | duckboost JSON | Yes |
| `xgboost` | Optional (`DUCKBOOST_WITH_XGBOOST`) | `dump_model(..., dump_format='json')` | Yes |
| `lightgbm` | Optional (`DUCKBOOST_WITH_LIGHTGBM`) | `booster_.save_model()` text | Yes |
| `catboost` | Optional (`DUCKBOOST_WITH_CATBOOST`) | `save_model(..., format='json')` float trees | Yes |

Native XGBoost / LightGBM linking is opt-in via `DUCKBOOST_WITH_*`. Linked builds train in-process through the vendor C API, then convert the dump into duckboost JSON. CatBoost remains import-only. Prefer `duckboost_import()` when you already train outside DuckDB; use `backend='reference'` for dependency-free experiments. [Use cases and choosing a trainer](https://javorraca.github.io/duckboost/use-cases.html) compares the three paths in detail.

### Dump import notes

- Imported models set `learning_rate = 1.0` and bake vendor shrinkage/scale into leaf values (XGBoost dump leaves already include η; LightGBM `shrinkage` and CatBoost `scale_and_bias` are applied at import).
- LightGBM numerical `<=` splits are converted to duckboost `<` via `nextafter(threshold, +∞)`.
- CatBoost support: `FloatFeature`, `OneHotFeature`, and `OnlineCtr` (Counter/Borders). CTR combinations may include `cat_feature_value`, `float_feature`, and `cat_feature_exact_value`. Pass categorical CityHash values as numeric features.
- Multiclass CatBoost JSON uses class-blocked `leaf_values` (`2^depth` values per class). Import expands each oblivious tree into `n_classes` duckboost trees (layout `[round][class]`).
- OnlineCtr requires `ctr_data` in the dump (`save_model(..., pool=...)`). `duckboost_to_sql` inlines CTR hash lookups via `UHUGEINT` modular arithmetic.
- Optional import map keys: `task`, `base_score`, `learning_rate`, `feature_names`, `n_classes`.

## Model format

Models are opaque `VARCHAR` JSON documents:

```json
{
  "duckboost_version": 1,
  "backend": "reference",
  "task": "regression",
  "base_score": 0.0,
  "learning_rate": 0.1,
  "n_features": 2,
  "n_classes": 1,
  "feature_names": ["x1", "x2"],
  "trees": [{"nodes": [{"is_leaf": false, "feature": 0, "threshold": 1.5, "left": 1, "right": 2, "value": 0.0}, ...]}]
}
```

For `task: "multiclass"`, `n_classes >= 2`, optional `base_scores` holds per-class bias, and trees are stored as `round * n_classes + class`. `duckboost_predict` returns the argmax class index; `duckboost_to_sql` exports an argmax over per-class score expressions.

## Design notes

- **Layout**: standalone community extension ([extension-template](https://github.com/duckdb/extension-template)) so optional vendor ML libraries stay out of core DuckDB builds. The `duckdb` submodule tracks DuckDB 2.0 (`v2.0-cyanoptera`).
- **SQL export** mirrors orbital's `separate_trees` idea so DuckDB can evaluate ensemble members as independent columns.
- **Reference trainer** is a didactic histogram/quantile-split GBDT (squared error, logistic, and softmax for multiclass). It is not a replacement for production XGBoost/LightGBM/CatBoost quality, but it exercises the full train → evaluate → SQL path.
- **Table macros** `duckboost_fit` / `duckboost_score` wrap `duckboost_train` / `duckboost_predict` with `query_table` for a compact SQL workflow.

## Roadmap

- [x] Native trainer scaffolding behind `DUCKBOOST_WITH_*` / `DUCKBOOST_NATIVE_STUB_ONLY` + `duckboost_build_info()`
- [x] Vendor C API bridges for XGBoost / LightGBM (train → dump → import into `BoostModel`)
- [x] Standalone extension-template repository (`Makefile`, `extension_config.cmake`, [`docs/community_extensions_description.yml`](docs/community_extensions_description.yml))
- [ ] Submit [`docs/community_extensions_description.yml`](docs/community_extensions_description.yml) to `duckdb/community-extensions` after DuckDB 2.0 is released
