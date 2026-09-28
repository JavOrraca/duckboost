# Packaging duckboost as a community / out-of-tree extension

`duckboost` is prototyped in-tree under `extension/duckboost` for DuckDB 2.0 development. The intended long-term home is a **community (out-of-tree) extension** so optional ML vendor libraries do not weigh down core DuckDB builds.

## In-tree build (current)

```bash
DUCKDB_EXTENSIONS='duckboost' make reldebug
# or
EXTENSION_CONFIGS=.github/config/extensions/duckboost.cmake make reldebug
# or
BUILD_DUCKBOOST=1 make reldebug
```

## Out-of-tree style load from this directory

`extension_config.cmake` in this folder mirrors the [extension-template](https://github.com/duckdb/extension-template) pattern:

```bash
EXTENSION_CONFIGS=extension/duckboost/extension_config.cmake make reldebug
```

`CMakeLists.txt` is dual-mode: in-tree builds use `build_extension_library`; standalone / community builds use `build_static_extension` + `build_loadable_extension`.

## Extract a standalone repository

```bash
scripts/extract_duckboost_oot.sh                  # -> build/duckboost_oot
scripts/extract_duckboost_oot.sh /path/to/duckboost
```

The extract produces an [extension-template](https://github.com/duckdb/extension-template)-compatible tree:

```text
duckboost/
  CMakeLists.txt          # dual-mode
  Makefile                # wraps extension-ci-tools
  vcpkg.json
  extension_config.cmake
  duckboost_extension.cpp
  include/duckboost/...
  src/...
  test/sql/...
  .github/workflows/MainDistributionPipeline.yml
  docs/community_extensions_description.yml
  LICENSE
  README.md
  PACKAGING.md
```

Then:

```bash
cd build/duckboost_oot   # or your output dir
git init
git add .
git commit -m "Initial duckboost community extension extract"
# Push to a public GitHub repo, e.g. JavOrraca/duckboost
make                     # clones duckdb + extension-ci-tools and builds
```

## Submit to DuckDB community extensions

See [`community/SUBMIT.md`](community/SUBMIT.md) for the current commit SHA and
one-shot PR steps.

Publish surface (already pushed): orphan branch
`cursor/duckboost-community-oot-0c09` on `JavOrraca/duckdb-2-alpha-testing`
(extension-template layout at repo root — **do not merge** into `v2.0-cyanoptera`).

Community extensions are registered with a single YAML descriptor in
[duckdb/community-extensions](https://github.com/duckdb/community-extensions):

1. Copy [`community/description.yml`](community/description.yml) into a fork as
   `extensions/duckboost/description.yml` (that file only).
2. Open a PR against `duckdb/community-extensions`.
3. After merge and CI, users install with:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

Docs: https://duckdb.org/community_extensions/documentation.html

### Descriptor fields (summary)

| Field | duckboost value |
| --- | --- |
| `extension.name` | `duckboost` |
| `extension.version` | `0.1.0` (bump on release) |
| `extension.license` | `MIT` |
| `extension.maintainers` | `JavOrraca` |
| `repo.github` | `JavOrraca/duckdb-2-alpha-testing` |
| `repo.ref` | SHA of `cursor/duckboost-community-oot-0c09` (see `description.yml`) |

Default community binaries intentionally omit vendor ML libraries: they ship the
reference trainer + dump import. Optional `DUCKBOOST_WITH_*` native trainers are for
custom / advanced builds.

## Native trainer flags

| CMake option | Env hint | Effect |
| --- | --- | --- |
| `DUCKBOOST_WITH_XGBOOST=ON` | `XGBOOST_ROOT` | Link XGBoost C API train bridge |
| `DUCKBOOST_WITH_LIGHTGBM=ON` | `LIGHTGBM_ROOT` | Link LightGBM C API train bridge |
| `DUCKBOOST_WITH_CATBOOST=ON` | — | Compile-time capability flag only (no train C API) |
| `DUCKBOOST_NATIVE_STUB_ONLY=ON` | — | Compile `#ifdef` paths without linking vendor libs |

Example:

```bash
EXTRA_CMAKE_VARIABLES='-DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_WITH_LIGHTGBM=ON' \
  DUCKDB_EXTENSIONS='duckboost' make reldebug
```

Inspect the active build:

```sql
SELECT * FROM duckboost_build_info();
SELECT * FROM duckboost_backends();
```

Linked XGBoost/LightGBM builds set `training_supported=true` for those backends. CatBoost remains import-only.
