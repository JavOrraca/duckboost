# Packaging duckboost as a community extension

This repository is a standalone DuckDB extension in the [extension-template](https://github.com/duckdb/extension-template) layout. It targets DuckDB 2.0 through the `duckdb` submodule pinned to `v2.0-cyanoptera`. Optional ML vendor libraries stay out of the default community binary.

## Build

```bash
git submodule update --init --recursive
make
# or
GEN=ninja make
```

Artifacts:

- Shell: `./build/release/duckdb`
- Tests: `make test`
- Loadable extension: `build/release/extension/duckboost/duckboost.duckdb_extension`

The shell, `libduckdb` and `make test` have duckboost statically linked (`duckdb_extension_statically_link` in `extension_config.cmake`), so `LOAD duckboost;` works there, and a path `LOAD` there uses the built-in copy rather than the file. To check the loadable file itself, load it from an unsigned DuckDB 2.0 shell built from the same `duckdb` submodule commit without duckboost linked:

```bash
/path/to/other/duckdb -unsigned
```

```sql
LOAD '<path>/build/release/extension/duckboost/duckboost.duckdb_extension';
```

## Community extensions submission

Submission is on hold until DuckDB 2.0 is released.

The descriptor is [`docs/community_extensions_description.yml`](docs/community_extensions_description.yml). Community extensions are registered with that YAML in [duckdb/community-extensions](https://github.com/duckdb/community-extensions):

1. After DuckDB 2.0 is released, copy `docs/community_extensions_description.yml` into a fork as `extensions/duckboost/description.yml` (that file only).
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
| `extension.requires_toolchains` | *(omit)* — default build is dependency-free CMake |
| `repo.github` | `JavOrraca/duckboost` |
| `repo.ref` | commit SHA to build (see `docs/community_extensions_description.yml`; bump when submitting) |
| `docs.hello_world` | Native first model on Palmer penguins via the public raw CSV URL |

Default community binaries intentionally omit vendor ML libraries: they ship the
native trainer + dump import. Optional `DUCKBOOST_WITH_*` native trainers are for
custom builds.

Before opening the community-extensions PR, set `repo.ref` to the commit you want
built, and keep `docs.hello_world` aligned with [Getting Started](https://javorraca.github.io/duckboost/getting-started.html). The catalog snippet loads the CSV over HTTPS (`httpfs` autoload on released DuckDB).

## Native trainer flags

| CMake option | Env hint | Effect |
| --- | --- | --- |
| `DUCKBOOST_WITH_XGBOOST=ON` | `XGBOOST_ROOT` | Link XGBoost C API train bridge |
| `DUCKBOOST_WITH_LIGHTGBM=ON` | `LIGHTGBM_ROOT` | Link LightGBM C API train bridge |
| `DUCKBOOST_WITH_CATBOOST=ON` | — | Compile-time capability flag only (no train C API) |
| `DUCKBOOST_NATIVE_STUB_ONLY=ON` | — | Compile `#ifdef` paths without linking vendor libs |

Example:

```bash
EXT_FLAGS='-DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_WITH_LIGHTGBM=ON' make release
```

Compile the native `#ifdef` paths without linking vendor libraries:

```bash
EXT_FLAGS='-DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_NATIVE_STUB_ONLY=ON' make release
```

`EXT_FLAGS` is the extension-ci-tools makefile knob (passed into DuckDB's cmake). Reconfigure after changing it (`rm -rf build/release` or delete `CMakeCache.txt`) so cached `OFF` options are not reused.

Inspect the active build:

```sql
SELECT * FROM duckboost_build_info();
SELECT * FROM duckboost_backends();
```

Linked XGBoost/LightGBM builds set `training_supported=true` for those backends. CatBoost remains import-only.

Native train tests (linked builds only). The **Native trainers** GitHub Actions workflow installs the pinned wheels from `scripts/vendor_parity/requirements.txt`, builds with `DUCKBOOST_WITH_XGBOOST/LIGHTGBM=ON`, and runs:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/xgboost/lib:$HOME/.local/lib/python3.12/site-packages/lightgbm/lib:${LD_LIBRARY_PATH}"
DUCKBOOST_NATIVE_TRAIN_TEST=1 make test T=test/sql/duckboost/native_train.test
```

Vendor parity fixture regeneration also runs on PRs that touch `src/import.cpp` / `src/model.cpp` (not only the generator or committed probes).

## GitHub Pages (`site/`)

`.github/workflows/publish-docs.yml` renders `site/` with Quarto and deploys to Pages. It installs Jupyter but does **not** build DuckDB. Executable examples are stored in `site/_freeze` (`execute.freeze: auto`). Editing any `.qmd` that contains `{python}` cells changes that page's freeze hash and forces re-execution in CI — pages that call `site/_helpers/duckrun.py` then fail unless freeze is refreshed.

After changing those pages:

```bash
make
python3 -m venv .venv-quarto
.venv-quarto/bin/pip install jupyter nbformat nbclient ipykernel
QUARTO_PYTHON=$PWD/.venv-quarto/bin/python quarto render site
python3 scripts/check_site_freeze.py   # also runs in Publish docs CI
git add site/_freeze && git commit
```
