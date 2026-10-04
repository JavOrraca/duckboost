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
| `extension.version` | latest release in [`CHANGELOG.md`](CHANGELOG.md) (set when cutting a release) |
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

`.github/workflows/publish-docs.yml` builds DuckDB with duckboost (ccache keeps rebuilds short), renders `site/` from scratch with Quarto, and deploys to Pages on pushes to `main`. Pull requests that touch the site, `CHANGELOG.md` or the extension sources render the site too, which checks that every example still runs, but they don't deploy. Nothing is frozen: the pages run their SQL through `./build/release/duckdb` on every render, so there is no `site/_freeze` to refresh.

Preview locally:

```bash
make
python3 -m venv .venv-quarto
.venv-quarto/bin/pip install -r site/requirements.txt
# Only the ticket NLP vignette needs these, and only to rebuild .cache/tickets:
.venv-quarto/bin/pip install -r site/requirements-embeddings.txt
QUARTO_PYTHON=$PWD/.venv-quarto/bin/python quarto render site   # or: quarto preview site
```

## Releases

Versions follow [semantic versioning](https://semver.org/), and git tags `vX.Y.Z` are the source of truth. DuckDB's build reports `vX.Y.Z` as `extension_version` for a build of the tagged commit and the short commit hash otherwise. Until DuckDB 2.0 ships, releases are GitHub pre-releases.

Unreleased changes go under `## X.Y.Z (unreleased)` at the top of [`CHANGELOG.md`](CHANGELOG.md), in the same PR as the change. To cut a release:

1. In a PR, replace `(unreleased)` with the date, `## 0.0.2 (YYYY-MM-DD)`, set `extension.version` in `docs/community_extensions_description.yml`, and merge.
2. Tag the merge commit and publish the release:

   ```bash
   git switch main && git pull
   git tag -a v0.0.2 -m "duckboost 0.0.2"
   git push origin v0.0.2
   gh release create v0.0.2 --prerelease --title "duckboost 0.0.2" --notes "See CHANGELOG.md"
   ```

3. Check that a build of the tag reports the version:

   ```sql
   SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'duckboost';
   ```

4. Start the next `## X.Y.Z (unreleased)` section when the next change lands.
