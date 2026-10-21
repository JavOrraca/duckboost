# Packaging duckboost as a community extension

This repository is a standalone DuckDB extension in the
[extension-template](https://github.com/duckdb/extension-template) layout. It
targets DuckDB **2.0.0 LTS** (“Cinnamon Teal”):

| Knob | Value |
| --- | --- |
| `duckdb` submodule | `d54cbbe3df17e8e5ce68312f483b4155836112aa` (`v2.0.0`) |
| Community deploy version | `v2.0.0` |
| `extension-ci-tools` | `v2.0-cyanoptera` (`78bd1e4`) |

Optional ML vendor libraries stay out of the default community binary.

## Why the two git submodules stay

Do **not** remove `duckdb` or `extension-ci-tools`. The official extension
template uses both:

- `extension-ci-tools` provides the shared `Makefile` (`make` / `make test` /
  format / CI reusable workflows).
- `duckdb` is the engine the local shell and tests link against.

Community CI ignores these submodule SHAs and builds from its own DuckDB pin
plus your `repo.ref`. Keeping the submodules aligned with that pin is what makes
local builds match community binaries.

## Build

```bash
git submodule update --init --recursive
make
# or
GEN=ninja make
```

The root `Makefile` sets `DUCKDB_VERSION` / `OVERRIDE_GIT_DESCRIBE` to `v2.0.0`
and `DUCKDB_COMMIT` to the release SHA so extension footers match community
hosts.

Artifacts:

- Shell: `./build/release/duckdb`
- Tests: `make test`
- Loadable extension: `build/release/extension/duckboost/duckboost.duckdb_extension`
- Footer check: `make check-extension-metadata` (expects `v2.0.0` in the binary)

The shell, `libduckdb` and `make test` have duckboost statically linked
(`duckdb_extension_statically_link` in `extension_config.cmake`), so
`LOAD duckboost;` works there. To check the loadable file itself, load it from
an unsigned DuckDB 2.0.0 shell built from the same submodule commit **without**
duckboost linked:

```bash
/path/to/other/duckdb -unsigned
```

```sql
LOAD '<path>/build/release/extension/duckboost/duckboost.duckdb_extension';
```

## Community extensions submission

Descriptor: [`docs/community_extensions_description.yml`](docs/community_extensions_description.yml).
Full checklist: [`docs/COMMUNITY_PUBLISH.md`](docs/COMMUNITY_PUBLISH.md).

Summary:

1. Cut a release and set `repo.ref` to that commit SHA.
2. Copy that YAML into a fork as `extensions/duckboost/description.yml`.
3. Open a PR against [duckdb/community-extensions](https://github.com/duckdb/community-extensions).
4. After merge and CI, users install with:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

Docs: https://duckdb.org/community_extensions/documentation.html

### Descriptor fields (summary)

| Field | duckboost value |
| --- | --- |
| `extension.name` | `duckboost` |
| `extension.version` | latest release in [`CHANGELOG.md`](CHANGELOG.md) |
| `extension.license` | `MIT` |
| `extension.maintainers` | `JavOrraca` |
| `extension.requires_toolchains` | *(omit)* — default build is dependency-free CMake |
| `repo.github` | `JavOrraca/duckboost` |
| `repo.ref` | commit SHA to build (bump when submitting) |
| `repo.variegata` | *(omit)* — duckboost is 2.0-only |
| `repo.ref_next` | *(omit)* |
| `docs.hello_world` | Native first model on Palmer penguins via the public raw CSV URL |

Default community binaries intentionally omit vendor ML libraries: they ship the
native trainer + dump import. Optional `DUCKBOOST_WITH_*` native trainers are for
custom builds.

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

`EXT_FLAGS` is the extension-ci-tools makefile knob (passed into DuckDB's cmake).
Reconfigure after changing it (`rm -rf build/release` or delete `CMakeCache.txt`)
so cached `OFF` options are not reused.

```sql
SELECT * FROM duckboost_build_info();
SELECT * FROM duckboost_backends();
```

Native train tests (linked builds only):

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/xgboost/lib:$HOME/.local/lib/python3.12/site-packages/lightgbm/lib:${LD_LIBRARY_PATH}"
DUCKBOOST_NATIVE_TRAIN_TEST=1 make test T=test/sql/duckboost/native_train.test
```

## GitHub Pages (`site/`)

`.github/workflows/publish-docs.yml` builds DuckDB with duckboost, renders
`site/` from scratch with Quarto, and deploys to Pages on pushes to `main`.

```bash
make
python3 -m venv .venv-quarto
.venv-quarto/bin/pip install -r site/requirements.txt
QUARTO_PYTHON=$PWD/.venv-quarto/bin/python quarto render site
```

## Releases

Versions follow [semantic versioning](https://semver.org/). Git tags `vX.Y.Z`
are the source of truth. After DuckDB 2.0.0, duckboost GitHub releases can be
full (non-prerelease) releases once you are ready to publish to community.

Unreleased changes go under `## X.Y.Z (unreleased)` at the top of
[`CHANGELOG.md`](CHANGELOG.md). To cut a release:

1. In a PR, replace `(unreleased)` with the date, set `extension.version` in
   `docs/community_extensions_description.yml`, and merge.
2. Tag and publish:

   ```bash
   git switch main && git pull
   git tag -a v0.0.4 -m "duckboost 0.0.4"
   git push origin v0.0.4
   gh release create v0.0.4 --title "duckboost 0.0.4" --notes "See CHANGELOG.md"
   ```

3. Confirm:

   ```sql
   SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'duckboost';
   ```
