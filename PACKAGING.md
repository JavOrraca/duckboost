# Packaging duckboost as a community extension

This repository is a standalone DuckDB extension in the
[extension-template](https://github.com/duckdb/extension-template) layout. It
targets the DuckDB **2.0 community alpha** pin:

| Knob | Value |
| --- | --- |
| `duckdb` submodule | `b15358c371daf10dad799a43b6167f6ccdce84d8` |
| Community deploy tag | `v2.0.0-alpha46057` |
| `extension-ci-tools` | `v2.0-cyanoptera` (`969bc76`) |

Optional ML vendor libraries stay out of the default community binary.

> Discord prep text mentioned `v2.0.0-alpha44357`. Live
> `duckdb/community-extensions` `build.yml` uses `v2.0.0-alpha46057` — follow that.

## Build

```bash
git submodule update --init --recursive
make
# or
GEN=ninja make
```

The root `Makefile` sets `DUCKDB_VERSION` / `OVERRIDE_GIT_DESCRIBE` to
`v2.0.0-alpha46057` and `DUCKDB_COMMIT` to the community commit so extension
footers match what community hosts expect (DuckDB 2.0 no longer derives the label
from git tags; see [extension-ci-tools#439](https://github.com/duckdb/extension-ci-tools/pull/439)).

Artifacts:

- Shell: `./build/release/duckdb`
- Tests: `make test`
- Loadable extension: `build/release/extension/duckboost/duckboost.duckdb_extension`
- Footer check: `make check-extension-metadata` (expects `v2.0.0-alpha46057` in the binary)

The shell, `libduckdb` and `make test` have duckboost statically linked
(`duckdb_extension_statically_link` in `extension_config.cmake`), so
`LOAD duckboost;` works there, and a path `LOAD` there uses the built-in copy
rather than the file. To check the loadable file itself, load it from an unsigned
DuckDB 2.0 shell built from the same `duckdb` submodule commit **without**
duckboost linked, and with a matching engine label:

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

1. Set `repo.ref` to the commit SHA to build.
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
| `extension.version` | latest release in [`CHANGELOG.md`](CHANGELOG.md) (set when cutting a release) |
| `extension.license` | `MIT` |
| `extension.maintainers` | `JavOrraca` |
| `extension.requires_toolchains` | *(omit)* — default build is dependency-free CMake |
| `repo.github` | `JavOrraca/duckboost` |
| `repo.ref` | commit SHA to build (bump when submitting) |
| `repo.variegata` | *(omit)* — 1.5.6 last-minute track; duckboost is 2.0-only |
| `repo.ref_next` | *(omit)* — prerelease leg no longer builds for new PRs |
| `docs.hello_world` | Native first model on Palmer penguins via the public raw CSV URL |

Default community binaries intentionally omit vendor ML libraries: they ship the
native trainer + dump import. Optional `DUCKBOOST_WITH_*` native trainers are for
custom builds.

Before opening the community-extensions PR, set `repo.ref` to the commit you want
built, and keep `docs.hello_world` aligned with
[Getting Started](https://javorraca.github.io/duckboost/getting-started.html).
The catalog snippet loads the CSV over HTTPS (`httpfs` autoload on released DuckDB).

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

`EXT_FLAGS` is the extension-ci-tools makefile knob (passed into DuckDB's cmake).
Reconfigure after changing it (`rm -rf build/release` or delete `CMakeCache.txt`)
so cached `OFF` options are not reused.

Inspect the active build:

```sql
SELECT * FROM duckboost_build_info();
SELECT * FROM duckboost_backends();
```

Linked XGBoost/LightGBM builds set `training_supported=true` for those backends.
CatBoost remains import-only.

Native train tests (linked builds only). The **Native trainers** GitHub Actions
workflow installs the pinned wheels from `scripts/vendor_parity/requirements.txt`,
builds with `DUCKBOOST_WITH_XGBOOST/LIGHTGBM=ON`, and runs:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/xgboost/lib:$HOME/.local/lib/python3.12/site-packages/lightgbm/lib:${LD_LIBRARY_PATH}"
DUCKBOOST_NATIVE_TRAIN_TEST=1 make test T=test/sql/duckboost/native_train.test
```

Vendor parity fixture regeneration also runs on PRs that touch `src/import.cpp` /
`src/model.cpp` (not only the generator or committed probes).

## GitHub Pages (`site/`)

`.github/workflows/publish-docs.yml` builds DuckDB with duckboost (ccache keeps
rebuilds short), renders `site/` from scratch with Quarto, and deploys to Pages
on pushes to `main`. Pull requests that touch the site, `CHANGELOG.md` or the
extension sources render the site too, which checks that every example still
runs, but they don't deploy. Nothing is frozen: the pages run their SQL through
`./build/release/duckdb` on every render, so there is no `site/_freeze` to refresh.

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

Versions follow [semantic versioning](https://semver.org/), and git tags `vX.Y.Z`
are the source of truth. DuckDB's build reports `vX.Y.Z` as `extension_version`
for a build of the tagged commit and the short commit hash otherwise.

Until clients ship DuckDB 2.0.0, GitHub releases stay pre-releases. Community
binaries for the alpha pin can still ship via `INSTALL … FROM community` once the
descriptor PR merges.

Unreleased changes go under `## X.Y.Z (unreleased)` at the top of
[`CHANGELOG.md`](CHANGELOG.md), in the same PR as the change. To cut a release:

1. In a PR, replace `(unreleased)` with the date, `## 0.0.2 (YYYY-MM-DD)`, set
   `extension.version` in `docs/community_extensions_description.yml`, and merge.
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
