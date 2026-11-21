# Publishing duckboost as a community extension

duckboost targets DuckDB **2.0.1 LTS** (“Cinnamon Teal”) only. It is already listed at
https://duckdb.org/community_extensions/extensions/duckboost (merged via
[duckdb/community-extensions#3236](https://github.com/duckdb/community-extensions/pull/3236)).

The release candidate uses these pins. The [community build](https://github.com/duckdb/community-extensions/blob/main/.github/workflows/build.yml) still defaulted to 2.0.0 at catalog commit `f47566c7` on November 21, 2026.

| Input | Value |
| --- | --- |
| `duckdb_version` | `v2.0.1` |
| `duckdb_commit` | `8d78114e90a53823803660bf821f6073ac87da8c` |
| `ci_tools_version` | `43dae9ec2b843bf9c6200ea4bbe5ad703cd01f77` (v2.0-cyanoptera) |

Announcement: https://duckdb.org/2026/11/20/announcing-duckdb-201

Users install with:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

## Checklist before bumping / republishing

1. **Public GitHub repo** — `JavOrraca/duckboost` (already).
2. **Cut a duckboost release** that builds cleanly on `v2.0.1` so
   `extension.version` matches what community ships.
3. **Build and test locally** against the pinned `duckdb` submodule:

   ```bash
   git submodule update --init --recursive
   make
   make test
   make check-extension-metadata
   ```

   Configure should log `version v2.0.1, extension folder v2.0.1`.
   `make check-extension-metadata` asserts the same string is in the
   `.duckdb_extension` footer.
4. **Bump `repo.ref`** in
   [`docs/community_extensions_description.yml`](community_extensions_description.yml)
   to the **commit SHA** you want built (never a movable tag). Usually the
   release-tag commit.
5. Keep `extension.version` in sync with [`CHANGELOG.md`](../CHANGELOG.md) /
   the git tag.
6. Keep `docs.hello_world` aligned with
   [Getting Started](https://javorraca.github.io/duckboost/getting-started.html)
   (HTTPS CSV; no repo checkout required).
7. Keep
   [`docs/community_extensions_function_descriptions.csv`](community_extensions_function_descriptions.csv)
   in sync with the public SQL surface. Copy it to
   `extensions/duckboost/docs/function_descriptions.csv` in the community-
   extensions PR so the [Added Functions](https://duckdb.org/community_extensions/extensions/duckboost)
   table is not all `NULL` (community docs merge this CSV over
   `duckdb_functions()`).
8. Do **not** add `variegata` (v1.5.x track) or `ref_next`. duckboost is 2.0-only.
9. Note: DuckDB no longer builds community extensions for `osx_amd64`
   (Intel Mac). Users on that platform need to build from source or use arm64.

## Open / update the community-extensions PR

1. Work in your fork of [duckdb/community-extensions](https://github.com/duckdb/community-extensions)
   (for example [`JavOrraca/community-extensions`](https://github.com/JavOrraca/community-extensions)).
2. Copy `docs/community_extensions_description.yml` →
   `extensions/duckboost/description.yml`.
3. Copy `docs/community_extensions_function_descriptions.csv` →
   `extensions/duckboost/docs/function_descriptions.csv` (create the `docs/`
   directory if needed). This is the same pattern as the `h3` extension.
4. Open a PR. Community CI builds and tests via
   `extension-ci-tools` `@v2.0-cyanoptera`.
5. In the PR checks, record the engine and tools actually selected by upstream. A successful 2.0.0 job verifies that engine only.

Official docs: https://duckdb.org/community_extensions/documentation.html

## Submodules in this repo

`duckdb` and `extension-ci-tools` stay as git submodules. That is the
[extension-template](https://github.com/duckdb/extension-template) layout:
local `make` / `make test` and the docs site build need them. Community CI does
**not** use your submodule pins — it checks out DuckDB from its own
`duckdb_version` / `duckdb_commit` and builds whatever commit `repo.ref` points
at in *this* repository. Keep the submodules in sync with the community pin so
local builds match what community ships.

After upstream deployment, use a fresh extension directory in a stock 2.0.1 shell. Run the install and load above, train and predict, and verify the engine and extension independently.

```sql
SELECT version() AS engine, extension_version
FROM duckdb_extensions() WHERE extension_name = 'duckboost' AND loaded;
```

See also [the packaging guide](../PACKAGING.md).
