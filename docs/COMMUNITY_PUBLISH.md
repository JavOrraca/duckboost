# Publishing duckboost as a community extension

duckboost targets DuckDB **2.0.0 LTS** (“Cinnamon Teal”) only. As of the
2026-10-21 release, [`duckdb/community-extensions`](https://github.com/duckdb/community-extensions)
builds new PRs from `repo.ref` against:

| Input | Value |
| --- | --- |
| `duckdb_version` | `v2.0.0` |
| `duckdb_commit` | `d54cbbe3df17e8e5ce68312f483b4155836112aa` |
| `ci_tools_version` | `v2.0-cyanoptera` |

Announcement: https://duckdb.org/2026/10/21/announcing-duckdb-200

## Checklist before opening the PR

1. **Public GitHub repo** — `JavOrraca/duckboost` (already).
2. **Cut a duckboost release** that builds cleanly on `v2.0.0` (recommended:
   `0.0.4` / tag `v0.0.4`) so `extension.version` matches what community ships.
3. **Build and test locally** against the pinned `duckdb` submodule:

   ```bash
   git submodule update --init --recursive
   make
   make test
   make check-extension-metadata
   ```

   Configure should log `version v2.0.0, extension folder v2.0.0`.
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
7. Do **not** add `variegata` (v1.5.x track) or `ref_next`. duckboost is 2.0-only.
8. Note: DuckDB no longer builds community extensions for `osx_amd64`
   (Intel Mac). Users on that platform need to build from source or use arm64.

## Open the community-extensions PR

1. Fork [duckdb/community-extensions](https://github.com/duckdb/community-extensions).
2. Copy `docs/community_extensions_description.yml` to
   `extensions/duckboost/description.yml` (**that file only** — no other files).
3. Open a PR. Community CI builds and tests via
   `extension-ci-tools` `@v2.0-cyanoptera`.
4. In the PR checks, confirm configure logs `version v2.0.0`.
5. After merge + deploy, users can:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

Official docs: https://duckdb.org/community_extensions/documentation.html

## Submodules in this repo

`duckdb` and `extension-ci-tools` stay as git submodules. That is the
[extension-template](https://github.com/duckdb/extension-template) layout:
local `make` / `make test` and the docs site build need them. Community CI does
**not** use your submodule pins — it checks out DuckDB from its own
`duckdb_version` / `duckdb_commit` and builds whatever commit `repo.ref` points
at in *this* repository. Keep the submodules in sync with the community pin so
local builds match what community ships.

## After clients catch up

Python / CLI DuckDB 2.0.0 may land before every client library (R, Java, Go,
etc.). The announcement notes some clients and extensions can take a few days.
Community `INSTALL` only requires a DuckDB 2.0 host that can load community
extensions.

See also [`PACKAGING.md`](../PACKAGING.md).
