# Publishing duckboost as a community extension

duckboost targets the DuckDB **2.0** community track only. As of 2026-10-15,
[`duckdb/community-extensions`](https://github.com/duckdb/community-extensions)
builds new PRs from `repo.ref` against DuckDB commit
`b15358c371daf10dad799a43b6167f6ccdce84d8`, labeled and deployed as
`v2.0.0-alpha46057`. After the official `v2.0.0` release (planned 2026-10-21),
the DuckDB team redeploys under `v2.0.0`.

> The Discord prep note also mentioned `v2.0.0-alpha44357`. The **live** pin in
> `community-extensions` `.github/workflows/build.yml` is `v2.0.0-alpha46057` —
> that is what this repo matches.

## Checklist before opening the PR

1. **Public GitHub repo** — `JavOrraca/duckboost` (already).
2. **Build locally against the community DuckDB pin** (this repo’s `duckdb`
   submodule and `Makefile` defaults):

   ```bash
   git submodule update --init --recursive
   make
   make test
   ```

   Confirm configure logs something like:
   `version v2.0.0-alpha46057, extension folder v2.0.0-alpha46057`.

3. **Bump `repo.ref`** in
   [`docs/community_extensions_description.yml`](community_extensions_description.yml)
   to the **commit SHA** you want built (never a movable tag). Usually the merge
   commit of the release you are publishing.
4. Keep `extension.version` in sync with [`CHANGELOG.md`](../CHANGELOG.md) /
   the git tag.
5. Keep `docs.hello_world` aligned with
   [Getting Started](https://javorraca.github.io/duckboost/getting-started.html)
   (HTTPS CSV; no repo checkout required).
6. Do **not** add `variegata` (v1.5.6 last-minute track) or `ref_next`
   (prerelease leg retired for new PRs). duckboost is 2.0-only.

## Open the community-extensions PR

1. Fork [duckdb/community-extensions](https://github.com/duckdb/community-extensions).
2. Copy `docs/community_extensions_description.yml` to
   `extensions/duckboost/description.yml` (**that file only**).
3. Open a PR. Community CI builds and tests via
   `extension-ci-tools` `@v2.0-cyanoptera`.
4. After merge + deploy, users can:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

Docs: https://duckdb.org/community_extensions/documentation.html

## `duckdb_version` vs `duckdb_tag` (important)

Community CI passes:

| Input | Example | Role |
| --- | --- | --- |
| `duckdb_version` | `b15358c…` (commit) | Git checkout of DuckDB |
| `duckdb_tag` | `v2.0.0-alpha46057` | Deploy path + extension footer label |

DuckDB 2.0 stamps the engine label from `DUCKDB_VERSION` and no longer derives it
from git tags. Until
[duckdb/extension-ci-tools#439](https://github.com/duckdb/extension-ci-tools/pull/439)
merges into `v2.0-cyanoptera`, community builds can label the footer with the
**commit** instead of the tag, and `LOAD` against an alpha host then fails unless
`allow_extensions_metadata_mismatch` is set. This repo’s `Makefile` exports
`DUCKDB_VERSION` / `OVERRIDE_GIT_DESCRIBE` so **local** builds get the correct
footer today. Watch #439 for the community CI side.

## After official DuckDB 2.0.0

No descriptor change is required for the redeploy under `v2.0.0` — that is on the
DuckDB team. Optionally cut a duckboost release once clients ship 2.0.0 and update
docs that still say “alpha”.

See also [`PACKAGING.md`](../PACKAGING.md).
