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

## Gate: wait for extension-ci-tools#439 (or equivalent)

Community CI passes a **commit** as `duckdb_version` and `v2.0.0-alpha46057` as
`duckdb_tag`. DuckDB 2.0 stamps the engine label from `DUCKDB_VERSION`, so until
[extension-ci-tools#439](https://github.com/duckdb/extension-ci-tools/pull/439)
merges into `v2.0-cyanoptera`, community builds label the footer with the commit
and `LOAD` on an alpha host fails unless `allow_extensions_metadata_mismatch` is
set.

This was reproduced on duckboost’s own CI after the pin bump:

```
-- git hash b15358c371, version b15358c371daf10dad799a43b6167f6ccdce84d8,
   extension folder b15358c371daf10dad799a43b6167f6ccdce84d8
```

Local `make` is fine (Makefile exports `DUCKDB_VERSION=v2.0.0-alpha46057`). Do
**not** open the community-extensions PR until #439 is merged (or DuckDB confirms
another fix), or the published binary will not `LOAD`.

## Checklist before opening the PR

1. **Public GitHub repo** — `JavOrraca/duckboost` (already).
2. **Prefer cutting a release** that includes the community-alpha pin (CHANGELOG
   `0.0.4` / tag `v0.0.4`) so `extension.version` matches what community ships.
3. **Build locally against the community DuckDB pin**:

   ```bash
   git submodule update --init --recursive
   make
   make test
   make check-extension-metadata
   ```

   Configure should log
   `version v2.0.0-alpha46057, extension folder v2.0.0-alpha46057`.
   `make check-extension-metadata` asserts the same string is in the
   `.duckdb_extension` footer.
4. **Bump `repo.ref`** in
   [`docs/community_extensions_description.yml`](community_extensions_description.yml)
   to the **commit SHA** you want built (never a movable tag). Usually the merge
   commit of the release you are publishing.
5. Keep `extension.version` in sync with [`CHANGELOG.md`](../CHANGELOG.md) /
   the git tag.
6. Keep `docs.hello_world` aligned with
   [Getting Started](https://javorraca.github.io/duckboost/getting-started.html)
   (HTTPS CSV; no repo checkout required).
7. Do **not** add `variegata` (v1.5.6 last-minute track) or `ref_next`
   (prerelease leg retired for new PRs). duckboost is 2.0-only.
8. Confirm #439 (above) is on the community CI tools ref.

## Open the community-extensions PR

1. Fork [duckdb/community-extensions](https://github.com/duckdb/community-extensions).
2. Copy `docs/community_extensions_description.yml` to
   `extensions/duckboost/description.yml` (**that file only**).
3. Open a PR. Community CI builds and tests via
   `extension-ci-tools` `@v2.0-cyanoptera`.
4. In the PR checks, confirm configure logs
   `version v2.0.0-alpha46057` (not the bare commit).
5. After merge + deploy, users can:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

Docs: https://duckdb.org/community_extensions/documentation.html

## `duckdb_version` vs `duckdb_tag`

| Input | Example | Role |
| --- | --- | --- |
| `duckdb_version` | `b15358c…` (commit) | Git checkout of DuckDB |
| `duckdb_tag` | `v2.0.0-alpha46057` | Deploy path + extension footer label |

See [extension-ci-tools#439](https://github.com/duckdb/extension-ci-tools/pull/439)
for why both inputs are required on DuckDB 2.0.

## After official DuckDB 2.0.0

No descriptor change is required for the redeploy under `v2.0.0` — that is on the
DuckDB team. Optionally cut a duckboost release once clients ship 2.0.0 and update
docs that still say “alpha”.

See also [`PACKAGING.md`](../PACKAGING.md).
