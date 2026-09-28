# Publishing duckboost as a community extension

1. Push this repository to GitHub (public), e.g. `JavOrraca/duckboost`.
2. Note the commit SHA to publish.
3. Copy `docs/community_extensions_description.yml` to a fork of
   [duckdb/community-extensions](https://github.com/duckdb/community-extensions) as
   `extensions/duckboost/description.yml`, set `repo.ref` to that SHA, and open a PR.
4. After merge, users can:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

See also `PACKAGING.md` in the repository root.
