# Publishing duckboost as a community extension

Hold until DuckDB 2.0 is released (this repo tracks the 2.0 alpha).

1. Push this repository to GitHub (public), e.g. `JavOrraca/duckboost`.
2. Set `repo.ref` in [`docs/community_extensions_description.yml`](community_extensions_description.yml) to the commit SHA to publish.
3. Copy that file to a fork of
   [duckdb/community-extensions](https://github.com/duckdb/community-extensions) as
   `extensions/duckboost/description.yml` and open a PR.
4. After merge, users can:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

`docs.hello_world` is the Palmer penguins first-model loop (same idea as the site Getting Started page), reading the CSV from the public raw GitHub URL so the catalog runner does not need a checkout.

See also `PACKAGING.md` in the repository root.
