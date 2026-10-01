"""Canonical HTTPS URLs for Getting Started / vignette example datasets.

Printed SQL uses these URLs so readers do not need a repo checkout. Docs
execution rewrites ``read_csv('https://...')`` to a locally cached file because
this repository's DuckDB 2.0 alpha pin often cannot install ``httpfs`` from the
CDN. End users on released DuckDB load the URLs directly (``httpfs`` autoload).
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / ".cache" / "example_data"

PENGUINS_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "master/inst/extdata/penguins.csv"
)
IRIS_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
    "master/csv/datasets/iris.csv"
)

# Drop-in loaders for vignette SQL (snake_case columns match prior local CSVs).
IRIS_FROM_URL = f"""
SELECT
  "Sepal.Length" AS sepal_length,
  "Sepal.Width" AS sepal_width,
  "Petal.Length" AS petal_length,
  "Petal.Width" AS petal_width,
  Species AS species
FROM read_csv('{IRIS_URL}')
""".strip()

PENGUINS_FROM_URL = f"""
FROM read_csv('{PENGUINS_URL}', nullstr = 'NA')
""".strip()

_HTTPS_CSV = re.compile(
    r"""read_csv\(\s*(['"])(https://[^'"]+\.csv)\1""",
    re.IGNORECASE,
)


def cache_csv(url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(url.encode()).hexdigest()[:16] + ".csv"
    path = CACHE / name
    if path.is_file() and path.stat().st_size > 0:
        return path
    with urllib.request.urlopen(url, timeout=120) as resp:
        path.write_bytes(resp.read())
    return path


def rewrite_https_csv_for_exec(sql: str) -> str:
    """Replace HTTPS read_csv paths with cached local files for docs execution."""

    def repl(match: re.Match[str]) -> str:
        quote = match.group(1)
        url = match.group(2)
        local = cache_csv(url).as_posix()
        return f"read_csv({quote}{local}{quote}"

    return _HTTPS_CSV.sub(repl, sql)
