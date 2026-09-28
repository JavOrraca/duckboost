"""Run duckboost SQL through the repo's release DuckDB shell.

Every call starts a new in-memory database. Display helpers print a ```sql
fence and the shell's box output so Quarto pages can hide this module
(`echo: false`, `output: asis`).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DUCKDB = REPO_ROOT / "build" / "release" / "duckdb"
EXTENSION = REPO_ROOT / "build" / "release" / "extension" / "duckboost" / "duckboost.duckdb_extension"

# Relative to the repository root. The shell is started with that cwd.
LOAD = "LOAD 'build/release/extension/duckboost/duckboost.duckdb_extension';"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _require_binary() -> None:
    if not DUCKDB.is_file():
        raise FileNotFoundError(
            f"DuckDB binary not found at {DUCKDB}. Run `make` first."
        )


def run_sql(sql: str, *, json_output: bool = False) -> str:
    """Execute SQL in a fresh in-memory database. Raises on a non-zero exit."""
    _require_binary()
    cmd = [
        str(DUCKDB),
        "-unsigned",
        "-bail",
        "-json" if json_output else "-box",
        "-c",
        sql,
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = _ANSI.sub("", proc.stdout)
    stderr = _ANSI.sub("", proc.stderr)
    if proc.returncode != 0:
        detail = (stderr or stdout).strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"duckdb failed ({DUCKDB}):\n{detail}\n\nSQL:\n{sql}")
    return stdout


def parse_json_sets(text: str) -> list:
    decoder = json.JSONDecoder()
    index = 0
    sets: list = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, end = decoder.raw_decode(text, index)
        sets.append(value)
        index = end
    return sets


def show(sql: str) -> None:
    """Print the SQL and its box output as Markdown."""
    sql = sql.strip()
    output = run_sql(sql).rstrip()
    print(f"```sql\n{sql}\n```\n")
    print("```text")
    print(output)
    print("```")


def show_export_roundtrip(
    setup_sql: str, native_sql: str, export_sql: str, score_ddl: str
) -> None:
    """Train in one shell, then run the exported SQL in a second shell.

    The second shell's script is only `score_ddl` plus the generated SELECT.
    It does not call `duckboost_*`. This repository's `make` shell statically
    links duckboost, so the extension is still registered there; the scoring
    statements do not use it.
    """
    setup_sql = setup_sql.strip()
    native_sql = native_sql.strip()
    export_sql = export_sql.strip()
    score_ddl = score_ddl.strip()

    with_extension = f"{setup_sql}\n{native_sql}\n{export_sql}"
    sets = parse_json_sets(run_sql(with_extension, json_output=True))
    if len(sets) < 2:
        raise RuntimeError(
            "Expected native predictions and exported SQL as the last two result sets."
        )
    native_rows = sets[-2]
    export_rows = sets[-1]
    if len(export_rows) != 1:
        raise RuntimeError(f"Expected one exported SQL string, got {export_rows!r}")
    exported = next(iter(export_rows[0].values()))
    # Tree subqueries are named "_duckboost_trees". A real extension call is
    # an unprefixed duckboost_* function.
    if not isinstance(exported, str) or re.search(r"(?<![_\w])duckboost_", exported):
        raise RuntimeError("Exported SQL was empty or still calls duckboost.")

    plain_sql = f"{score_ddl}\n{exported.rstrip().rstrip(';')};"
    plain_sets = parse_json_sets(run_sql(plain_sql, json_output=True))
    if not plain_sets:
        raise RuntimeError("Plain-SQL session returned no rows.")
    plain_rows = plain_sets[-1]

    native_preds = [float(row["pred"]) for row in native_rows]
    plain_preds = [float(next(iter(row.values()))) for row in plain_rows]
    if len(native_preds) != len(plain_preds):
        raise RuntimeError(
            f"Prediction row counts differ: extension {len(native_preds)}, SQL {len(plain_preds)}"
        )
    max_abs = max(abs(a - b) for a, b in zip(native_preds, plain_preds))
    if max_abs > 1e-8:
        raise RuntimeError(
            f"Exported SQL predictions differ from duckboost_predict (max abs {max_abs})."
        )

    print("Extension session: train, then `duckboost_predict`.\n")
    show(f"{setup_sql}\n{native_sql}")
    print("\nSame model, exported with `duckboost_to_sql`:\n")
    print("```sql")
    print(export_sql)
    print("```")
    print()
    print("Generated query:\n")
    print("```sql")
    print(exported.strip())
    print("```")
    print(
        "\nThe next result is that query in a new `build/release/duckdb` process. "
        "The script is the score-table DDL plus the generated SELECT, and it never "
        "calls `duckboost_*`. `make` links duckboost into that shell, so the "
        "extension is registered, but these statements do not use it.\n"
    )
    print("```sql")
    print(score_ddl)
    print("-- then the generated SELECT above")
    print("```\n")
    print("```text")
    print(run_sql(plain_sql).rstrip())
    print("```")
    print()
    print(f"Maximum absolute difference versus `duckboost_predict`: `{max_abs}`.")
