#!/usr/bin/env python3
"""Fail if site/_freeze is stale for Quarto pages with executable cells.

Publish docs uses execute.freeze: auto and does not build DuckDB. Any edit to a
.qmd that contains ```{python} cells changes Quarto's freeze hash (MD5 of the
file) and forces re-execution. Pages that call site/_helpers/duckrun.py then
fail in CI. Refresh with:

  make
  python3 -m venv .venv-quarto && .venv-quarto/bin/pip install jupyter nbformat nbclient ipykernel
  QUARTO_PYTHON=$PWD/.venv-quarto/bin/python quarto render site
  git add site/_freeze && git commit
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
FREEZE = SITE / "_freeze"


def has_executable_cells(text: str) -> bool:
    return "```{python}" in text or "```{r}" in text


def freeze_path_for(qmd: Path) -> Path:
    rel = qmd.relative_to(SITE)
    return FREEZE / rel.with_suffix("") / "execute-results" / "html.json"


def main() -> int:
    stale: list[str] = []
    checked = 0
    for qmd in sorted(SITE.rglob("*.qmd")):
        text = qmd.read_text(encoding="utf-8")
        if not has_executable_cells(text):
            continue
        checked += 1
        digest = hashlib.md5(qmd.read_bytes()).hexdigest()
        freeze = freeze_path_for(qmd)
        rel = qmd.relative_to(SITE)
        if not freeze.is_file():
            stale.append(f"{rel}: missing {freeze.relative_to(REPO)}")
            continue
        try:
            payload = json.loads(freeze.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            stale.append(f"{rel}: invalid freeze JSON ({exc})")
            continue
        stored = payload.get("hash")
        if stored != digest:
            stale.append(f"{rel}: freeze hash {stored!r} != file md5 {digest!r}")

    if stale:
        print("site/_freeze is out of date for executable Quarto pages:", file=sys.stderr)
        for item in stale:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nRefresh freeze locally (needs `make` + Jupyter), then commit site/_freeze:\n"
            "  python3 -m venv .venv-quarto\n"
            "  .venv-quarto/bin/pip install jupyter nbformat nbclient ipykernel\n"
            "  QUARTO_PYTHON=$PWD/.venv-quarto/bin/python quarto render site\n",
            file=sys.stderr,
        )
        return 1

    print(f"site/_freeze ok for {checked} executable page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
