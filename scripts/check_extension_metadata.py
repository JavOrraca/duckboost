#!/usr/bin/env python3
"""Assert a built .duckdb_extension footer carries the expected DuckDB version.

DuckDB appends fixed-width metadata fields near the end of the binary. Local
`make` labels with DUCKDB_COMMUNITY_TAG (v2.0.0). Community CI passes
duckdb_version=v2.0.0 with duckdb_commit set to the release SHA.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_EXTENSION = Path("build/release/extension/duckboost/duckboost.duckdb_extension")
DEFAULT_TAG = "v2.0.0"


def printable_tail(data: bytes, nbytes: int = 8192) -> list[str]:
    tail = data[-nbytes:]
    return [m.decode("ascii") for m in re.findall(rb"[\x20-\x7e]{4,}", tail)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "extension",
        nargs="?",
        type=Path,
        default=DEFAULT_EXTENSION,
        help=f"path to .duckdb_extension (default: {DEFAULT_EXTENSION})",
    )
    parser.add_argument(
        "--expect-tag",
        default=DEFAULT_TAG,
        help=f"required DuckDB version label in the footer (default: {DEFAULT_TAG})",
    )
    args = parser.parse_args()

    if not args.extension.is_file():
        print(f"error: missing extension binary: {args.extension}", file=sys.stderr)
        print("hint: run `make` / `make release` first", file=sys.stderr)
        return 2

    data = args.extension.read_bytes()
    strings = printable_tail(data)
    if args.expect_tag not in strings:
        print(
            f"error: footer does not contain {args.expect_tag!r} in {args.extension}",
            file=sys.stderr,
        )
        print("nearby footer strings:", file=sys.stderr)
        for item in strings[-20:]:
            print(f"  {item}", file=sys.stderr)
        print(
            "hint: local builds need DUCKDB_VERSION / OVERRIDE_GIT_DESCRIBE set "
            "(see Makefile).",
            file=sys.stderr,
        )
        return 1

    print(f"ok: {args.extension} footer includes {args.expect_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
