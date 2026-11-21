#!/usr/bin/env python3
"""Assert a built .duckdb_extension footer carries the expected DuckDB version.

DuckDB appends fixed-width metadata fields near the end of the binary. Local
`make` labels with DUCKDB_COMMUNITY_TAG (v2.0.1). Community CI passes
duckdb_version=v2.0.1 with duckdb_commit set to the release SHA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_EXTENSION = Path("build/release/extension/duckboost/duckboost.duckdb_extension")
DEFAULT_TAG = "v2.0.1"


def engine_version(path: Path) -> str:
    # DuckDB's 512-byte footer begins with eight 32-byte fields in reverse
    # order, followed by the signature. Field 5 is the engine version.
    with path.open("rb") as extension:
        if extension.seek(0, 2) < 512:
            return ""
        extension.seek(-512, 2)
        fields = [extension.read(32).rstrip(b"\0") for _ in range(8)]
    if fields[7] != b"4" or fields[3] != b"CPP":
        return ""
    return fields[5].decode("ascii", errors="replace")


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

    actual = engine_version(args.extension)
    if actual != args.expect_tag:
        print(
            f"error: {args.extension} has engine footer {actual!r}, expected {args.expect_tag!r}",
            file=sys.stderr,
        )
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
