#!/usr/bin/env python3
"""Compare two vendor parity fixture directories.

Checks that both manifests list the same cases and that every probe CSV has
the same rows, kinds, and feature values, with expected / proba / margin
columns inside each case's manifest tolerance. Byte differences in dumps and
configs are reported but only fail with --strict-dumps.

Usage:
  python scripts/vendor_parity/compare.py BASELINE_DIR CANDIDATE_DIR [--report-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

VALUE_PREFIXES = ("expected", "proba_", "margin")


def load_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def read_probes(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def parse(value: str) -> float:
    return math.nan if value == "" else float(value)


def close(a: float, b: float, tol: dict) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol["abs"] + tol["rel"] * abs(a)


def compare_case(name: str, base_dir: Path, cand_dir: Path, base: dict, cand: dict,
                 tolerance_scale: float) -> list[str]:
    problems = []
    tol = {k: v * tolerance_scale for k, v in base["tolerance"].items()}
    base_cols, base_rows = read_probes(base_dir / base["files"]["probes"])
    cand_cols, cand_rows = read_probes(cand_dir / cand["files"]["probes"])
    if base_cols != cand_cols:
        return [f"{name}: probe columns differ: {base_cols} vs {cand_cols}"]
    if len(base_rows) != len(cand_rows):
        return [f"{name}: probe row count {len(base_rows)} vs {len(cand_rows)}"]
    worst = 0.0
    for b, c in zip(base_rows, cand_rows):
        if b["kind"] != c["kind"]:
            problems.append(f"{name}: probe {b['id']} kind {b['kind']} vs {c['kind']}")
            continue
        for column in base_cols:
            if column in ("id", "kind"):
                continue
            bv, cv = parse(b[column]), parse(c[column])
            is_value = column.startswith(VALUE_PREFIXES)
            if not close(bv, cv, tol if is_value else {"abs": 0.0, "rel": 0.0}):
                label = "prediction" if is_value else "feature"
                problems.append(f"{name}: probe {b['id']} ({b['kind']}) {label} {column}: {bv!r} vs {cv!r}")
            elif is_value and not math.isnan(bv):
                worst = max(worst, abs(bv - cv))
    if not problems:
        print(f"  ok    {name:<24} {len(base_rows):>3} probes, max |diff| {worst:.2g}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report-only", action="store_true", help="print differences but exit 0")
    parser.add_argument("--strict-dumps", action="store_true", help="fail when dump or config bytes differ")
    parser.add_argument("--tolerance-scale", type=float, default=1.0,
                        help="multiply every case tolerance by this factor")
    args = parser.parse_args(argv)

    base_manifest, cand_manifest = load_manifest(args.baseline), load_manifest(args.candidate)
    print(f"baseline:  {args.baseline} {base_manifest['versions']}")
    print(f"candidate: {args.candidate} {cand_manifest['versions']}")
    base_cases = {c["name"]: c for c in base_manifest["cases"]}
    cand_cases = {c["name"]: c for c in cand_manifest["cases"]}

    problems = []
    for missing in sorted(base_cases.keys() - cand_cases.keys()):
        problems.append(f"{missing}: missing from candidate")
    for extra in sorted(cand_cases.keys() - base_cases.keys()):
        problems.append(f"{extra}: not in baseline")
    notes = []
    for name in [c["name"] for c in base_manifest["cases"] if c["name"] in cand_cases]:
        base, cand = base_cases[name], cand_cases[name]
        for kind, filename in base["files"].items():
            if kind == "probes":
                continue
            base_file, cand_file = args.baseline / filename, args.candidate / cand["files"].get(kind, filename)
            if not cand_file.exists() or base_file.read_bytes() != cand_file.read_bytes():
                (problems if args.strict_dumps else notes).append(f"{name}: {kind} file {filename} differs")
        problems += compare_case(name, args.baseline, args.candidate, base, cand, args.tolerance_scale)

    for note in notes:
        print(f"  note  {note}")
    for problem in problems:
        print(f"  FAIL  {problem}")
    if problems:
        print(f"{len(problems)} difference(s) found")
        return 0 if args.report_only else 1
    print("fixtures match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
