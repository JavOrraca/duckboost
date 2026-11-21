#!/usr/bin/env python3
"""Medium-scale native vs LightGBM holdout benchmark for duckboost.

Generates a deterministic synthetic tabular problem, trains with the same
hyperparameter grid on backend='native' and backend='lightgbm', and writes
holdout metrics + wall times to results.json.

Requires a release build with LightGBM linked (see native-train CI flags).

Usage (from repo root):
  python3 scripts/native_benchmark/run.py
  python3 scripts/native_benchmark/run.py --n-rows 200000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DUCKDB = REPO / "build" / "release" / "duckdb"
EXTENSION = REPO / "build" / "release" / "extension" / "duckboost" / "duckboost.duckdb_extension"
RESULTS = Path(__file__).resolve().parent / "results.json"

# Shared grid — keep options both backends accept.
TRAIN_OPTIONS = {
    "n_estimators": "80",
    "max_depth": "5",
    "learning_rate": "0.1",
    "min_samples_leaf": "20",
    "max_bins": "64",
    "reg_lambda": "1.0",
    "subsample": "0.8",
    "colsample_bytree": "0.8",
    "seed": "7",
}


def _ensure_lib_path() -> None:
    extras = [
        Path.home() / ".local/lib/python3.12/site-packages/lightgbm/lib",
        Path.home() / ".local/lib/python3.12/site-packages/xgboost/lib",
    ]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(p) for p in extras if p.is_dir()]
    if cur:
        parts.append(cur)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def run_sql(sql: str, *, json_output: bool = False) -> tuple[str, float | None]:
    if not DUCKDB.is_file():
        raise FileNotFoundError(f"DuckDB binary not found at {DUCKDB}; run `make` first.")
    cmd = [str(DUCKDB), "-unsigned", "-bail"]
    if json_output:
        cmd.append("-json")
    else:
        cmd.append("-box")
    cmd.extend(["-c", sql])
    # wait4 reports the peak RSS of this CLI, independently of previous fits.
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        with subprocess.Popen(cmd, cwd=REPO, stdout=stdout, stderr=stderr) as proc:
            if hasattr(os, "wait4"):
                _, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
                rss_bytes = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
                peak_mib = round(rss_bytes / (1024 * 1024), 2)
            else:
                proc.wait()
                peak_mib = None
            stdout.seek(0)
            output = stdout.read()
            if proc.returncode:
                stderr.seek(0)
                raise RuntimeError(f"duckdb failed:\n{stderr.read() or output}\n\nSQL:\n{sql}")
    return output, peak_mib


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


def map_literal(options: dict[str, str]) -> str:
    inner = ", ".join(f"'{k}': '{v}'" for k, v in options.items())
    return f"MAP {{{inner}}}"


def build_dataset_sql(n_rows: int, seed: int) -> str:
    # Deterministic features from row index; target has additive + interaction signal + noise.
    return f"""
CREATE OR REPLACE TABLE bench AS
SELECT
  i,
  (i % 17)::DOUBLE AS cat_a,
  (i % 9)::DOUBLE AS cat_b,
  ((i * 3) % 23)::DOUBLE AS cat_c,
  ((i * 7) % 5)::DOUBLE AS cat_d,
  sin(i * 0.017)::DOUBLE AS x1,
  cos(i * 0.013)::DOUBLE AS x2,
  ((i % 100) / 50.0 - 1.0)::DOUBLE AS x3,
  ln(1.0 + (i % 200))::DOUBLE AS x4,
  ((i * 11) % 97) / 97.0::DOUBLE AS x5,
  ((i * 13) % 53) / 53.0::DOUBLE AS x6,
  ((i * 17) % 41) / 41.0::DOUBLE AS x7,
  ((i * 19) % 37) / 37.0::DOUBLE AS x8,
  (
    1.7 * sin(i * 0.017)
    + 2.3 * cos(i * 0.013)
    + 0.9 * ((i % 100) / 50.0 - 1.0)
    + 0.4 * ((i % 17) - 8)
    + 0.55 * ((i % 9) - 4)
    + 1.1 * sin(i * 0.017) * ((i % 9) - 4)
    + 0.15 * (((i * {seed}) % 1000) / 1000.0 - 0.5)
  )::DOUBLE AS y_reg,
  CASE
    WHEN (
      1.7 * sin(i * 0.017)
      + 2.3 * cos(i * 0.013)
      + 0.4 * ((i % 17) - 8)
      + 0.15 * (((i * {seed}) % 1000) / 1000.0 - 0.5)
    ) > 0 THEN 1.0 ELSE 0.0
  END AS y_bin,
  CASE WHEN hash(i, {seed}) % 5 = 0 THEN 'test' ELSE 'train' END AS split
FROM range({n_rows}) t(i);

CREATE OR REPLACE TABLE bench_feat AS
SELECT
  i, split, y_reg, y_bin,
  [cat_a, cat_b, cat_c, cat_d, x1, x2, x3, x4, x5, x6, x7, x8] AS features
FROM bench;
"""


def wide_skewed_sql(n_rows: int, seed: int) -> str:
    # Each of the 32 numeric columns has a zero/one spike and a long right tail.
    columns = []
    for j in range(32):
        h = f"hash(i, {j}, {seed})"
        columns.append(f"(CASE WHEN {h} % 100 < 48 THEN 0 WHEN {h} % 100 < 96 "
                       f"THEN 1 ELSE 2 + {h} % 10000 END)::DOUBLE")
    return f"""
CREATE TABLE wide AS SELECT i, [{', '.join(columns)}] AS features FROM range({n_rows}) t(i);
CREATE TABLE bench_feat AS
SELECT *, IF(hash(i, {seed}) % 5 = 0, 'test', 'train') AS split,
       (3 * (features[1] > 0)::INT + 2 * (features[2] > 0)::INT +
        (features[3] > 0)::INT + 0.2 * sin(i))::DOUBLE AS y_reg
FROM wide;
"""


def train_and_eval(backend: str, task: str, n_rows: int, seed: int,
                   threads: int | None = None, wide: bool = False) -> dict:
    options = dict(TRAIN_OPTIONS)
    options["backend"] = backend
    options["seed"] = str(seed)
    if task == "binary":
        options["task"] = "binary"
        y_col = "y_bin"
        metric = "logloss"
        secondary = "accuracy"
    else:
        options["task"] = "regression"
        y_col = "y_reg"
        metric = "rmse"
        secondary = None

    if not wide:
        options["feature_names"] = "cat_a,cat_b,cat_c,cat_d,x1,x2,x3,x4,x5,x6,x7,x8"
        options["categorical_features"] = "cat_a,cat_b,cat_c,cat_d"

    load = f"LOAD '{EXTENSION.as_posix()}';" if EXTENSION.is_file() else "LOAD duckboost;"
    settings = f"SET threads={threads};" if threads else ""
    dataset = wide_skewed_sql if wide else build_dataset_sql
    setup = load + settings + "\n" + dataset(n_rows, seed)

    # One session: time only duckboost_train via epoch_ms, then evaluate holdout.
    sql = f"""
{setup}
CREATE OR REPLACE TABLE _t0 AS SELECT epoch_ms(current_timestamp) AS ms;
CREATE OR REPLACE TABLE model_one AS
SELECT duckboost_train(
  {y_col}, features,
  {map_literal(options)}
) AS model
FROM bench_feat
WHERE split = 'train';
CREATE OR REPLACE TABLE _t1 AS SELECT epoch_ms(current_timestamp) AS ms;

SELECT (SELECT ms FROM _t1) - (SELECT ms FROM _t0) AS train_ms;

SELECT
  round(duckboost_evaluate_agg(m.model, t.{y_col}, t.features, MAP {{'metric': '{metric}'}}), 6) AS primary_metric
FROM model_one m, bench_feat t
WHERE t.split = 'test';
"""
    if secondary:
        sql += f"""
SELECT
  round(duckboost_evaluate_agg(m.model, t.{y_col}, t.features, MAP {{'metric': '{secondary}'}}), 6) AS secondary_metric
FROM model_one m, bench_feat t
WHERE t.split = 'test';
"""
    sql += """
SELECT
  count(*) FILTER (WHERE split = 'train') AS n_train,
  count(*) FILTER (WHERE split = 'test') AS n_test
FROM bench_feat;
"""

    output, peak_mib = run_sql(sql, json_output=True)
    sets = parse_json_sets(output)
    train_ms = float(sets[0][0]["train_ms"])
    primary = float(sets[1][0]["primary_metric"])
    idx = 2
    secondary_val = None
    if secondary:
        secondary_val = float(sets[idx][0]["secondary_metric"])
        idx += 1
    counts = sets[idx][0]

    result = {
        "backend": backend,
        "peak_process_mib": peak_mib,
        "threads": threads,
        "workload": "wide-skewed" if wide else "standard",
        "task": task,
        "metric": metric,
        "holdout_metric": primary,
        "train_seconds": round(train_ms / 1000.0, 3),
        "n_train": int(counts["n_train"]),
        "n_test": int(counts["n_test"]),
        "options": options,
    }
    if secondary:
        result["secondary_metric_name"] = secondary
        result["secondary_metric"] = secondary_val
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-lightgbm", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--wide", action="store_true", help="Run the 32-feature skewed regression workload.")
    parser.add_argument("--output", type=Path, default=RESULTS)
    global DUCKDB, EXTENSION
    parser.add_argument("--cli", type=Path, default=DUCKDB)
    parser.add_argument("--extension", type=Path, default=EXTENSION)
    args = parser.parse_args()
    if args.repeats < 1 or (args.threads is not None and args.threads < 1):
        parser.error("--repeats and --threads must be positive")
    DUCKDB, EXTENSION = args.cli.resolve(), args.extension.resolve()

    _ensure_lib_path()
    load = f"LOAD '{EXTENSION.as_posix()}';" if EXTENSION.is_file() else "LOAD duckboost;"
    info = parse_json_sets(
        run_sql(
            f"""
{load}
SELECT name, enabled FROM duckboost_build_info()
WHERE name IN ('native_lightgbm_linked', 'native_xgboost_linked');
""",
            json_output=True,
        )[0]
    )[0]
    linked = {row["name"]: bool(row["enabled"]) for row in info}
    if not args.skip_lightgbm and not linked.get("native_lightgbm_linked"):
        print("LightGBM is not linked in this build; pass --skip-lightgbm or rebuild with DUCKBOOST_WITH_LIGHTGBM.", file=sys.stderr)
        return 2

    backends = ["native"]
    if not args.skip_lightgbm:
        backends.append("lightgbm")

    runs = []
    for task in (("regression",) if args.wide else ("regression", "binary")):
        for backend in backends:
            print(f"training task={task} backend={backend} n_rows={args.n_rows} ...", flush=True)
            samples = [
                train_and_eval(backend, task, args.n_rows, args.seed, args.threads, args.wide)
                for _ in range(args.repeats)
            ]
            run = samples[0].copy()
            run["train_seconds_samples"] = [sample["train_seconds"] for sample in samples]
            run["peak_process_mib_samples"] = [sample["peak_process_mib"] for sample in samples]
            run["train_seconds"] = statistics.median(run["train_seconds_samples"])
            measured_peaks = [value for value in run["peak_process_mib_samples"] if value is not None]
            run["peak_process_mib"] = max(measured_peaks, default=None)
            run["holdout_metric_samples"] = [sample["holdout_metric"] for sample in samples]
            if "secondary_metric" in run:
                run["secondary_metric_samples"] = [sample["secondary_metric"] for sample in samples]
            runs.append(run)
            print(json.dumps(runs[-1], indent=2), flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "n_rows": args.n_rows,
        "seed": args.seed,
        "shared_options": TRAIN_OPTIONS,
        "runs": runs,
        "notes": [
            "Synthetic data generated in DuckDB from range(n); ~20% holdout via hash(i, seed) % 5 = 0.",
            "Train wall time is epoch_ms(current_timestamp) around duckboost_train only (dataset build excluded).",
            "Holdout metrics use duckboost_evaluate_agg on the test split.",
            "Training includes sorting and binning. Peak process RSS also includes dataset creation and evaluation.",
            "Same hyperparameter map is passed to both trainers (mapped where the vendor bridge supports it).",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
