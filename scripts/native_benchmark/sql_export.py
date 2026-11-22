"""Measure generated SQL size, PREPARE time and EXECUTE time in a stock DuckDB CLI."""

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess


def run(cli, sql):
    return subprocess.run(
        [cli, "-no-agent", "-unsigned", "-bail", "-batch", "-json"],
        input=sql,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def full_tree(depth, offset):
    nodes = []
    for i in range(2 ** (depth + 1) - 1):
        if i >= 2**depth - 1:
            nodes.append({"is_leaf": True, "value": 100000 + offset + i})
        else:
            level = (i + 1).bit_length() - 1
            split = (2 * (i + 1 - 2**level) + 1) * 2 ** (depth - level - 1)
            nodes.append(
                {
                    "is_leaf": False,
                    "feature": 0,
                    "threshold": split,
                    "left": 2 * i + 1,
                    "right": 2 * i + 2,
                    "default_left": True,
                }
            )
    return {"nodes": nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("--cli", default="duckdb")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    load = str(args.extension.resolve()).replace("'", "''")
    for depth, trees in [(8, 1), (5, 8)]:
        model = json.dumps(
            {
                "duckboost_version": 1,
                "task": "regression",
                "n_features": 1,
                "base_score": 0,
                "learning_rate": 1,
                "trees": [full_tree(depth, t * 1000) for t in range(trees)],
            }
        )
        for separate in ["false", "true"]:
            exported = json.loads(
                run(
                    args.cli,
                    f"""
LOAD '{load}';
SELECT duckboost_to_sql('{model}', 'probes', ['x'],
                        MAP {{'separate_trees':'{separate}'}}) AS q;
""",
                )
            )[0]["q"]
            plan, score = [], []
            for _ in range(args.repeats):
                # Measure planning and execution independently; exclude setup, CLI startup and printing rows.
                output = run(
                    args.cli,
                    f"""
SET threads=1;
CREATE TABLE probes AS SELECT (i % {2**depth})::DOUBLE AS x FROM range(100000) t(i);
.timer on
PREPARE score AS SELECT sum(prediction) AS total FROM ({exported});
EXECUTE score;
""",
                )
                times = [
                    float(x)
                    for x in re.findall(r"Run Time \(s\): real ([0-9.]+)", output)
                ]
                p, s = times
                plan.append(p)
                score.append(s)
            print(
                json.dumps(
                    dict(
                        depth=depth,
                        trees=trees,
                        separate_trees=separate == "true",
                        bytes=len(exported.encode()),
                        leaf_occurrences=len(re.findall(r"\b1\d{5}e0\b", exported)),
                        prepare_seconds=statistics.median(plan),
                        score_seconds=statistics.median(score),
                        checksum=json.loads(
                            next(
                                line
                                for line in output.splitlines()
                                if line.startswith("[{")
                            )
                        )[0]["total"],
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
