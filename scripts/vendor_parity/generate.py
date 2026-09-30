#!/usr/bin/env python3
"""Generate XGBoost / LightGBM parity fixtures for duckboost.

Each case trains a tiny deterministic model and writes, under
test/sql/duckboost/data/parity/:

  <case>.json / <case>.txt   vendor dump consumed by duckboost_import()
  <case>.config.json         XGBoost Booster.save_config() (XGBoost only)
  <case>.probes.csv          id, f0..fN-1, expected[, margin | proba_k, margin_k], kind
  manifest.json              case metadata and prediction semantics

Before anything is written, every probe is re-scored by an independent
pure-Python evaluator of the dump (see `xgb_score` / `lgb_score`). Generation
fails if that evaluator disagrees with the vendor, so the dump plus the
manifest metadata is always sufficient to reproduce `expected`.

CSV conventions: an empty field is a missing value (NaN in the vendor, NULL in
DuckDB). Floats are written with Python repr, which round-trips doubles.

Usage:
  python scripts/vendor_parity/generate.py [--out DIR] [--strict-versions]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "test" / "sql" / "duckboost" / "data" / "parity"
PINNED = {"xgboost": "3.4.1", "lightgbm": "4.7.0"}
SCHEMA_VERSION = 1
MAX_TOTAL_BYTES = 300 * 1024
N_TRAIN_ROWS = 200
N_TRAIN_PROBES = 12
MAX_THRESHOLD_SPLITS = 8
# LightGBM treats |x| <= kZeroThreshold (a float literal 1e-35f) as zero.
LGB_ZERO_THRESHOLD = float(np.float32(1e-35))
# XGBoost categories must be non-negative and exactly representable as float32 ints.
XGB_MAX_CAT = float(1 << 24)

TOLERANCE = {
    # XGBoost accumulates float32 leaves in float32; importers sum in double.
    "xgboost": {"abs": 1e-5, "rel": 1e-5},
    "lightgbm": {"abs": 1e-9, "rel": 1e-9},
}


@dataclass
class Case:
    name: str
    vendor: str
    description: str
    task: str
    X: np.ndarray
    y: np.ndarray
    params: dict
    num_rounds: int
    categorical: list[int] = field(default_factory=list)
    feature_names: list[str] | None = None
    n_classes: int = 0
    notes: list[str] = field(default_factory=list)
    max_ref_diff: float = 0.0

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def rng_for(name: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def numeric_matrix(rng, n_features, nan_frac=0.0):
    X = rng.normal(0.0, 1.0, (N_TRAIN_ROWS, n_features)).round(3)
    if nan_frac:
        X[rng.random(X.shape) < nan_frac] = np.nan
    return X


def regression_target(rng, X):
    Z = np.nan_to_num(X, nan=0.0)
    y = 2.0 * Z[:, 0] - Z[:, 1] ** 2 + 0.5 * np.sin(3.0 * Z[:, 2]) + rng.normal(0.0, 0.1, len(X))
    # Missingness carries signal so learned default directions matter.
    return y + 1.5 * np.isnan(X[:, 0]) - 1.0 * np.isnan(X[:, 1])


def binary_logit(X):
    Z = np.nan_to_num(X, nan=0.0)
    return 1.5 * Z[:, 0] - Z[:, 1] + Z[:, 0] * Z[:, 2]


def binary_target(rng, X):
    return (rng.random(len(X)) < sigmoid(binary_logit(X))).astype(float)


def multiclass_target(rng, X):
    W = np.array([[1.5, -1.0, 0.0], [-0.5, 1.2, 0.3], [0.2, 0.4, -1.4]])
    scores = np.nan_to_num(X, nan=0.0) @ W + rng.normal(0.0, 0.3, (len(X), 3))
    return scores.argmax(axis=1).astype(float)


def categorical_data(rng, n_categories=8):
    cat = rng.integers(0, n_categories, N_TRAIN_ROWS).astype(float)
    num = rng.normal(0.0, 1.0, N_TRAIN_ROWS).round(3)
    effects = rng.normal(0.0, 2.0, n_categories)
    y = effects[cat.astype(int)] + 0.5 * num + rng.normal(0.0, 0.1, N_TRAIN_ROWS)
    X = np.column_stack([cat, num])
    X[rng.random(N_TRAIN_ROWS) < 0.05, 0] = np.nan
    return X, y


def zero_heavy_data(rng):
    X = rng.normal(0.0, 1.0, (N_TRAIN_ROWS, 3)).round(3)
    X[rng.random(X.shape) < 0.25] = 0.0
    X[rng.random(X.shape) < 0.05] = np.nan
    Z = np.nan_to_num(X, nan=0.0)
    y = Z[:, 0] + 2.0 * (Z[:, 1] == 0.0) - 1.5 * (Z[:, 2] == 0.0) + rng.normal(0.0, 0.1, N_TRAIN_ROWS)
    return X, y


# --------------------------------------------------------------------------
# Case catalogue
# --------------------------------------------------------------------------

XGB_BASE = {"nthread": 1, "seed": 0, "tree_method": "hist", "eta": 0.3, "max_depth": 3}
LGB_BASE = {
    "num_leaves": 8,
    "max_depth": 3,
    "learning_rate": 0.3,
    "min_data_in_leaf": 5,
    "num_threads": 1,
    "deterministic": True,
    "force_col_wise": True,
    "seed": 7,
    "bagging_fraction": 1.0,
    "feature_fraction": 1.0,
    "verbose": -1,
}


def build_cases() -> list[Case]:
    cases: list[Case] = []

    rng = rng_for("xgb_regression_nan")
    X = numeric_matrix(rng, 3, nan_frac=0.15)
    cases.append(Case(
        "xgb_regression_nan", "xgboost", "reg:squarederror with 15% NaN inputs",
        "regression", X, regression_target(rng, X),
        {**XGB_BASE, "objective": "reg:squarederror"}, 10,
        notes=["NaN follows the node's 'missing' child.",
               "XGBoost compares float32(x) < float32(split_condition)."],
    ))

    rng = rng_for("xgb_binary")
    X = numeric_matrix(rng, 3, nan_frac=0.05)
    cases.append(Case(
        "xgb_binary", "xgboost", "binary:logistic; base_score is a probability",
        "binary", X, binary_target(rng, X),
        {**XGB_BASE, "objective": "binary:logistic"}, 10,
        notes=["base_score in the config is a probability; base_margin = logit(base_score)."],
    ))

    rng = rng_for("xgb_multiclass")
    X = numeric_matrix(rng, 3)
    cases.append(Case(
        "xgb_multiclass", "xgboost", "multi:softprob with 3 classes and per-class intercepts",
        "multiclass", X, multiclass_target(rng, X),
        {**XGB_BASE, "objective": "multi:softprob", "num_class": 3}, 5, n_classes=3,
        notes=["Trees are interleaved: tree i belongs to class i % 3.",
               "XGBoost 3.x stores one base_score per class (margin space)."],
    ))

    rng = rng_for("xgb_categorical")
    X, y = categorical_data(rng)
    cases.append(Case(
        "xgb_categorical", "xgboost", "partition-based categorical splits (max_cat_to_onehot=1)",
        "regression", X, y,
        {**XGB_BASE, "objective": "reg:squarederror", "max_cat_to_onehot": 1}, 8,
        categorical=[0],
        notes=["split_condition is a category list; members follow 'yes'.",
               "Categories truncate toward zero; negative or >= 2^24 values follow 'no'.",
               "NaN follows 'missing'."],
    ))

    rng = rng_for("xgb_named_reordered")
    X = numeric_matrix(rng, 3)
    y = 3.0 * np.sin(2.0 * X[:, 2]) + X[:, 0] - 0.2 * X[:, 1] + rng.normal(0.0, 0.1, N_TRAIN_ROWS)
    cases.append(Case(
        "xgb_named_reordered", "xgboost",
        "pandas columns with non-alphabetical names; first split is on the last column",
        "regression", X, y,
        {**XGB_BASE, "objective": "reg:squarederror"}, 8,
        feature_names=["zeta", "alpha", "mid"],
        notes=["Dump splits reference names, not indices. Column order must come from "
               "feature_names (config), not from first appearance in the dump."],
    ))

    rng = rng_for("lgb_regression_stump")
    X = numeric_matrix(rng, 2)
    cases.append(Case(
        "lgb_regression_stump", "lightgbm", "regression with 2-leaf trees",
        "regression", X, 2.0 * X[:, 0] - X[:, 1] + rng.normal(0.0, 0.1, N_TRAIN_ROWS),
        {**LGB_BASE, "objective": "regression", "num_leaves": 2, "max_depth": 1}, 8,
        notes=["The boost_from_average init score is folded into the first tree's leaves."],
    ))

    for name, objective, extra, desc in [
        ("lgb_regression", "regression", {}, "L2 regression with NaN inputs"),
        ("lgb_regression_l1", "regression_l1", {}, "L1 regression (leaf values renewed to medians)"),
        ("lgb_quantile", "quantile", {"alpha": 0.8}, "quantile regression, alpha=0.8"),
    ]:
        rng = rng_for(name)
        X = numeric_matrix(rng, 3, nan_frac=0.1)
        cases.append(Case(
            name, "lightgbm", desc, "regression", X, regression_target(rng, X),
            {**LGB_BASE, "objective": objective, **extra}, 8,
        ))

    for name, scale in [("lgb_binary_sigmoid1", 1.0), ("lgb_binary_sigmoid2", 2.0)]:
        rng = rng_for(name)
        X = numeric_matrix(rng, 3, nan_frac=0.05)
        cases.append(Case(
            name, "lightgbm", f"binary with sigmoid={scale:g}", "binary", X, binary_target(rng, X),
            {**LGB_BASE, "objective": "binary", "sigmoid": scale}, 8,
            notes=[f"probability = 1 / (1 + exp(-{scale:g} * raw)); objective line is "
                   f"'binary sigmoid:{scale:g}'."],
        ))

    rng = rng_for("lgb_cross_entropy")
    X = numeric_matrix(rng, 3)
    cases.append(Case(
        "lgb_cross_entropy", "lightgbm", "cross_entropy (xentropy) on soft labels in [0, 1]",
        "binary", X, np.clip(sigmoid(binary_logit(X)) + rng.normal(0.0, 0.05, N_TRAIN_ROWS), 0.0, 1.0),
        {**LGB_BASE, "objective": "cross_entropy"}, 8,
        notes=["probability = 1 / (1 + exp(-raw)); objective line is 'cross_entropy' "
               "(no 'binary' substring)."],
    ))

    rng = rng_for("lgb_multiclass")
    X = numeric_matrix(rng, 3)
    cases.append(Case(
        "lgb_multiclass", "lightgbm", "multiclass softmax with 3 classes",
        "multiclass", X, multiclass_target(rng, X),
        {**LGB_BASE, "objective": "multiclass", "num_class": 3}, 5, n_classes=3,
        notes=["num_tree_per_iteration=3; tree i belongs to class i % 3."],
    ))

    rng = rng_for("lgb_categorical")
    X, y = categorical_data(rng)
    cases.append(Case(
        "lgb_categorical", "lightgbm", "many-vs-many categorical bitset splits",
        "regression", X, y,
        {**LGB_BASE, "objective": "regression", "max_cat_to_onehot": 1, "min_data_per_group": 5,
         "cat_smooth": 1.0, "cat_l2": 1.0},
        8, categorical=[0],
        notes=["Bitset members go LEFT. Categories truncate toward zero; NaN and negative "
               "truncated values go RIGHT."],
    ))

    rng = rng_for("lgb_zero_as_missing")
    X, y = zero_heavy_data(rng)
    cases.append(Case(
        "lgb_zero_as_missing", "lightgbm", "zero_as_missing=true (missing_type=Zero)",
        "regression", X, y,
        {**LGB_BASE, "objective": "regression", "zero_as_missing": True}, 8,
        notes=["missing_type=Zero sends NaN and 0 (including |x| <= 1e-35f) to the default child."],
    ))

    return cases


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------


def xgb_feature_types(case: Case) -> list[str]:
    return ["c" if i in case.categorical else "q" for i in range(case.n_features)]


def xgb_dmatrix(case: Case, X, y=None):
    if case.feature_names:
        return xgb.DMatrix(pd.DataFrame(X, columns=case.feature_names), label=y)
    return xgb.DMatrix(X, label=y, feature_types=xgb_feature_types(case),
                       enable_categorical=bool(case.categorical))


def xgb_iter_splits(node):
    if "leaf" in node:
        return
    yield node
    for child in node["children"]:
        yield from xgb_iter_splits(child)


def xgb_cat_in(x32: np.float32, categories) -> bool:
    if x32 < 0 or x32 >= XGB_MAX_CAT:
        return False
    return int(x32) in categories


def xgb_eval_tree(node, row32, name_to_idx) -> float:
    while "leaf" not in node:
        x = row32[name_to_idx[node["split"]]]
        cond = node["split_condition"]
        if np.isnan(x):
            target = node["missing"]
        elif isinstance(cond, list):
            target = node["yes"] if xgb_cat_in(x, set(cond)) else node["no"]
        else:
            target = node["yes"] if x < np.float32(cond) else node["no"]
        node = next(c for c in node["children"] if c["nodeid"] == target)
    return float(node["leaf"])


def xgb_base_margin(config: dict) -> list[float]:
    raw = config["learner"]["learner_model_param"]["base_score"].strip("[]")
    base = [float(v) for v in raw.split(",")]
    objective = config["learner"]["objective"]["name"]
    if objective == "binary:logistic":
        return [math.log(p / (1.0 - p)) for p in base]
    return base


def xgb_score(trees, config, names, P, n_classes):
    name_to_idx = {n: i for i, n in enumerate(names)}
    base = xgb_base_margin(config)
    k = max(n_classes, 1)
    margins = np.zeros((len(P), k))
    for r, row in enumerate(P.astype(np.float32)):
        margins[r] = base
        for t, tree in enumerate(trees):
            margins[r, t % k] += xgb_eval_tree(tree, row, name_to_idx)
    return margins


def run_xgboost(case: Case, out: Path) -> dict:
    booster = xgb.train(case.params, xgb_dmatrix(case, case.X, case.y), case.num_rounds)
    dump_path = out / f"{case.name}.json"
    booster.dump_model(str(dump_path), dump_format="json")
    raw_config = booster.save_config()
    (out / f"{case.name}.config.json").write_text(raw_config + "\n")
    config = json.loads(raw_config)
    trees = json.loads(dump_path.read_text())
    names = case.feature_names or [f"f{i}" for i in range(case.n_features)]

    splits = []
    categories = {f: set() for f in case.categorical}
    for tree in trees:
        for node in xgb_iter_splits(tree):
            f = names.index(node["split"])
            if isinstance(node["split_condition"], list):
                categories[f].update(node["split_condition"])
            else:
                splits.append((f, float(np.float32(node["split_condition"]))))

    P, kinds = build_probes(case, splits, xgb_threshold_edges)
    dm = xgb_dmatrix(case, P)
    margin = booster.predict(dm, output_margin=True).astype(np.float64).reshape(len(P), -1)
    expected = booster.predict(dm).astype(np.float64).reshape(len(P), -1)
    ours = xgb_score(trees, config, names, P, case.n_classes)
    check_close(case, "margin", ours, margin)
    check_close(case, "prediction", transform(case, ours, 1.0), expected)

    base_margin = xgb_base_margin(config)
    import_options = {"task": case.task}
    if case.n_classes:
        import_options["n_classes"] = str(case.n_classes)
    else:
        import_options["base_score"] = repr(base_margin[0])
    if case.feature_names:
        import_options["feature_names"] = ",".join(case.feature_names)
    return {
        "booster_objective": config["learner"]["objective"]["name"],
        "num_trees": len(trees),
        "files": {"dump": dump_path.name, "config": f"{case.name}.config.json"},
        "base_score": config["learner"]["learner_model_param"]["base_score"],
        "base_margin": base_margin,
        "import_options": import_options,
        "categories_in_splits": {str(f): sorted(v) for f, v in categories.items()},
        "_probes": (P, kinds, expected, margin),
    }


def xgb_threshold_edges(t: float):
    t32 = np.float32(t)
    prev32 = np.nextafter(t32, np.float32(-np.inf))
    exact = float(t32)
    return [
        ("threshold_exact", exact),
        ("threshold_prev_f32", float(prev32)),
        # Distinct doubles that round to the threshold in float32 (XGBoost's input type).
        ("threshold_below_f64", float(np.nextafter(exact, -np.inf))),
        ("threshold_mid_f32", (float(prev32) + exact) / 2.0),
    ]


# --------------------------------------------------------------------------
# LightGBM
# --------------------------------------------------------------------------


def lgb_parse(text: str):
    header, trees, current = {}, [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Tree="):
            current = {}
            trees.append(current)
        elif line == "end of trees":
            break
        elif "=" in line:
            key, value = line.split("=", 1)
            (current if current is not None else header)[key] = value
    return header, trees


def _ints(tree, key):
    return [int(v) for v in tree.get(key, "").split()]


def _floats(tree, key):
    return [float(v) for v in tree.get(key, "").split()]


def lgb_eval_tree(tree, row) -> float:
    leaves = _floats(tree, "leaf_value")
    if int(tree["num_leaves"]) == 1:
        return leaves[0]
    feature, threshold = _ints(tree, "split_feature"), _floats(tree, "threshold")
    decision, left, right = _ints(tree, "decision_type"), _ints(tree, "left_child"), _ints(tree, "right_child")
    boundaries, bitset = _ints(tree, "cat_boundaries"), _ints(tree, "cat_threshold")
    node = 0
    while node >= 0:
        x, dt = row[feature[node]], decision[node]
        if dt & 1:
            member = False
            if not math.isnan(x) and int(x) >= 0:
                cat, idx = int(x), int(threshold[node])
                words = bitset[boundaries[idx]:boundaries[idx + 1]]
                member = cat // 32 < len(words) and (words[cat // 32] >> (cat % 32)) & 1 == 1
            node = left[node] if member else right[node]
            continue
        missing_type, default_left = (dt >> 2) & 3, bool(dt & 2)
        if math.isnan(x) and missing_type != 2:
            x = 0.0
        if (missing_type == 1 and x == 0.0) or (
                missing_type == 2 and math.isnan(x)):
            node = left[node] if default_left else right[node]
        else:
            node = left[node] if x <= threshold[node] else right[node]
    return leaves[~node]


def lgb_score(text: str, P, n_classes):
    header, trees = lgb_parse(text)
    k = int(header.get("num_tree_per_iteration", "1"))
    assert k == max(n_classes, 1), (k, n_classes)
    margins = np.zeros((len(P), k))
    for r, row in enumerate(P):
        # The dense-row predictor keeps only |x| > kZeroThreshold (or NaN); anything else is 0.
        row = [v if math.isnan(v) or abs(v) > LGB_ZERO_THRESHOLD else 0.0 for v in row]
        for t, tree in enumerate(trees):
            margins[r, t % k] += lgb_eval_tree(tree, row)
    return header, trees, margins


def lgb_sigmoid_scale(objective_line: str) -> float:
    for token in objective_line.split():
        if token.startswith("sigmoid:"):
            return float(token.split(":", 1)[1])
    return 1.0


def run_lightgbm(case: Case, out: Path) -> dict:
    dataset = lgb.Dataset(case.X, case.y, categorical_feature=case.categorical or "auto",
                          params={"verbose": -1}, free_raw_data=False)
    booster = lgb.train(case.params, dataset, num_boost_round=case.num_rounds)
    text = booster.model_to_string()
    dump_path = out / f"{case.name}.txt"
    dump_path.write_text(text)

    header, trees = lgb_parse(text)
    splits = []
    for tree in trees:
        for f, t, dt in zip(_ints(tree, "split_feature"), _floats(tree, "threshold"),
                            _ints(tree, "decision_type")):
            if not dt & 1:
                splits.append((f, t))

    P, kinds = build_probes(case, splits, lgb_threshold_edges)
    raw = booster.predict(P, raw_score=True).reshape(len(P), -1)
    expected = booster.predict(P).reshape(len(P), -1)
    _, _, ours = lgb_score(text, P, case.n_classes)
    check_close(case, "margin", ours, raw)
    scale = lgb_sigmoid_scale(header["objective"])
    check_close(case, "prediction", transform(case, ours, scale), expected)

    missing_types = Counter()
    for tree in trees:
        for dt in _ints(tree, "decision_type"):
            missing_types["categorical" if dt & 1 else ["none", "zero", "nan", "?"][(dt >> 2) & 3]] += 1
    return {
        "booster_objective": header["objective"],
        "num_trees": len(trees),
        "files": {"dump": dump_path.name},
        "sigmoid_scale": scale if case.task == "binary" else None,
        "split_kinds": dict(sorted(missing_types.items())),
        "import_options": {},
        "_probes": (P, kinds, expected, raw),
    }


def lgb_threshold_edges(t: float):
    return [
        ("threshold_exact", t),
        ("threshold_above_f64", float(np.nextafter(t, np.inf))),
        ("threshold_below_f64", float(np.nextafter(t, -np.inf))),
    ]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def transform(case: Case, margins: np.ndarray, sigmoid_scale: float) -> np.ndarray:
    if case.task == "multiclass":
        return softmax(margins)
    if case.task == "binary":
        return sigmoid(sigmoid_scale * margins)
    return margins


def check_close(case: Case, what: str, ours: np.ndarray, vendor: np.ndarray):
    tol = TOLERANCE[case.vendor]
    diff = np.abs(ours - vendor)
    bound = tol["abs"] + tol["rel"] * np.abs(vendor)
    if ours.shape != vendor.shape or not np.all(diff <= bound):
        worst = int(np.argmax(diff - bound)) // max(vendor.shape[1], 1) if ours.shape == vendor.shape else -1
        raise SystemExit(
            f"{case.name}: reference evaluator disagrees with {case.vendor} {what} "
            f"(shape {ours.shape} vs {vendor.shape}, worst probe row {worst}, "
            f"max diff {float(diff.max()) if diff.size else float('nan'):.3g})")
    case.max_ref_diff = max(case.max_ref_diff, float(diff.max()))


def base_row(case: Case) -> np.ndarray:
    row = np.nanmedian(case.X, axis=0).round(3)
    for f in case.categorical:
        values, counts = np.unique(case.X[~np.isnan(case.X[:, f]), f], return_counts=True)
        row[f] = values[np.argmax(counts)]
    return row


def build_probes(case: Case, splits, edge_fn):
    rows, kinds = [], []

    def add(row, kind):
        rows.append(np.array(row, dtype=np.float64))
        kinds.append(kind)

    for row in case.X[:N_TRAIN_PROBES]:
        add(row, "train")
    base = base_row(case)

    seen = []
    for f, t in splits:
        if (f, t) not in seen:
            seen.append((f, t))
    for f, t in seen[:MAX_THRESHOLD_SPLITS]:
        for kind, value in edge_fn(t):
            row = base.copy()
            row[f] = value
            add(row, kind)

    for f in case.categorical:
        n_categories = int(np.nanmax(case.X[:, f])) + 1
        edges = [("category", float(c)) for c in range(n_categories)]
        edges += [("category_fractional", v) for v in (1.5, 2.9, n_categories - 0.5)]
        edges += [("category_negative", v) for v in (-0.5, -1.0, -3.0)]
        edges += [("category_unseen", v) for v in (float(n_categories), 99.0)]
        for kind, value in edges:
            row = base.copy()
            row[f] = value
            add(row, kind)

    for f in range(case.n_features):
        specials = [("nan", np.nan), ("zero", 0.0), ("negative_zero", -0.0)]
        if case.vendor == "lightgbm" and f not in case.categorical:
            specials += [("tiny_positive", 1e-36), ("tiny_negative", -1e-36)]
        for kind, value in specials:
            row = base.copy()
            row[f] = value
            add(row, kind)
    add(np.full(case.n_features, np.nan), "all_nan")
    add(np.zeros(case.n_features), "all_zero")
    return np.vstack(rows), kinds


def fmt(value) -> str:
    value = float(value)
    return "" if math.isnan(value) else repr(value)


def write_probes(case: Case, out: Path, P, kinds, expected, margin) -> tuple[str, list[str]]:
    path = out / f"{case.name}.probes.csv"
    columns = ["id"] + [f"f{i}" for i in range(case.n_features)] + ["expected"]
    if case.n_classes:
        columns += [f"proba_{k}" for k in range(case.n_classes)]
        columns += [f"margin_{k}" for k in range(case.n_classes)]
    elif case.task == "binary":
        columns += ["margin"]
    columns += ["kind"]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for i, (row, kind) in enumerate(zip(P, kinds)):
            record = [str(i)] + [fmt(v) for v in row]
            if case.n_classes:
                record += [str(int(np.argmax(expected[i])))]
                record += [fmt(v) for v in expected[i]] + [fmt(v) for v in margin[i]]
            elif case.task == "binary":
                record += [fmt(expected[i, 0]), fmt(margin[i, 0])]
            else:
                record += [fmt(expected[i, 0])]
            writer.writerow(record + [kind])
    return path.name, columns


def output_semantics(case: Case, sigmoid_scale: float | None) -> dict:
    if case.task == "multiclass":
        return {"expected": "class", "transform": "softmax", "argmax_of": "proba_*"}
    if case.task == "binary":
        return {"expected": "probability", "transform": "sigmoid", "sigmoid_scale": sigmoid_scale or 1.0}
    return {"expected": "value", "transform": "identity"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    produced = {"manifest.json"}
    for case in build_cases():
        info = run_xgboost(case, out) if case.vendor == "xgboost" else run_lightgbm(case, out)
        P, kinds, expected, margin = info.pop("_probes")
        probes_file, columns = write_probes(case, out, P, kinds, expected, margin)
        info["files"]["probes"] = probes_file
        produced.update(info["files"].values())
        entry = {
            "name": case.name,
            "vendor": case.vendor,
            "vendor_version": xgb.__version__ if case.vendor == "xgboost" else lgb.__version__,
            "description": case.description,
            "task": case.task,
            "objective": case.params["objective"],
            "booster_objective": info.pop("booster_objective"),
            "n_features": case.n_features,
            "feature_names": case.feature_names,
            "categorical_features": case.categorical,
            "n_classes": case.n_classes or None,
            "num_rounds": case.num_rounds,
            "num_trees": info.pop("num_trees"),
            "params": case.params,
            "files": info.pop("files"),
            "dump_sha256": sha256(out / f"{case.name}.{'json' if case.vendor == 'xgboost' else 'txt'}"),
            "output": output_semantics(case, info.pop("sigmoid_scale", None)),
            "import_options": info.pop("import_options"),
            "tolerance": TOLERANCE[case.vendor],
            "probes": {
                "rows": len(P),
                "columns": columns,
                "kinds": dict(sorted(Counter(kinds).items())),
            },
            **info,
            "notes": case.notes,
            "reference_max_abs_diff": case.max_ref_diff,
        }
        manifest_cases.append(entry)

    for stale in out.iterdir():
        if stale.is_file() and stale.name not in produced and stale.name.startswith(("xgb_", "lgb_")):
            stale.unlink()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/vendor_parity/generate.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
        },
        "conventions": {
            "missing": "empty CSV field; NaN for vendors, NULL in DuckDB",
            "tolerance": "|prediction - expected| <= abs + rel * |expected|",
            "margin": "raw score before the output transform (includes base_margin); "
                      "omitted when the transform is identity",
        },
        "vendor_semantics": {
            "xgboost": [
                "Inputs are cast to float32; numeric splits go 'yes' when float32(x) < split_condition.",
                "NaN follows 'missing'. The dump has no intercept: add base_margin (from the config).",
            ],
            "lightgbm": [
                "Before traversal, inputs with |x| <= 1e-35f become exactly 0.",
                "Numeric splits go left when x <= threshold. NaN becomes 0 unless missing_type is NaN.",
                "Categorical bitset members go left; NaN and negative truncated values go right.",
                "The init score is folded into the first tree; there is no separate intercept.",
            ],
        },
        "cases": manifest_cases,
    }
    (out / "manifest.json").write_text(compact_json(manifest) + "\n")
    return manifest


def compact_json(value, indent: int = 0) -> str:
    """Indented JSON that keeps flat lists and flat objects on a single line."""
    flat = json.dumps(value)
    if not isinstance(value, (dict, list)) or len(flat) + indent <= 120:
        return flat
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(value, list):
        items = [inner + compact_json(v, indent + 2) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    items = [f"{inner}{json.dumps(k)}: {compact_json(v, indent + 2)}" for k, v in value.items()]
    return "{\n" + ",\n".join(items) + "\n" + pad + "}"


def print_summary(manifest: dict, out: Path) -> int:
    print(f"vendor parity fixtures -> {out}")
    print(f"  xgboost {manifest['versions']['xgboost']}, lightgbm {manifest['versions']['lightgbm']}, "
          f"numpy {manifest['versions']['numpy']}, pandas {manifest['versions']['pandas']}")
    header = f"  {'case':<24} {'objective':<28} {'trees':>5} {'probes':>6} {'bytes':>7} {'ref diff':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    total = 0
    for case in manifest["cases"]:
        size = sum((out / name).stat().st_size for name in case["files"].values())
        total += size
        print(f"  {case['name']:<24} {case['booster_objective']:<28} {case['num_trees']:>5} "
              f"{case['probes']['rows']:>6} {size:>7} {case['reference_max_abs_diff']:>9.2g}")
    total += (out / "manifest.json").stat().st_size
    print(f"  total fixture size: {total} bytes ({total / 1024:.1f} KiB, limit {MAX_TOTAL_BYTES // 1024} KiB)")
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--strict-versions", action="store_true",
                        help="fail unless the pinned xgboost/lightgbm versions are installed")
    parser.add_argument("--max-bytes", type=int, default=MAX_TOTAL_BYTES,
                        help="fail if the fixtures exceed this many bytes")
    args = parser.parse_args(argv)

    installed = {"xgboost": xgb.__version__, "lightgbm": lgb.__version__}
    mismatched = {k: v for k, v in installed.items() if v != PINNED[k]}
    if mismatched:
        message = ", ".join(f"{k} {v} (pinned {PINNED[k]})" for k, v in mismatched.items())
        if args.strict_versions:
            print(f"error: version mismatch: {message}", file=sys.stderr)
            return 2
        print(f"warning: version mismatch: {message}", file=sys.stderr)

    manifest = generate(args.out.resolve())
    total = print_summary(manifest, args.out.resolve())
    if total > args.max_bytes:
        print(f"error: fixtures are {total} bytes, over the {args.max_bytes} byte budget", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
