#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate DiCE outputs (long-format CSV written by your flip scripts) for RQ1, RQ2, RQ3,
with support for different methods (random, kdtree, genetic) and organized file structure.
All outputs are suffixed accordingly to avoid overwrites.
"""

import math
import json
from argparse import ArgumentParser
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, PROPOSED_CHANGES

# ----------------------------- config / helpers -----------------------------

MODEL_ABBR = {"SVM": "SVM", "RandomForest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM", "CatBoost": "CatB"}

DEFAULT_GROUPS = [
    ["activemq@0", "activemq@1", "activemq@2", "activemq@3"],
    ["camel@0", "camel@1", "camel@2"],
    ["derby@0", "derby@1"],
    ["groovy@0", "groovy@1"],
    ["hbase@0", "hbase@1"],
    ["hive@0", "hive@1"],
    ["jruby@0", "jruby@1", "jruby@2"],
    ["lucene@0", "lucene@1", "lucene@2"],
    ["wicket@0", "wicket@1"],
]

def _flip_path(project: str, model_type: str, method: str) -> Path:
    """Updated to include method subdirectory"""
    return Path(EXPERIMENTS) / project / model_type / method / "DiCE_all.csv"

def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _feature_cols(df: pd.DataFrame) -> list[str]:
    non_feats = {"test_idx", "candidate_id", "proba0", "proba1", "target"}
    return [c for c in df.columns if c not in non_feats]

def _features_only(df_or_row, label="target"):
    if isinstance(df_or_row, pd.Series):
        return df_or_row[df_or_row.index != label]
    return df_or_row.loc[:, df_or_row.columns != label]

def _generate_all_combinations(values_map: dict[str, list]) -> pd.DataFrame:
    if not values_map:
        return pd.DataFrame()
    cols = list(values_map.keys())
    combos = list(product(*[values_map[c] for c in cols]))
    return pd.DataFrame(combos, columns=cols)

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _cosine_all(df: pd.DataFrame, x: pd.Series) -> list[float]:
    xv = x.values.astype(float)
    return [_cosine_similarity(xv, r.values.astype(float)) for _, r in df.iterrows()]

def _normalized_mahalanobis_distance(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> float:
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0.0

    mu = df.mean()
    sd = df.std(ddof=0).replace(0, 1.0)
    zdf = (df - mu) / sd

    xz = (x[df.columns] - mu) / sd
    yz = (y[df.columns] - mu) / sd

    cov = np.cov(zdf.T)
    inv_cov = np.linalg.pinv(cov) if cov.ndim > 0 else (np.array([[1 / cov]]) if cov != 0 else np.array([[np.inf]]))

    dist = float(mahalanobis(xz.values, yz.values, inv_cov))

    zmin = ((df.min() - mu) / sd).values
    zmax = ((df.max() - mu) / sd).values
    denom = float(mahalanobis(zmin, zmax, inv_cov))
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return dist / denom

def _mahalanobis_all(df: pd.DataFrame, x: pd.Series) -> list[float]:
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return []

    mu = df.mean()
    sd = df.std(ddof=0).replace(0, 1.0)
    zdf = (df - mu) / sd
    xz = (x[df.columns] - mu) / sd

    cov = np.cov(zdf.T)
    inv_cov = np.linalg.pinv(cov) if cov.ndim > 0 else (np.array([[1 / cov]]) if cov != 0 else np.array([[np.inf]]))

    zmin = ((df.min() - mu) / sd).values
    zmax = ((df.max() - mu) / sd).values
    denom = float(mahalanobis(zmin, zmax, inv_cov))
    if denom == 0 or not np.isfinite(denom):
        return []

    out = []
    for _, row in df.iterrows():
        yz = ((row[df.columns] - mu) / sd).values
        d = float(mahalanobis(xz.values, yz, inv_cov))
        out.append(d / denom)
    return out

# ----------------------------- RQ1: Flip rates -----------------------------

def _count_flips_for_project_model(project: str, model_type: str, method: str) -> tuple[int, int, int]:
    """
    Returns (flipped_TPs, unique_testidx_in_file, num_TP).
    A TP is counted as flipped if *any* saved candidate for that test_idx predicts class 0.
    """
    ds = read_dataset()
    if project not in ds:
        return 0, 0, 0

    train, test = ds[project]
    feat_cols = list(_features_only(test).columns)

    model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)

    # True positives (defects predicted as defects)
    tp_df = get_true_positives(model, train, test)
    tp_idx_set = set(tp_df.index.astype(int).tolist())
    tp_count = len(tp_idx_set)

    flips = _safe_read_csv(_flip_path(project, model_type, method))
    if flips is None or flips.empty:
        return 0, 0, tp_count

    if "test_idx" not in flips.columns:
        flips["test_idx"] = range(len(flips))  # fallback

    # consider only genuine TPs
    flips = flips[flips["test_idx"].astype(int).isin(tp_idx_set)]
    if flips.empty:
        return 0, 0, tp_count

    fcols = [c for c in _feature_cols(flips) if c in feat_cols]

    flipped_set = set()
    for test_idx, group in flips.groupby("test_idx"):
        t = int(test_idx)
        X = group[fcols].astype(float).values

        # reconstruct to full feature order if needed
        if set(fcols) != set(feat_cols):
            orig = test.loc[t, feat_cols].astype(float)
            full_rows = []
            for r in X:
                v = orig.copy()
                v[fcols] = r
                full_rows.append(v.values)
            X_use = np.asarray(full_rows)
        else:
            X_use = X[:, [feat_cols.index(c) for c in fcols]] if fcols != feat_cols else X

        preds = model.predict(scaler.transform(X_use))
        if np.any(preds == 0):
            flipped_set.add(t)

    flipped = len(flipped_set)
    computed = flips["test_idx"].astype(int).nunique()
    return flipped, computed, tp_count

def rq1_flip_rates(model_types: list[str], projects: list[str], methods: list[str]):
    ds = read_dataset()
    project_list = list(sorted(ds.keys())) if projects is None else projects
    rows = []
    
    for m in model_types:
        for method in methods:
            for p in project_list:
                flipped, computed, tp_count = _count_flips_for_project_model(p, m, method)
                rate = flipped / tp_count if tp_count > 0 else 0.0
                rows.append([m, method, p, flipped, computed, tp_count, rate])

    df = pd.DataFrame(rows, columns=["Model", "Method", "Project", "Flip", "Computed", "#TP", "Flip%"])
    
    # Group by Model and Method for summary
    model_method_means = (
        df.groupby(["Model", "Method"])[["Flip", "Computed", "#TP", "Flip%"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    out = f"./evaluations/flip_rates_DiCE_all_methods.csv"
    df.to_csv(out, index=False)

    print("\nPer-project flip rates (DiCE):")
    print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))
    print("\nPer-model-method means (DiCE):")
    print(tabulate(model_method_means, headers=model_method_means.columns, tablefmt="github", showindex=False))
    print(f"\nSaved to {out}")

# ----------------------------- RQ2: Plan similarity (if plans exist) -----------------------------

def plan_similarity_dice(project: str, model_type: str, method: str) -> dict[int, dict]:
    """
    For each TP, among its DiCE candidates compute normalized Mahalanobis score
    against the plan's combination space (anchored at first value per feature).
    Keep the **lowest** score per TP.
    """
    plan_path = Path(PROPOSED_CHANGES) / project / model_type / "DiCE" / "plans_all.json"
    flip_path = _flip_path(project, model_type, method)
    if not plan_path.exists():
        return {}

    flips = _safe_read_csv(flip_path)
    if flips is None or flips.empty:
        return {}

    with open(plan_path, "r") as f:
        plans = json.load(f)

    ds = read_dataset()
    train, test = ds[project]
    feat_cols = list(_features_only(test).columns)
    model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)

    if "test_idx" not in flips.columns:
        return {}

    res = {}
    for test_idx, g in flips.groupby("test_idx"):
        t = int(test_idx)
        if str(t) not in plans:
            continue

        original = test.loc[t, feat_cols]
        _ = model.predict(scaler.transform([original.values.astype(float)]))  # sanity

        best = None
        for _, cand in g.iterrows():
            changed = {}
            for f, vals in plans[str(t)].items():
                if f not in feat_cols:
                    continue
                if not math.isclose(float(cand[f]), float(original[f]), rel_tol=1e-9, abs_tol=1e-9):
                    changed[f] = [float(v) for v in vals]
            if not changed:
                continue

            flipped_vec = pd.Series({f: float(cand[f]) for f in changed})
            min_changes = pd.Series([changed[f][0] for f in changed], index=list(changed.keys()), dtype=float)
            combi = _generate_all_combinations(changed)
            if combi.shape[1] == 0:
                continue

            score = _normalized_mahalanobis_distance(combi, flipped_vec, min_changes)
            if best is None or score < best:
                best = score

        if best is not None:
            res[t] = {"score": float(best)}

    return res

def rq2_similarity(model_types: list[str], projects: list[str], methods: list[str]):
    ds = read_dataset()
    project_list = list(sorted(ds.keys())) if projects is None else projects
    Path("./evaluations/similarities").mkdir(parents=True, exist_ok=True)

    for m in model_types:
        for method in methods:
            all_scores = pd.DataFrame()
            for p in project_list:
                d = plan_similarity_dice(p, m, method)
                if not d:
                    continue
                dfp = pd.DataFrame(d).T
                dfp["project"] = p
                dfp["explainer"] = "DiCE"
                dfp["method"] = method
                dfp["model"] = MODEL_ABBR.get(m, m)
                all_scores = pd.concat([all_scores, dfp], axis=0)
            if not all_scores.empty:
                out = f"./evaluations/similarities/{MODEL_ABBR.get(m, m)}_DiCE_{method}.csv"
                all_scores.to_csv(out)

# ----------------------------- RQ3: Feasibility vs historical deltas -----------------------------

def _flip_feasibility_for_group(
    project_list,
    model_type,
    method,
    *,
    distance="mahalanobis",
    min_rows=5,
    require_all_nonzero=True,
    selection_strategy="best"  # "best" or "first"
):
    ds = read_dataset()

    # Build historical delta pool
    pool = pd.DataFrame()
    for p in project_list:
        train, test = ds[p]
        common = train.index.intersection(test.index)
        deltas = test.loc[common, test.columns != "target"] - \
                 train.loc[common, train.columns != "target"]
        pool = pd.concat([pool, deltas], axis=0)

    results_rows, totals, cannot = [], 0, 0

    for p in project_list:
        flip_path = _flip_path(p, model_type, method)
        flips = _safe_read_csv(flip_path)
        if flips is None or flips.empty or "test_idx" not in flips.columns:
            continue

        train, test = ds[p]
        feat_cols = [c for c in test.columns if c != "target"]

        for test_idx, g in flips.groupby("test_idx"):
            t = int(test_idx)
            totals += 1
            if t not in test.index:
                cannot += 1
                continue

            original = test.loc[t, feat_cols]
            best = None

            # Filter candidates based on selection strategy
            if selection_strategy == "first":
                # Only use candidate_id=0 (first generated CF)
                if "candidate_id" in g.columns:
                    g = g[g["candidate_id"] == 0]
                else:
                    g = g.iloc[:1]  # fallback: take first row
        
            for _, cand in g.iterrows():
                changed = {
                    f: float(cand[f]) - float(original[f])
                    for f in feat_cols if f in g.columns
                    if not math.isclose(float(cand[f]), float(original[f]), rel_tol=1e-9, abs_tol=1e-9)
                }
                if not changed:
                    continue

                names = list(changed.keys())
                x = pd.Series(changed, index=names, dtype=float)

                sub = pool[names].dropna()
                sub = sub.loc[(sub != 0).all(axis=1)] if require_all_nonzero else sub.loc[(sub != 0).any(axis=1)]

                if distance == "mahalanobis":
                    if len(sub) < min_rows:
                        continue
                    dists = _mahalanobis_all(sub, x)
                else:
                    if len(sub) == 0:
                        continue
                    dists = _cosine_all(sub, x)

                if len(dists) == 0:
                    continue

                mean_d = float(np.mean(dists))
                min_d  = float(np.min(dists))
                max_d  = float(np.max(dists))

                # Store candidate data for selection
                candidate_data = {
                    "mean": mean_d,   # selection criterion for "best"
                    "min":  min_d,    # minimum distance
                    "max":  max_d,    # maximum distance
                }

                if selection_strategy == "best":
                    if (best is None) or (mean_d < best["mean"]):
                        best = candidate_data
                else:  # "first" - just take the first (and only) candidate
                    best = candidate_data
                    break  # exit after first candidate

            if best is not None:
                results_rows.append(best)
            else:
                cannot += 1

    return results_rows, totals, cannot

def rq3_feasibility(
    model_types: list[str],
    projects: list[str],
    methods: list[str],
    distance: str = "mahalanobis",
    use_default_groups: bool = True,
    min_rows: int = 5,
    require_all_nonzero: bool = True,
    selection_strategy: str = "best"
):
    # Always include strategy in filename for clarity
    strategy_suffix = f"_{selection_strategy}"
    Path(f"./evaluations/feasibility/{distance}").mkdir(parents=True, exist_ok=True)
    
    ds = read_dataset()
    all_projects = list(sorted(ds.keys()))
    groups = DEFAULT_GROUPS if use_default_groups else [[p] for p in (projects or all_projects)]
    
    summary = []
    
    for m in model_types:
        for method in methods:
            all_rows, totals, cannots = [], 0, 0
            for g in groups:
                rows, tot, cannot = _flip_feasibility_for_group(
                    g, m, method,
                    distance=distance,
                    min_rows=min_rows,
                    require_all_nonzero=require_all_nonzero,
                    selection_strategy=selection_strategy
                )
                totals += tot
                cannots += cannot
                all_rows.extend(rows)

            if all_rows:
                df = pd.DataFrame(all_rows)
                # Include method and strategy in filename
                out = f"./evaluations/feasibility/{distance}/{MODEL_ABBR.get(m, m)}_DiCE_{method}{strategy_suffix}.csv"
                df.to_csv(out, index=False)

                # Use the same column names as original format for summary
                mean_min = df["min"].mean()
                mean_max = df["max"].mean()
                mean_mean = df["mean"].mean()
                summary.append([m, method, "DiCE", mean_min, mean_max, mean_mean])

            print(f"[{m}/{method}/{selection_strategy}] totals={totals}, cannot={cannots}")

    if summary:
        s = pd.DataFrame(summary, columns=["Model", "Method", "Explainer", "Min", "Max", "Mean"])
        s.to_csv(f"./evaluations/feasibility_{distance}_DiCE_all_methods{strategy_suffix}.csv", index=False)
        print(f"\nFeasibility summary ({selection_strategy} candidate per TP):")
        print(tabulate(s, headers=s.columns, tablefmt="github", showindex=False))
        print(f"Total: {totals}, Cannot: {cannots} ({cannots/totals*100:.2f}%)" if totals > 0 else "No data processed")
    else:
        print("No feasibility results to summarize.")

# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="Evaluate DiCE counterfactuals across different methods")
    ap.add_argument("--rq1", action="store_true", help="Flip rates")
    ap.add_argument("--rq2", action="store_true", help="Plan similarity (needs plans_all.json)")
    ap.add_argument("--rq3", action="store_true", help="Feasibility vs historical deltas")
    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated DiCE methods: random,kdtree,genetic (default: random)")
    ap.add_argument("--projects", type=str, default="all",
                    help="Project name(s) or 'all' (space/comma separated allowed)")
    ap.add_argument("--distance", type=str, default="mahalanobis", choices=["mahalanobis", "cosine"],
                    help="Distance metric for RQ3")
    ap.add_argument("--use_default_groups", action="store_true",
                    help="Use predefined release groups for RQ3")
    ap.add_argument("--selection_strategy", type=str, choices=["best", "first"], default="best",
                    help="Selection strategy for RQ3: 'best' (lowest mean distance) or 'first' (first candidate)")
    args = ap.parse_args()

    model_types = [m.strip() for m in args.models.replace(",", " ").split() if m.strip()]
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]

    # Validate methods
    valid_methods = ["random", "kdtree", "genetic"]
    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        print(f"ERROR: Invalid methods: {invalid_methods}")
        print(f"Valid methods are: {valid_methods}")
        return

    all_ds = read_dataset()
    if args.projects == "all":
        project_list = list(sorted(all_ds.keys()))
    else:
        project_list = [p.strip() for p in args.projects.replace(",", " ").split() if p.strip()]

    print(f"Evaluating {len(model_types)} models × {len(methods)} methods × {len(project_list)} projects")
    print(f"Models: {model_types}")
    print(f"Methods: {methods}")
    print(f"Projects: {project_list[:3]}{'...' if len(project_list) > 3 else ''}")
    print()

    if args.rq1:
        print("=== Running RQ1: Flip Rates ===")
        rq1_flip_rates(model_types, project_list, methods)

    if args.rq2:
        print("\n=== Running RQ2: Plan Similarity ===")
        rq2_similarity(model_types, project_list, methods)

    if args.rq3:
        print(f"\n=== Running RQ3: Feasibility (strategy: {args.selection_strategy}) ===")
        rq3_feasibility(
            model_types, project_list, methods,
            distance=args.distance, 
            use_default_groups=args.use_default_groups,
            selection_strategy=args.selection_strategy
        )

if __name__ == "__main__":
    main()