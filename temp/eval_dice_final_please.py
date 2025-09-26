#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate DiCE counterfactuals generated with per-instance top-K LIME features.

Outputs:
  - RQ1 (flip rates): ./evaluations/flip_rates_DiCE_topkLIME.csv
  - RQ3 (feasibility vs history):
        ./evaluations/feasibility/{distance}/{MODEL}_DiCE_{method}_{selection}.csv
        ./evaluations/feasibility_{distance}_DiCE_all_methods_{selection}.csv
"""

import math
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS

# ----------------------------- config -----------------------------

MODEL_ABBR = {
    "SVM": "SVM",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

# release groups (unchanged from earlier usage)
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

# ----------------------------- utils -----------------------------

def _flip_path(project: str, model_type: str, method: str) -> Path:
    """Path to generator output."""
    return Path(EXPERIMENTS) / project / model_type / method / "DiCE_all.csv"

def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        return df
    except Exception:
        return None

def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in {"test_idx", "candidate_id", "proba0", "proba1", "target"}]

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _cosine_distance_all(df: pd.DataFrame, x: pd.Series) -> List[float]:
    xv = x.values.astype(float)
    # distance = 1 - cosine similarity ∈ [0, 2]
    return [1.0 - _cosine_similarity(xv, r.values.astype(float)) for _, r in df.iterrows()]

def _mahalanobis_all_normalized(df: pd.DataFrame, x: pd.Series) -> List[float]:
    """Normalized by distance between per-dim min and max in z-space."""
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

def _count_flips(project: str, model_type: str, method: str) -> Tuple[int, int, int]:
    """
    Returns (flipped_TPs, unique_testidx_in_file, num_TP).
    A TP is counted as flipped if any saved candidate for that test_idx predicts class 0.
    """
    ds = read_dataset()
    if project not in ds:
        return 0, 0, 0

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    # model + scaler (mimic generator)
    base_model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)

    def _predict(X: np.ndarray) -> np.ndarray:
        Xs = scaler.transform(X)
        return base_model.predict(Xs)

    # true positives (actual 1 AND predicted 1)
    tp_df = get_true_positives(base_model, train, test)
    tp_ids = set(tp_df.index.astype(int).tolist())
    tp_count = len(tp_ids)

    flips_path = _flip_path(project, model_type, method)
    flips = _safe_read_csv(flips_path)
    if flips is None or flips.empty:
        return 0, 0, tp_count

    if "test_idx" not in flips.columns:
        return 0, 0, tp_count

    # consider only genuine TPs
    flips = flips[flips["test_idx"].astype(int).isin(tp_ids)]
    if flips.empty:
        return 0, 0, tp_count

    fcols = [c for c in _feature_cols(flips) if c in feat_cols]

    flipped_set = set()
    for tid, g in flips.groupby("test_idx"):
        t = int(tid)
        X = g[fcols].astype(float).values
        # Reorder columns to feat_cols
        X_full = np.zeros((len(X), len(feat_cols)))
        for j, c in enumerate(feat_cols):
            if c in fcols:
                X_full[:, j] = X[:, fcols.index(c)]
            else:
                # backfill with the original value (shouldn’t happen with generator)
                X_full[:, j] = float(test.loc[t, c])
        preds = _predict(X_full)
        if np.any(preds == 0):
            flipped_set.add(t)

    flipped = len(flipped_set)
    computed = flips["test_idx"].astype(int).nunique()
    return flipped, computed, tp_count

def rq1_flip_rates(model_types: List[str], projects: List[str], methods: List[str]):
    ds = read_dataset()
    proj_list = projects or list(sorted(ds.keys()))
    rows = []
    for m in model_types:
        for method in methods:
            for p in proj_list:
                f, comp, tps = _count_flips(p, m, method)
                rate = f / tps if tps > 0 else 0.0
                rows.append([m, method, p, f, comp, tps, rate])

    df = pd.DataFrame(rows, columns=["Model", "Method", "Project", "Flip", "Computed", "#TP", "Flip%"])
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    out = "./evaluations/flip_rates_DiCE_topkLIME.csv"
    df.to_csv(out, index=False)

    means = (
        df.groupby(["Model", "Method"])[["Flip", "Computed", "#TP", "Flip%"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    print("\nPer-project flip rates (DiCE, top-K LIME):")
    print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))
    print("\nPer-model-method means:")
    print(tabulate(means, headers=means.columns, tablefmt="github", showindex=False))
    print(f"\nSaved to {out}")

# ----------------------------- RQ3: Feasibility vs history -----------------------------

def _flip_feasibility_for_group(
    project_list: List[str],
    model_type: str,
    method: str,
    *,
    distance: str = "mahalanobis",
    selection_strategy: str = "best",
    nonzero_strict: bool = True,
    tol: float = 1e-7
) -> Tuple[List[Dict[str, float]], int, int]:
    """
    For each TP with candidates, compute distance between the actual change vector (CF - original)
    and historical release deltas restricted to the same changed feature set.
    Returns (rows, totals, cannot) where rows have keys min/max/mean for the chosen candidate.
    """
    ds = read_dataset()

    # pool of historical deltas across the group
    pool = pd.DataFrame()
    for proj in project_list:
        train, test = ds[proj]
        common = train.index.intersection(test.index)
        deltas = test.loc[common, test.columns != "target"] - \
                 train.loc[common, train.columns != "target"]
        pool = pd.concat([pool, deltas], axis=0)

    totals, cannot = 0, 0
    rows_out: List[Dict[str, float]] = []

    for proj in project_list:
        flips = _safe_read_csv(_flip_path(proj, model_type, method))
        if flips is None or flips.empty:
            continue

        flips = flips.dropna(axis=0, how="any")
        train, test = ds[proj]
        feat_cols = [c for c in test.columns if c != "target"]

        if "test_idx" in flips.columns:
            group_iter = flips.groupby("test_idx")
        else:
            group_iter = ((idx, flips.loc[[idx]]) for idx in flips.index.unique())

        for tid, g in group_iter:
            try:
                t = int(tid)
                original = test.loc[t, feat_cols].astype(float)
            except Exception:
                continue

            totals += 1
            best: Dict[str, float] | None = None

            # choose first or scan all
            if selection_strategy == "first":
                if "candidate_id" in g.columns:
                    g = g.sort_values("candidate_id").iloc[:1]
                else:
                    g = g.iloc[:1]

            for _, cand in g.iterrows():
                # actual change vector (use tolerance to ignore float noise)
                changed = {
                    f: float(cand[f]) - float(original[f])
                    for f in feat_cols if f in g.index or f in g.keys()
                    if not math.isclose(float(cand[f]), float(original[f]), rel_tol=0, abs_tol=tol)
                }
                if not changed:
                    continue

                names = list(changed.keys())
                x = pd.Series(changed, index=names, dtype=float)

                # restrict pool to same features
                sub = pool[names].dropna()
                if nonzero_strict:
                    sub = sub.loc[(sub != 0).all(axis=1)]

                if distance == "mahalanobis":
                    # at least d+1 rows for stable covariance (loose check)
                    if len(sub) <= len(names):
                        continue
                    dists = _mahalanobis_all_normalized(sub, x)
                else:
                    if len(sub) == 0:
                        continue
                    dists = _cosine_distance_all(sub, x)

                if not dists:
                    continue

                res = {
                    "min": float(np.min(dists)),
                    "max": float(np.max(dists)),
                    "mean": float(np.mean(dists)),
                }
                if selection_strategy == "best":
                    if best is None or res["mean"] < best["mean"]:
                        best = res
                else:
                    best = res
                    break

            if best is not None:
                rows_out.append(best)
            else:
                cannot += 1

    return rows_out, totals, cannot

def rq3_feasibility(
    model_types: List[str],
    projects: List[str],
    methods: List[str],
    *,
    distance: str = "mahalanobis",
    use_default_groups: bool = True,
    selection_strategy: str = "best",
    nonzero_strict: bool = True
):
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
                    selection_strategy=selection_strategy,
                    nonzero_strict=nonzero_strict
                )
                totals += tot
                cannots += cannot
                all_rows.extend(rows)

            if all_rows:
                df = pd.DataFrame(all_rows)
                out = f"./evaluations/feasibility/{distance}/{MODEL_ABBR.get(m, m)}_DiCE_{method}_{selection_strategy}.csv"
                df.to_csv(out, index=False)
                summary.append([
                    m, method, "DiCE",
                    float(df["min"].mean()),
                    float(df["max"].mean()),
                    float(df["mean"].mean())
                ])

            print(f"[{m}/{method}/{selection_strategy}] totals={totals}, cannot={cannots}")

    # if summary:
    if True:
        s = pd.DataFrame(summary, columns=["Model", "Method", "Explainer", "Min", "Max", "Mean"])
        s_out = f"./evaluations/feasibility_{distance}_DiCE_all_methods_{selection_strategy}.csv"
        s.to_csv(s_out, index=False)
        print("\nFeasibility summary:")
        print(tabulate(s, headers=s.columns, tablefmt="github", showindex=False))
        print(f"\nSaved to {s_out}")

# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="Evaluate DiCE CFs generated with per-instance top-K LIME features")
    ap.add_argument("--rq1", action="store_true", help="Flip rates")
    ap.add_argument("--rq3", action="store_true", help="Feasibility vs historical deltas")
    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated DiCE methods: random,kdtree,genetic")
    ap.add_argument("--projects", type=str, default="all",
                    help="Project name(s) or 'all' (space/comma separated allowed)")
    ap.add_argument("--distance", type=str, default="mahalanobis",
                    choices=["mahalanobis", "cosine"], help="Distance metric for RQ3")
    ap.add_argument("--use_default_groups", action="store_true",
                    help="Use predefined release groups for RQ3 (recommended)")
    ap.add_argument("--selection_strategy", type=str, choices=["best", "first"], default="best",
                    help="Pick per-instance candidate by 'best' (lowest mean distance) or 'first'")
    ap.add_argument("--nonzero_strict", action="store_true",
                    help="Require all features in a historical delta row to be nonzero (default True)")
    args = ap.parse_args()

    # parse lists
    model_types = [m.strip() for m in args.models.replace(",", " ").split() if m.strip()]
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]

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
        print("=== Running RQ1: Flip Rates (DiCE, top-K LIME) ===")
        rq1_flip_rates(model_types, project_list, methods)

    if args.rq3:
        print(f"\n=== Running RQ3: Feasibility (distance={args.distance}, selection={args.selection_strategy}) ===")
        rq3_feasibility(
            model_types, project_list, methods,
            distance=args.distance,
            use_default_groups=args.use_default_groups,
            selection_strategy=args.selection_strategy,
            nonzero_strict=args.nonzero_strict
        )

if __name__ == "__main__":
    main()
