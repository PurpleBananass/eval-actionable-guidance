#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit RQ3 evaluation-stage counts: Legacy explainer pipeline vs New (DiCE) pipeline.

What this script reports (per model × group × pipeline):
  - files loaded, rows loaded, distinct test_idx
  - invalid/missing test_idx rows
  - TPs with candidate rows but:
      * no actual changes (by tolerance)
      * filtered out by nonzero-delta rule on the pool
      * too few pool rows for Mahalanobis (min_rows)
      * distances not computed (empty after guards)
  - selected TPs (after 'first'/'best' selection)

Outputs:
  - Pretty console summaries
  - CSVs with raw stats: ./evaluations/debug/audit_rq3_eval/*.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler

# Your helpers (must exist in your repo)
from data_utils import read_dataset, get_model
from hyparams import EXPERIMENTS

# ----------------------------- constants -----------------------------

MODEL_ABBR = {
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "SVM": "SVM",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}
LEGACY_EXPLAINERS = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]

# Legacy release groups
LEGACY_GROUPS = [
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

# ----------------------------- IO helpers -----------------------------

def _lenient_read_csv(path: Path) -> pd.DataFrame | None:
    """Load CSV (legacy-friendly). Return None on any failure/missing."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, index_col=0)
        except Exception:
            return None

def _read_flips_legacy(project: str, model_type: str, explainer: str) -> pd.DataFrame | None:
    """./experiments/{project}/{model}/{explainer}_all.csv"""
    path = Path(EXPERIMENTS) / project / model_type / f"{explainer}_all.csv"
    df = _lenient_read_csv(path)
    if df is None:
        return None
    if "test_idx" not in df.columns:
        df = df.reset_index().rename(columns={"index": "test_idx"})
    if "candidate_id" not in df.columns:
        df["candidate_id"] = 0
    return df

def _read_flips_new(project: str, model_type: str, method: str, hist_seed: bool) -> pd.DataFrame | None:
    """./experiments/{project}/{model}/{method}/DiCE_all[_hist_seed].csv"""
    filename = "DiCE_all_hist_seed.csv" if hist_seed else "DiCE_all.csv"
    path = Path(EXPERIMENTS) / project / model_type / method / filename
    df = _lenient_read_csv(path)
    if df is None:
        return None
    if "test_idx" not in df.columns:
        df = df.reset_index().rename(columns={"index": "test_idx"})
    if "candidate_id" not in df.columns:
        df["candidate_id"] = 0
    return df

# ----------------------------- math helpers -----------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _cosine_all(df: pd.DataFrame, x: pd.Series) -> List[float]:
    xv = x.values.astype(float)
    return [_cosine_similarity(xv, r.values.astype(float)) for _, r in df.iterrows()]

def _mahalanobis_all(df: pd.DataFrame, x: pd.Series) -> List[float]:
    # Drop constant columns
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return []
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, 1.0)
    zdf = (df - mu) / sd
    xz = (x[df.columns] - mu) / sd

    cov = np.cov(zdf.T)
    inv = np.linalg.pinv(cov) if cov.ndim > 0 else (np.array([[1 / cov]]) if cov != 0 else np.array([[np.inf]]))

    zmin = ((df.min() - mu) / sd).values
    zmax = ((df.max() - mu) / sd).values
    denom = float(mahalanobis(zmin, zmax, inv))
    if denom == 0 or not np.isfinite(denom):
        return []

    out = []
    for _, row in df.iterrows():
        yz = ((row[df.columns] - mu) / sd).values
        d = float(mahalanobis(xz.values, yz, inv))
        out.append(d / denom)
    return out

# ----------------------------- evaluation audit -----------------------------

def _build_pool_for_group(ds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], projects: List[str]) -> pd.DataFrame:
    pool = pd.DataFrame()
    for p in projects:
        train, test = ds[p]
        common = train.index.intersection(test.index)
        deltas = test.loc[common, test.columns != "target"] - \
                 train.loc[common, train.columns != "target"]
        pool = pd.concat([pool, deltas], axis=0)
    return pool

def _eval_one_flips_df(
    flips: pd.DataFrame,
    ds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    project: str,
    model_type: str,
    pool: pd.DataFrame,
    *,
    distance: str,
    selection: str,
    min_rows: int,
    require_all_nonzero: bool,
    change_tol: float
) -> Tuple[List[dict], dict]:
    """Evaluate a single flips file and return (selected_rows, stats)."""
    stats = dict(
        files_loaded=1,
        rows_loaded=len(flips),
        distinct_test_idx=0,
        invalid_test_idx=0,
        tps=0,
        no_candidates=0,
        no_change=0,
        pool_filtered=0,
        too_few_rows=0,
        distances_empty=0,
        selected=0,
    )
    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    selected_rows = []
    grouped = flips.groupby("test_idx")
    stats["distinct_test_idx"] = len(grouped)
    stats["tps"] = len(grouped)

    for test_idx, g in grouped:
        try:
            t = int(test_idx)
        except Exception:
            stats["invalid_test_idx"] += 1
            continue
        if t not in test.index:
            stats["invalid_test_idx"] += 1
            continue

        original = test.loc[t, feat_cols]
        if selection == "first":
            g = g[g["candidate_id"] == 0] if "candidate_id" in g.columns else g.iloc[:1]

        best = None
        had_any_candidate = False

        for _, cand in g.iterrows():
            had_any_candidate = True
            names = [f for f in feat_cols if f in g.columns]
            # changed features by tolerance
            changed = {
                f: float(cand[f]) - float(original[f])
                for f in names
                if abs(float(cand[f]) - float(original[f])) > change_tol
            }
            if not changed:
                stats["no_change"] += 1
                continue

            sub = pool[list(changed.keys())].dropna()
            if require_all_nonzero:
                sub = sub.loc[(sub != 0).all(axis=1)]
            else:
                sub = sub.loc[(sub != 0).any(axis=1)]

            if distance == "mahalanobis":
                if len(sub) < min_rows:
                    stats["too_few_rows"] += 1
                    continue
                dists = _mahalanobis_all(sub, pd.Series(changed, dtype=float))
            else:
                if len(sub) == 0:
                    stats["pool_filtered"] += 1
                    continue
                dists = _cosine_all(sub, pd.Series(changed, dtype=float))

            if not dists:
                stats["distances_empty"] += 1
                continue

            cand_data = {
                "test_idx": t,
                "min": float(np.min(dists)),
                "max": float(np.max(dists)),
                "mean": float(np.mean(dists)),
            }
            if selection == "best":
                if (best is None) or (cand_data["mean"] < best["mean"]):
                    best = cand_data
            else:
                best = cand_data
                break

        if not had_any_candidate:
            stats["no_candidates"] += 1

        if best is not None:
            selected_rows.append(best)
            stats["selected"] += 1

    return selected_rows, stats

def audit_pipeline(
    *,
    pipeline: str,  # "legacy" or "dice"
    model_type: str,
    groups: List[List[str]],
    methods: List[str] | None,
    explainers: List[str] | None,
    hist_seed: bool,
    distance: str,
    selection: str,
    min_rows: int,
    require_all_nonzero: bool,
    change_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run evaluation audit for a whole pipeline and return:
      - rows_df: per-TP selected rows with distances
      - stats_df: per (group, project/file) stats
    """
    ds = read_dataset()
    rows_out = []
    stats_out = []

    for gi, group in enumerate(groups, start=1):
        pool = _build_pool_for_group(ds, group)

        for p in group:
            if p not in ds:
                continue

            if pipeline == "legacy":
                for exp in (explainers or []):
                    flips = _read_flips_legacy(p, model_type, exp)
                    files_loaded = 1 if flips is not None else 0
                    if flips is None or flips.empty:
                        stats_out.append(dict(pipeline=pipeline, model=model_type, group=gi, project=p,
                                              tag=exp, files_loaded=0, rows_loaded=0, distinct_test_idx=0,
                                              invalid_test_idx=0, tps=0, no_candidates=0, no_change=0,
                                              pool_filtered=0, too_few_rows=0, distances_empty=0, selected=0))
                        continue
                    selected, stats = _eval_one_flips_df(
                        flips, ds, p, model_type, pool,
                        distance=distance,
                        selection=selection,
                        min_rows=min_rows,
                        require_all_nonzero=require_all_nonzero,
                        change_tol=change_tol,
                    )
                    for r in selected:
                        r.update(dict(pipeline=pipeline, model=model_type, group=gi, project=p, tag=exp))
                    rows_out.extend(selected)
                    stats_out.append(dict(pipeline=pipeline, model=model_type, group=gi, project=p, tag=exp, **stats))

            else:  # dice
                for method in (methods or []):
                    flips = _read_flips_new(p, model_type, method, hist_seed)
                    if flips is None or flips.empty:
                        stats_out.append(dict(pipeline=pipeline, model=model_type, group=gi, project=p,
                                              tag=method, files_loaded=0, rows_loaded=0, distinct_test_idx=0,
                                              invalid_test_idx=0, tps=0, no_candidates=0, no_change=0,
                                              pool_filtered=0, too_few_rows=0, distances_empty=0, selected=0))
                        continue
                    selected, stats = _eval_one_flips_df(
                        flips, ds, p, model_type, pool,
                        distance=distance,
                        selection=selection,
                        min_rows=min_rows,
                        require_all_nonzero=require_all_nonzero,
                        change_tol=change_tol,
                    )
                    for r in selected:
                        r.update(dict(pipeline=pipeline, model=model_type, group=gi, project=p, tag=method))
                    rows_out.extend(selected)
                    stats_out.append(dict(pipeline=pipeline, model=model_type, group=gi, project=p, tag=method, **stats))

    return pd.DataFrame(rows_out), pd.DataFrame(stats_out)

# ----------------------------- pretty printing -----------------------------

def _pp(title: str, df: pd.DataFrame):
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("(no rows)")
        return
    try:
        from tabulate import tabulate
        print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))
    except Exception:
        print(df.to_string(index=False))

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Audit RQ3 evaluation pipeline differences (legacy vs new/DiCE).")
    ap.add_argument("--models", type=str, default="RandomForest,XGBoost,SVM,LightGBM,CatBoost",
                    help="Comma/space-separated model types")
    ap.add_argument("--methods", type=str, default="random",
                    help="DiCE methods (comma/space separated)")
    ap.add_argument("--hist_seed", action="store_true", help="Use _hist_seed DiCE files")
    ap.add_argument("--no_hist_seed", dest="hist_seed", action="store_false")
    ap.set_defaults(hist_seed=True)

    # Evaluation knobs (match your current evaluator or legacy by changing defaults)
    ap.add_argument("--distance", type=str, default="mahalanobis", choices=["mahalanobis","cosine"])
    ap.add_argument("--selection", type=str, default="best", choices=["best","first"],
                    help="Candidate selection per TP")
    ap.add_argument("--min_rows", type=int, default=5, help="Min rows for Mahalanobis")
    ap.add_argument("--require_all_nonzero", action="store_true",
                    help="Require all changed-feature deltas nonzero in pool rows (strict)")
    ap.add_argument("--change_tol", type=float, default=0.0,
                    help="Change tolerance (0.0 ~ legacy `!=`, larger ~= strict `isclose`)")

    args = ap.parse_args()
    model_types = [m.strip() for m in args.models.replace(",", " ").split() if m.strip()]
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]

    outdir = Path("./evaluations/debug/audit_rq3_eval")
    outdir.mkdir(parents=True, exist_ok=True)

    # Use legacy groups (as discussed, groups are the same)
    groups = LEGACY_GROUPS

    all_stats = []
    for model_type in model_types:
        # LEGACY
        legacy_rows, legacy_stats = audit_pipeline(
            pipeline="legacy",
            model_type=model_type,
            groups=groups,
            methods=None,
            explainers=LEGACY_EXPLAINERS,
            hist_seed=args.hist_seed,
            distance=args.distance,
            selection=args.selection,
            min_rows=args.min_rows,
            require_all_nonzero=args.require_all_nonzero,
            change_tol=args.change_tol,
        )
        legacy_rows.to_csv(outdir / f"legacy_rows_{MODEL_ABBR.get(model_type, model_type)}.csv", index=False)
        legacy_stats.to_csv(outdir / f"legacy_stats_{MODEL_ABBR.get(model_type, model_type)}.csv", index=False)

        # DICE
        dice_rows, dice_stats = audit_pipeline(
            pipeline="dice",
            model_type=model_type,
            groups=groups,
            methods=methods,
            explainers=None,
            hist_seed=args.hist_seed,
            distance=args.distance,
            selection=args.selection,
            min_rows=args.min_rows,
            require_all_nonzero=args.require_all_nonzero,
            change_tol=args.change_tol,
        )
        dice_rows.to_csv(outdir / f"dice_rows_{MODEL_ABBR.get(model_type, model_type)}.csv", index=False)
        dice_stats.to_csv(outdir / f"dice_stats_{MODEL_ABBR.get(model_type, model_type)}.csv", index=False)

        # Summaries (per model)
        legacy_sum = legacy_stats.groupby("tag", as_index=False)[["rows_loaded","distinct_test_idx","selected",
                                                                  "no_candidates","no_change","pool_filtered",
                                                                  "too_few_rows","distances_empty","invalid_test_idx"]].sum()
        dice_sum   = dice_stats.groupby("tag", as_index=False)[["rows_loaded","distinct_test_idx","selected",
                                                                "no_candidates","no_change","pool_filtered",
                                                                "too_few_rows","distances_empty","invalid_test_idx"]].sum()

        _pp(f"{model_type} — LEGACY (per explainer)", legacy_sum.rename(columns={"tag":"explainer"}))
        _pp(f"{model_type} — DICE (per method)", dice_sum.rename(columns={"tag":"method"}))

        # Combined comparison at model level
        legacy_total_selected = int(legacy_stats["selected"].sum()) if not legacy_stats.empty else 0
        dice_total_selected   = int(dice_stats["selected"].sum()) if not dice_stats.empty else 0
        print(f"\n>>> MODEL {model_type}: selected TPs — LEGACY={legacy_total_selected} vs DICE={dice_total_selected}")

        all_stats.append(pd.DataFrame({
            "model": [model_type],
            "legacy_selected": [legacy_total_selected],
            "dice_selected": [dice_total_selected],
            "diff": [legacy_total_selected - dice_total_selected]
        }))

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        _pp("Overall comparison (selected TPs)", combined)
        combined.to_csv(outdir / "overall_comparison.csv", index=False)

    print(f"\nSaved audit CSVs under: {outdir.resolve()}")

if __name__ == "__main__":
    main()
