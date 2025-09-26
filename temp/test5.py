#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare RQ1 raw counts (TPs, flipped TPs) with RQ3 Mahalanobis points.

Usage (example):
  python compare_rq1_rq3.py \
    --feas_csv /home/joony/saner/jit2/eval-actionable-guidance/evaluations/feasibility/XGBoost_TimeLIME.csv \
    --model XGBoost \
    --explainer TimeLIME \
    --projects all
"""

from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# your repo utilities
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS

NON_FEATS = {"test_idx", "candidate_id", "proba0", "proba1", "target"}

def _ensure_test_idx_column(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    if "test_idx" in df.columns:
        if df.index.name == "test_idx":
            df = df.reset_index(drop=True)
    else:
        if df.index.nlevels > 1:
            df = df.reset_index()
            first_level_name = df.columns[0]
            df = df.rename(columns={first_level_name: "test_idx"})
        else:
            idx_name = df.index.name if df.index.name is not None else "index"
            df = df.reset_index().rename(columns={idx_name: "test_idx"})
    df["test_idx"] = pd.to_numeric(df["test_idx"], errors="coerce")
    df = df.dropna(subset=["test_idx"]).copy()
    df["test_idx"] = df["test_idx"].astype(int)
    if df.index.name == "test_idx":
        df = df.reset_index(drop=True)
    return df

def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATS]

def _flip_path(project: str, model_type: str, explainer: str) -> Path:
    # matches your earlier convention: EXPERIMENTS/<project>/<model>/<explainer>_all.csv
    return Path(EXPERIMENTS) / project / model_type / f"{explainer}_all.csv"

def _load_flips(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            df = pd.read_csv(path)
        return _ensure_test_idx_column(df)
    except Exception:
        return None

def recompute_rq1_counts(model_type: str, explainer: str, projects: list[str]) -> tuple[int, int]:
    """
    Returns (flipped_TPs, total_TPs) aggregated across the given projects.
    """
    ds = read_dataset()
    flipped_total = 0
    tp_total = 0

    for p in projects:
        if p not in ds:
            continue
        train, test = ds[p]
        feat_cols = [c for c in test.columns if c != "target"]

        # model + scaler
        model = get_model(p, model_type)
        scaler = StandardScaler().fit(train[feat_cols].values)

        # true positives for denominator
        tp_df = get_true_positives(model, train, test)
        tp_idx_set = set(tp_df.index.astype(int).tolist())
        tp_total += len(tp_idx_set)

        # flips file
        fpath = _flip_path(p, model_type, explainer)
        flips = _load_flips(fpath)
        if flips is None or flips.empty:
            continue

        # restrict to TPs only
        flips = flips[flips["test_idx"].astype(int).isin(tp_idx_set)]
        if flips.empty:
            continue

        # features present in flips & test
        fcols = [c for c in _feature_cols(flips) if c in feat_cols]

        flipped_instances = set()
        for t, g in flips.groupby("test_idx"):
            t = int(t)
            # build candidate rows in full feature order, filling unchanged feats from original
            orig = test.loc[t, feat_cols].astype(float)
            Xc = []
            for _, row in g.iterrows():
                v = orig.copy()
                for c in fcols:
                    v[c] = float(row[c])
                Xc.append(v.values)
            Xc = np.asarray(Xc)
            preds = model.predict(scaler.transform(Xc))
            if np.any(preds == 0):
                flipped_instances.add(t)

        flipped_total += len(flipped_instances)

    return flipped_total, tp_total

def count_rq3_points(feas_csv: str) -> int:
    feas_df = pd.read_csv(feas_csv)
    # If your RQ3 CSV later includes identifiers, you could dedupe by ['project','test_idx'] here.
    return len(feas_df)

def main():
    ap = ArgumentParser()
    ap.add_argument("--feas_csv", required=True, help="Path to RQ3 feasibility CSV (Mahalanobis points)")
    ap.add_argument("--model", required=True, help="Model type, e.g., XGBoost, RandomForest, SVM, LightGBM, CatBoost")
    ap.add_argument("--explainer", required=True, help="Explainer name matching flip files, e.g., TimeLIME, LIME, LIME-HPO")
    ap.add_argument("--projects", default="all", help="Comma/space-separated project names, or 'all'")
    args = ap.parse_args()

    # Collect project list
    ds = read_dataset()
    if args.projects.strip().lower() == "all":
        projects = sorted(ds.keys())
    else:
        projects = [p.strip() for p in args.projects.replace(",", " ").split() if p.strip()]

    # RQ3
    feas_points = count_rq3_points(args.feas_csv)

    # RQ1 (raw counts)
    flipped, total_tp = recompute_rq1_counts(args.model, args.explainer, projects)

    # Ratios
    feas_per_flipped = (feas_points / flipped) if flipped else None
    feas_per_total   = (feas_points / total_tp) if total_tp else None

    # Print summary
    print("\n=== RQ1 vs RQ3 Summary ===")
    print(f"Model       : {args.model}")
    print(f"Explainer   : {args.explainer}")
    print(f"Projects    : {len(projects)} ({'all' if args.projects=='all' else ','.join(projects)})")
    print(f"RQ3 points  : {feas_points}")
    print(f"RQ1 flipped : {flipped}")
    print(f"RQ1 #TP     : {total_tp}")
    print(f"RQ3/RQ1 flipped ratio : {feas_per_flipped:.4f}" if feas_per_flipped is not None else "RQ3/RQ1 flipped ratio : n/a")
    print(f"RQ3/RQ1 TP ratio      : {feas_per_total:.4f}"   if feas_per_total   is not None else "RQ3/RQ1 TP ratio      : n/a")
    print()

if __name__ == "__main__":
    main()
