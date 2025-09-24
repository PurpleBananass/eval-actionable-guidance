#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute DiCE flip rates per project/model.

Definitions (per project/model):
- #Plan: number of TPs that have an entry in plans_all.json
- Computed: number of unique TPs present in DiCE_all.csv (i.e., for which we saved candidates)
- Flip: number of those TPs for which at least one saved candidate truly flips to class 0
- #TP: number of true positives on the test split (per your get_true_positives)

Flip% we print = Flip / #Plan  (so it's < 100% when we had plans but no saved flipper,
or when saved candidates didn't flip after full reconstruction and re-scoring).
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

from data_utils import read_dataset, get_model, get_true_positives
from hyparams import PROPOSED_CHANGES, EXPERIMENTS


# --------------------------- helpers ---------------------------

def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def _features_only(df_or_row, label="target"):
    if isinstance(df_or_row, pd.Series):
        return df_or_row[df_or_row.index != label]
    return df_or_row.loc[:, df_or_row.columns != label]


def _count_flips_for_project_model(project: str, model_type: str) -> tuple[int, int, int, int]:
    """
    Returns a 4-tuple: (flipped_count, computed_count, plan_count, tp_count).
    Works with either wide CSVs (no test_idx) or long CSVs (has test_idx & candidate rows).
    """
    ds = read_dataset()
    if project not in ds:
        return 0, 0, 0, 0

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]
    model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)

    plans_path = Path(PROPOSED_CHANGES) / project / model_type / "DiCE" / "plans_all.json"
    flips_path = Path(EXPERIMENTS) / project / model_type / "DiCE_all.csv"

    if not plans_path.exists():
        # No plans => nothing to compute/flip
        tp_count = len(get_true_positives(model, train, test))
        return 0, 0, 0, tp_count

    # Load plans and keep only those with at least one changeable value
    with open(plans_path, "r") as f:
        plans = json.load(f)
    plan_indices = {int(k) for k, v in plans.items()
                    if isinstance(v, dict) and any(len(vals) > 0 for vals in v.values())}
    plan_count = len(plan_indices)

    # All TPs (denominator for overall rate)
    tp_df = get_true_positives(model, train, test)
    tp_count = len(tp_df)

    flips_df = _safe_read_csv(flips_path)
    if flips_df is None or flips_df.empty:
        return 0, 0, plan_count, tp_count

    # Normalize common junk columns
    for junk in ("Unnamed: 0", "index"):
        if junk in flips_df.columns:
            flips_df = flips_df.drop(columns=[junk])

    # LONG format (multiple candidates): expect 'test_idx'
    if "test_idx" in flips_df.columns:
        flips_df["test_idx"] = flips_df["test_idx"].astype(int)
        # limit to TPs we have plans for
        flips_df = flips_df[flips_df["test_idx"].isin(plan_indices)]
        computed = flips_df["test_idx"].nunique()

        non_feat = {"test_idx", "candidate_id"}
        flip_feat_cols = [c for c in flips_df.columns if c not in non_feat and c in feat_cols]

        flipped_set = set()
        for tid, group in flips_df.groupby("test_idx"):
            if not flip_feat_cols:
                continue
            original = test.loc[tid, feat_cols].astype(float)
            full_rows = []
            for _, r in group.iterrows():
                v = original.copy()
                v[flip_feat_cols] = r[flip_feat_cols].astype(float).values
                full_rows.append(v.values)
            if not full_rows:
                continue
            preds = model.predict(scaler.transform(np.array(full_rows)))
            if np.any(preds == 0):
                flipped_set.add(tid)

        flipped = len(flipped_set)
        return flipped, computed, plan_count, tp_count

    # WIDE format (one row per TP; index is the TP id)
    # Keep only rows whose index is a planned TP
    # (Some wide CSVs save the TP id as the index; otherwise try to coerce.)
    if flips_df.index.dtype.kind in "iu":
        idx_tps = flips_df.index.astype(int)
    else:
        try:
            idx_tps = flips_df.index.astype(int)
        except Exception:
            # If we can't parse the index, we can't align to plans; treat as zero computed.
            return 0, 0, plan_count, tp_count

    mask = pd.Series(idx_tps, index=flips_df.index).isin(plan_indices)
    flips_df = flips_df.loc[mask]
    computed = int(mask.sum())  # rows == TPs in wide format

    flip_feat_cols = [c for c in flips_df.columns if c in feat_cols]
    if not flip_feat_cols or flips_df.empty:
        return 0, computed, plan_count, tp_count

    full_rows = []
    ids = []
    for tid, row in flips_df.iterrows():
        t = int(tid)
        original = test.loc[t, feat_cols].astype(float)
        v = original.copy()
        v[flip_feat_cols] = row[flip_feat_cols].astype(float).values
        full_rows.append(v.values)
        ids.append(t)

    preds = model.predict(scaler.transform(np.array(full_rows)))
    flipped = int(np.sum(preds == 0))
    return flipped, computed, plan_count, tp_count


def flip_rates_dice(model_types: list[str], project_list: list[str] | None = None):
    """
    Build a per-project table and per-model means.
    Returns (df, model_means).
    """
    projects = read_dataset()
    if project_list is None:
        project_iter = list(sorted(projects.keys()))
    else:
        project_iter = project_list

    rows = []
    for model_type in model_types:
        for project in project_iter:
            flipped, computed, plan_count, tp_count = _count_flips_for_project_model(project, model_type)
            rate = (flipped / plan_count) if plan_count > 0 else 0.0
            rows.append([model_type, project, flipped, computed, plan_count, tp_count, rate])

    df = pd.DataFrame(rows, columns=["Model", "Project", "Flip", "Computed", "#Plan", "#TP", "Flip%"])
    model_means = (
        df.groupby("Model")[["Flip", "Computed", "#Plan", "#TP", "Flip%"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    df.to_csv("./evaluations/flip_rates_DiCE.csv", index=False)
    return df, model_means


# --------------------------- CLI ---------------------------

def main():
    ap = ArgumentParser()
    ap.add_argument("--models", type=str,
                    default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--projects", type=str, default="all",
                    help="'all' or space-separated project names")
    args = ap.parse_args()

    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.projects == "all":
        project_list = None
    else:
        project_list = args.projects.split()

    df, model_means = flip_rates_dice(model_types, project_list)

    print("\nPer-project flip rates (DiCE):")
    print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))

    print("\nPer-model means (DiCE):")
    print(tabulate(model_means, headers=model_means.columns, tablefmt="github", showindex=False))

    print("\nSaved to ./evaluations/flip_rates_DiCE.csv")


if __name__ == "__main__":
    main()
