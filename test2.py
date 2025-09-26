#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

# use your existing helpers
from data_utils import read_dataset, get_model

LABEL = "target"

def _tp_iloc_bool(model, train_df: pd.DataFrame, test_df: pd.DataFrame, label: str = LABEL):
    """TPs via boolean mask + iloc (safe)."""
    assert label in test_df.columns
    feat_cols = [c for c in test_df.columns if c != label]

    gt = test_df.loc[test_df[label], feat_cols]               # actual positives
    scaler = StandardScaler().fit(train_df[feat_cols].values)
    preds = model.predict(scaler.transform(gt.values))        # could be 0/1 or bool
    mask = np.asarray(preds).astype(bool).ravel()
    return gt.iloc[mask], preds

def _tp_direct(model, train_df: pd.DataFrame, test_df: pd.DataFrame, label: str = LABEL):
    """TPs via direct bracket indexing (safe only if preds are boolean)."""
    assert label in test_df.columns
    feat_cols = [c for c in test_df.columns if c != label]

    gt = test_df.loc[test_df[label], feat_cols]               # actual positives
    scaler = StandardScaler().fit(train_df[feat_cols].values)
    preds = model.predict(scaler.transform(gt.values))        # expect boolean for this to work correctly
    # NOTE: if preds are 0/1 integers, pandas may treat them as *labels* not a mask.
    return gt[preds], preds

def compare_true_positive_indexing(project: str, model_type: str, show_examples: int = 10):
    """Load model & data like your pipeline, then compare TP selection methods."""
    datasets = read_dataset()
    if project not in datasets:
        print(f"[{project}] not found in read_dataset(); skipping.")
        return

    train_df, test_df = datasets[project]
    model = get_model(project, model_type)

    tp1, preds1 = _tp_iloc_bool(model, train_df, test_df, label=LABEL)
    tp2, preds2 = _tp_direct(model, train_df, test_df, label=LABEL)

    # Sanity: prediction dtype
    p_dtype1 = np.asarray(preds1).dtype
    p_dtype2 = np.asarray(preds2).dtype

    # Size & index diffs
    only_iloc = tp1.index.difference(tp2.index)
    only_direct = tp2.index.difference(tp1.index)
    both = tp1.index.intersection(tp2.index)

    print(f"\n=== {project} / {model_type} ===")
    print(f"preds dtype (iloc path):   {p_dtype1}")
    print(f"preds dtype (direct path): {p_dtype2}")
    print(f"TP count (iloc+bool): {len(tp1)}")
    print(f"TP count (direct[]):  {len(tp2)}")
    print(f"Overlap (both):       {len(both)}")
    print(f"Only in iloc+bool:    {len(only_iloc)}")
    print(f"Only in direct[]:     {len(only_direct)}")

    if show_examples > 0:
        if len(only_iloc):
            print(f"  e.g., only in iloc+bool: {list(only_iloc[:show_examples])}")
        if len(only_direct):
            print(f"  e.g., only in direct[] : {list(only_direct[:show_examples])}")

    return {
        "tp_iloc_bool": tp1,
        "tp_direct": tp2,
        "only_iloc_idxs": only_iloc,
        "only_direct_idxs": only_direct,
        "both_idxs": both,
        "preds_dtype_iloc": p_dtype1,
        "preds_dtype_direct": p_dtype2,
    }

def main():
    ap = ArgumentParser(description="Compare true-positive selection methods")
    ap.add_argument("--projects", type=str, default="all",
                    help="Comma-separated project names, or 'all'")
    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--show_examples", type=int, default=5, help="How many differing indices to print")
    args = ap.parse_args()

    datasets = read_dataset()
    project_list = (sorted(datasets.keys()) if args.projects.strip().lower() == "all"
                    else [p.strip() for p in args.projects.split(",") if p.strip()])
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    total = len(project_list) * len(model_types)
    print(f"Comparing true-positive indexing methods for {len(project_list)} projects × {len(model_types)} models = {total} runs.") 
    for proj in project_list:
        for m in model_types:
            try:
                compare_true_positive_indexing(proj, m, show_examples=args.show_examples)

            except Exception as e:
                print(f"[WARN] {proj}/{m}: comparison failed: {e}")
    print(total, "runs done.")
if __name__ == "__main__":
    main()
