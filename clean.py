#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract {project}/{model}/kdtree/DiCE_all_hist_seed2.csv and save a
one-row-per-test_idx version into {project}/{model}/DiCE_all_hist_seed2.csv.

- Keeps the 'first' candidate per test_idx (smallest candidate_id if present).
- Tolerant to files where test_idx is an index rather than a column.
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Try to reuse your repo's EXPERIMENTS root; fall back to ./experiments
try:
    from hyparams import EXPERIMENTS as DEFAULT_ROOT
except Exception:
    DEFAULT_ROOT = "./experiments"


def read_lenient_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        try:
            return pd.read_csv(path, index_col=0)
        except Exception:
            return pd.read_csv(path)
    except Exception:
        return None


def ensure_test_idx_column(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ensure exactly one 'test_idx' column; coerce to int; drop bad rows."""
    if df is None or df.empty:
        return df

    if "test_idx" in df.columns:
        if df.index.name == "test_idx":
            df = df.reset_index(drop=True)
    else:
        if df.index.nlevels > 1:
            df = df.reset_index()
            first_level = df.columns[0]
            df = df.rename(columns={first_level: "test_idx"})
        else:
            idx_name = df.index.name if df.index.name is not None else "index"
            df = df.reset_index().rename(columns={idx_name: "test_idx"})

    df["test_idx"] = pd.to_numeric(df["test_idx"], errors="ignore")
    df = df.dropna(subset=["test_idx"]).copy()
    df["test_idx"] = df["test_idx"].astype(int)

    if df.index.name == "test_idx":
        df = df.reset_index(drop=True)

    return df


def pick_first_per_group(g: pd.DataFrame) -> pd.Series:
    """Pick the first candidate per test_idx (by candidate_id if present)."""
    local = g.copy()
    if "candidate_id" in local.columns:
        local = local.assign(_cid=pd.to_numeric(local["candidate_id"], errors="coerce"))
        local = local.sort_values(by=["_cid"], kind="mergesort")
    row = local.iloc[0]
    return row.drop(labels=[c for c in ["_cid"] if c in row.index])


def process_one(kdtree_csv: Path) -> Path | None:
    """Read kdtree/DiCE_all_hist_seed2.csv and write parent-level per-test-idx file."""
    df = ensure_test_idx_column(read_lenient_csv(kdtree_csv))
    if df is None or df.empty:
        print(f"[skip] empty/unreadable: {kdtree_csv}")
        return None

    before = len(df)
    per = df.groupby("test_idx", sort=False, group_keys=False).apply(pick_first_per_group)
    if isinstance(per, pd.Series):  # single group edge case
        per = per.to_frame().T

    # Ensure 'test_idx' is a column and placed first
    if "test_idx" not in per.columns:
        per = per.reset_index(drop=False).rename(columns={"index": "test_idx"})
    cols = ["test_idx"] + [c for c in per.columns if c != "test_idx"]
    per = per[cols]

    out_path = kdtree_csv.parent.parent / "DiCE_all_hist_seed2.csv"
    per.to_csv(out_path, index=False)

    print(f"[ok] {kdtree_csv}  rows: {before} -> {len(per)}  ->  {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Extract kdtree/DiCE_all_hist_seed2.csv to model folder, 1 row per test_idx.")
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT,
                    help=f"Experiments root (default: {DEFAULT_ROOT})")
    ap.add_argument("--project", type=str, default=None,
                    help="Only process a specific project (folder under root)")
    ap.add_argument("--model", type=str, default=None,
                    help="Only process a specific model (e.g., RandomForest, XGBoost, ...)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[error] root not found: {root}")
        sys.exit(1)

    # Traverse: {root}/{project}/{model}/kdtree/DiCE_all_hist_seed2.csv
    count = 0
    for project_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if args.project and project_dir.name != args.project:
            continue
        for model_dir in sorted([m for m in project_dir.iterdir() if m.is_dir()]):
            if args.model and model_dir.name != args.model:
                continue
            kdtree_dir = model_dir / "kdtree"
            src = kdtree_dir / "DiCE_all_hist_seed2.csv"
            # also tolerate case variant "Dice_all_hist_seed2.csv"
            if not src.exists():
                alt = kdtree_dir / "Dice_all_hist_seed2.csv"
                src = alt if alt.exists() else src
            if src.exists():
                process_one(src)
                count += 1

    if count == 0:
        print("[warn] No kdtree/DiCE_all_hist_seed2.csv files found. "
              "Check --root, --project/--model filters, or filenames.")
    else:
        print(f"\nDone. Processed {count} file(s).")


if __name__ == "__main__":
    main()
