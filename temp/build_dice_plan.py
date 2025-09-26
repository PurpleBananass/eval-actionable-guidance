#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np

from data_utils import read_dataset
from hyparams import PROPOSED_CHANGES, EXPERIMENTS

META_COLS = {"test_idx", "candidate_id"}

def _load_cf_table(project: str, model_type: str) -> tuple[pd.DataFrame | None, bool]:
    """Return (df, is_long_format). Prefers long; falls back to wide; None if missing/empty."""
    base = Path(EXPERIMENTS) / project / model_type
    long_p = base / "DiCE_all_long.csv"
    wide_p = base / "DiCE_all.csv"

    def _safe_csv(p: Path):
        if not p.exists() or p.stat().st_size == 0:
            return None
        try:
            df = pd.read_csv(p)
            return df if not df.empty else None
        except Exception:
            return None

    df = _safe_csv(long_p)
    if df is not None and "test_idx" in df.columns:
        return df, True

    df = _safe_csv(wide_p)
    if df is not None:
        # older wide files sometimes have unnamed first col; drop it if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df, False

    return None, False

def build_plans_for_project(project: str, model_type: str, overwrite: bool) -> bool:
    """Build plans_all.json for a single project/model. Returns True if written."""
    ds = read_dataset()
    if project not in ds:
        print(f"[{project}/{model_type}] not in dataset, skipping.")
        return False

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    # where to write
    out_dir = Path(PROPOSED_CHANGES) / project / model_type / "DiCE"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "plans_all.json"
    if out_json.exists() and not overwrite:
        print(f"[{project}/{model_type}] plans_all.json exists; use --overwrite to rebuild.")
        return False

    df, is_long = _load_cf_table(project, model_type)
    if df is None:
        print(f"[{project}/{model_type}] no DiCE output found; skipping.")
        return False

    plans: dict[str, dict[str, list]] = {}

    if is_long:
        # long format: multiple rows per test_idx
        if not set(["test_idx"]).issubset(df.columns):
            print(f"[{project}/{model_type}] long file missing 'test_idx'; skipping.")
            return False

        for tidx, group in df.groupby("test_idx"):
            tidx_int = int(tidx)
            if tidx_int not in test.index:
                # if index was string in CSV, try coercion later
                continue
            orig = test.loc[tidx_int, feat_cols].astype(float)

            plan = {}
            # candidate feature set = all non-meta columns that are real features
            cand_feat_cols = [c for c in group.columns if c not in META_COLS and c in feat_cols]

            for f in cand_feat_cols:
                # values that differ from original
                vals = group[f].astype(float).unique().tolist()
                vals = [v for v in vals if not np.isclose(v, float(orig[f]), rtol=1e-9, atol=1e-9)]
                # keep unique and sorted for stability
                vals = sorted(set(vals))
                if len(vals) > 0:
                    plan[f] = vals

            if len(plan) > 0:
                plans[str(tidx_int)] = plan

    else:
        # wide format: one row per TP (best CF). Only one value per changed feature.
        # This will produce very small plans (often 1 value per feature).
        # RQ2 will still run, but combo spaces may be tiny.
        # We need an index -> try to align by DataFrame index if it looks like an id col.
        # If there is a 'test_idx' col in wide (rare), use it.
        idx_col = "test_idx" if "test_idx" in df.columns else None
        if idx_col:
            iterator = df.set_index(idx_col).iterrows()
        else:
            iterator = df.iterrows()

        for idx, row in iterator:
            try:
                tidx_int = int(idx)
            except Exception:
                continue
            if tidx_int not in test.index:
                continue
            orig = test.loc[tidx_int, feat_cols].astype(float)
            plan = {}
            for f in feat_cols:
                if f not in row.index:
                    continue
                val = float(row[f])
                if not np.isfinite(val):
                    continue
                if not np.isclose(val, float(orig[f]), rtol=1e-9, atol=1e-9):
                    plan[f] = [val]
            if len(plan) > 0:
                plans[str(tidx_int)] = plan

    if len(plans) == 0:
        print(f"[{project}/{model_type}] no plans could be built from CFs.")
        return False

    with open(out_json, "w") as f:
        json.dump(plans, f, indent=2)
    print(f"[OK] wrote {out_json} with {len(plans)} TPs.")
    return True

def main():
    ap = ArgumentParser()
    ap.add_argument("--project", type=str, default="all", help="Project name or 'all'")
    ap.add_argument("--model_types", type=str,
                    default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing plans_all.json")
    args = ap.parse_args()

    projects = read_dataset()
    project_list = list(sorted(projects.keys())) if args.project == "all" else args.project.split()
    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]

    wrote_any = False
    for p in project_list:
        for m in model_types:
            ok = build_plans_for_project(p, m, args.overwrite)
            wrote_any = wrote_any or ok

    if not wrote_any:
        print("Nothing written. (Either no CF outputs found, or plans already exist — use --overwrite.)")

if __name__ == "__main__":
    main()
