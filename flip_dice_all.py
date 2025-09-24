#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate DiCE counterfactuals (10 per true-positive instance) using different methods,
verify flips, and save all successful candidates in long format.

Output CSV (per project/model/method):
  experiments/{project}/{model_type}/{method}/DiCE_all.csv

Columns:
  - test_idx: original test row index
  - candidate_id: 0..(k-1) per test_idx
  - <all feature columns> (no 'target')
  - proba0, proba1: model probabilities for class 0 and 1
"""

import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

import dice_ml
from dice_ml import Dice

# your helpers
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)


# ----------------------------- model wrapper -----------------------------

class ScaledModel:
    """
    Wraps an sklearn-like classifier so it accepts *unscaled* X and internally
    applies a StandardScaler (fit on train features).
    Provides predict and predict_proba for Dice/verification.
    """

    def __init__(self, base_model, scaler: StandardScaler):
        self.model = base_model
        self.scaler = scaler
        # mirror common attrs if present (helps some toolchains)
        if hasattr(base_model, "classes_"):
            self.classes_ = base_model.classes_

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xs)
            # Ensure 2-column output (P0, P1). If single column, synthesize.
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba
            # If only one column (e.g., prob of positive class), synthesize P0
            if proba.ndim == 2 and proba.shape[1] == 1:
                p1 = proba[:, 0]
                p0 = 1.0 - p1
                return np.stack([p0, p1], axis=1)
            if proba.ndim == 1:
                p1 = proba
                p0 = 1.0 - p1
                return np.stack([p0, p1], axis=1)
        # fallback: decision_function -> sigmoid
        if hasattr(self.model, "decision_function"):
            s = self.model.decision_function(Xs)
            s = np.clip(s, -50, 50)
            p1 = 1.0 / (1.0 + np.exp(-s))
            if p1.ndim == 1:
                return np.stack([1.0 - p1, p1], axis=1)
            # multiclass not expected here; reduce to binary-ish
            p1r = p1[:, 0]
            return np.stack([1.0 - p1r, p1r], axis=1)
        # last resort: predict labels and one-hot-ish probs
        y = self.model.predict(Xs)
        p0 = (y == 0).astype(float)
        p1 = 1.0 - p0
        return np.stack([p0, p1], axis=1)

    # let dice_ml access other attrs/methods if needed
    def __getattr__(self, name):
        return getattr(self.model, name)


# ----------------------------- core per-project work -----------------------------

def generate_dice_flips_for_project(project: str,
                                    model_type: str,
                                    method: str = "random",
                                    total_cfs: int = 10,
                                    verbose: bool = True,
                                    overwrite: bool = True):
    """
    For the given project/model/method:
      - find true positives on test
      - generate 10 DiCE CFs per TP using specified method
      - keep only candidates that actually flip to class 0
      - save to experiments/{project}/{model_type}/{method}/DiCE_all.csv (long format)
    """
    # Validate method
    valid_methods = ["random", "kdtree", "genetic"]
    if method not in valid_methods:
        tqdm.write(f"[ERROR] Invalid method '{method}'. Must be one of: {valid_methods}")
        return

    # load data/model
    ds = read_dataset()
    if project not in ds:
        tqdm.write(f"[{project}/{model_type}/{method}] not found. Skipping.")
        return

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    base_model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)
    model = ScaledModel(base_model, scaler)

    # true positives (actual target=1 AND predicted 1)
    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}/{method}] no true positives. Skipping.")
        return

    # DiCE data/model interfaces - use ALL available data for search space
    all_data = pd.concat([
        train[feat_cols + ["target"]], 
        test[feat_cols + ["target"]]
    ], axis=0, ignore_index=True)
    
    data = dice_ml.Data(
        dataframe=all_data,  # Use combined dataset instead of just train
        continuous_features=feat_cols,
        outcome_name="target",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")  # wrap with scaler

    # Create explainer with specified method
    try:
        explainer = Dice(data, dice_model, method=method)
    except Exception as e:
        tqdm.write(f"[ERROR] Failed to create DiCE explainer with method '{method}': {e}")
        return

    # output path - organized by method
    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "DiCE_all.csv"

    if overwrite and out_csv.exists():
        out_csv.unlink(missing_ok=True)

    results = []

    for idx in tqdm(tp_df.index.astype(int), 
                   desc=f"{project}/{model_type}/{method}", 
                   leave=False, 
                   disable=not verbose):
        x0 = test.loc[idx, feat_cols]
        x0_df = x0.to_frame().T  # 1-row DF

        try:
            # Try with random_seed first (works for 'random' method)
            try:
                cf = explainer.generate_counterfactuals(
                    x0_df,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    random_seed=SEED,
                )
            except TypeError as te:
                # If random_seed is not supported by this method, try without it
                if "random_seed" in str(te):
                    cf = explainer.generate_counterfactuals(
                        x0_df,
                        total_CFs=total_cfs,
                        desired_class="opposite",
                    )
                else:
                    raise te
        except Exception as e:
            tqdm.write(f"[{project}/{model_type}/{method}] DiCE error @ {idx}: {e}")
            continue

        # extract candidates for this instance
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
        except Exception:
            cf_df = None

        if cf_df is None or len(cf_df) == 0:
            continue

        # keep feature subset, ensure numeric
        if "target" in cf_df.columns:
            cf_df = cf_df.drop(columns=["target"])
        for c in feat_cols:
            if c not in cf_df.columns:
                # if DiCE didn't include a column (unlikely), backfill with original
                cf_df[c] = x0[c]

        cf_df = cf_df[feat_cols].astype(float)

        # verify flips with the (scaled) model
        probs = model.predict_proba(cf_df.values)
        preds = (probs[:, 1] >= 0.5).astype(int)  # class-1 threshold at 0.5
        # we want flips to class 0 -> predicted 0 => probs[:,0] >= 0.5
        flips_mask = (preds == 0)

        if not np.any(flips_mask):
            continue

        kept = cf_df.loc[flips_mask].copy()
        kept["proba0"] = probs[flips_mask, 0]
        kept["proba1"] = probs[flips_mask, 1]
        kept.insert(0, "candidate_id", np.arange(len(kept)))
        kept.insert(0, "test_idx", idx)
        results.append(kept)

    # write
    if results:
        out_df = pd.concat(results, axis=0, ignore_index=False)
        out_df.to_csv(out_csv, index=False)
        flipped = out_df["test_idx"].nunique()
        computed = len(out_df)
        tqdm.write(f"[OK] {project}/{model_type}/{method}: wrote {computed} flipped candidates "
                   f"for {flipped} TP(s) -> {out_csv}")
    else:
        tqdm.write(f"[{project}/{model_type}/{method}] no flipped candidates found.")


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="Generate DiCE counterfactuals with different methods")
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated list: RandomForest,SVM,XGBoost,LightGBM,CatBoost")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated DiCE methods: random,kdtree,genetic (default: random)")
    ap.add_argument("--total_cfs", type=int, default=100,
                    help="How many CFs to request from DiCE per instance")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing experiment files")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable verbose output")
    args = ap.parse_args()

    # Parse arguments
    projects = read_dataset()
    project_list = list(sorted(projects.keys())) if args.project == "all" else args.project.split()
    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Validate methods
    valid_methods = ["random", "kdtree", "genetic"]
    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        print(f"ERROR: Invalid methods: {invalid_methods}")
        print(f"Valid methods are: {valid_methods}")
        return

    # Generate all combinations
    combos = [(p, m, method) for p in project_list for m in model_types for method in methods]
    
    print(f"Running {len(combos)} combinations:")
    print(f"  Projects: {project_list}")
    print(f"  Models: {model_types}")
    print(f"  Methods: {methods}")
    print()

    for p, m, method in tqdm(combos, desc="Projects/Models/Methods", leave=True, disable=not args.verbose):
        generate_dice_flips_for_project(
            project=p,
            model_type=m,
            method=method,
            total_cfs=args.total_cfs,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )

    print("Done!")


if __name__ == "__main__":
    main()