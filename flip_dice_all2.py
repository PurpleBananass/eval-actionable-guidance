#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate DiCE counterfactuals (k per true-positive instance) using different methods,
restricting changes to the per-instance top-K LIME features (default K=5).
Verifies flips and saves successful candidates in long format.

Output CSV (per project/model/method):
  experiments/{project}/{model_type}/{method}/DiCE_all.csv

Columns:
  - test_idx: original test row index
  - candidate_id: 0..(k-1) per test_idx (after filtering)
  - <all feature columns> (no 'target')
  - proba0, proba1: model probabilities for class 0 and 1
"""

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
from lime.lime_tabular import LimeTabularExplainer

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
    Provides predict and predict_proba for Dice/LIME verification.
    """

    def __init__(self, base_model, scaler: StandardScaler):
        self.model = base_model
        self.scaler = scaler
        if hasattr(base_model, "classes_"):
            self.classes_ = base_model.classes_

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xs)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba
            if proba.ndim == 2 and proba.shape[1] == 1:
                p1 = proba[:, 0]
                return np.stack([1.0 - p1, p1], axis=1)
            if proba.ndim == 1:
                p1 = proba
                return np.stack([1.0 - p1, p1], axis=1)
        if hasattr(self.model, "decision_function"):
            s = self.model.decision_function(Xs)
            s = np.clip(s, -50, 50)
            p1 = 1.0 / (1.0 + np.exp(-s))
            if p1.ndim == 1:
                return np.stack([1.0 - p1, p1], axis=1)
            p1r = p1[:, 0]
            return np.stack([1.0 - p1r, p1r], axis=1)
        y = self.model.predict(Xs)
        p0 = (y == 0).astype(float)
        return np.stack([p0, 1.0 - p0], axis=1)

    def __getattr__(self, name):
        return getattr(self.model, name)


# ----------------------------- core per-project work -----------------------------

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import dice_ml
from dice_ml import Dice
from hyparams import EXPERIMENTS, SEED

def _permitted_ranges_by_quantile(df: pd.DataFrame, cols: list[str], qlo=0.01, qhi=0.99) -> dict:
    """Quantile-based ranges to help DiCE search; does NOT change top-5 selection."""
    pr = {}
    for c in cols:
        lo = float(df[c].quantile(qlo))
        hi = float(df[c].quantile(qhi))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(df[c].min())
            hi = float(df[c].max())
        pr[c] = [lo, hi]
    return pr

def generate_dice_flips_for_project(project: str,
                                    model_type: str,
                                    method: str = "random",
                                    total_cfs: int = 10,
                                    # topk argument ignored to enforce hard 5
                                    verbose: bool = True,
                                    overwrite: bool = True):
    """
    Generate DiCE CFs restricted to the per-instance LIME **top-5** features,
    with top-5 chosen EXACTLY like your original code (class=1 map + first 5).
    No backoffs that change K or the top-5 methodology.
    """
    TOPK = 5  # hard cap; do not change

    valid_methods = ["random", "kdtree", "genetic"]
    if method not in valid_methods:
        tqdm.write(f"[ERROR] Invalid method '{method}'. Must be one of: {valid_methods}")
        return

    ds = read_dataset()
    if project not in ds:
        tqdm.write(f"[{project}/{model_type}/{method}] dataset not found. Skipping.")
        return

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    base_model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)

    class ScaledModel:
        def __init__(self, base, scaler):
            self.model = base; self.scaler = scaler
            if hasattr(base, "classes_"): self.classes_ = base.classes_
        def predict(self, X): return self.model.predict(self.scaler.transform(X))
        def predict_proba(self, X):
            Xs = self.scaler.transform(X)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(Xs)
                if proba.ndim == 2 and proba.shape[1] == 2: return proba
                if proba.ndim == 2 and proba.shape[1] == 1:
                    p1 = proba[:,0]; return np.stack([1.0-p1, p1], axis=1)
                if proba.ndim == 1:
                    p1 = proba; return np.stack([1.0-p1, p1], axis=1)
            if hasattr(self.model, "decision_function"):
                s = np.clip(self.model.decision_function(Xs), -50, 50)
                p1 = 1.0/(1.0+np.exp(-s))
                if p1.ndim == 1: return np.stack([1.0-p1, p1], axis=1)
                p1r = p1[:,0]; return np.stack([1.0-p1r, p1r], axis=1)
            y = self.model.predict(Xs); p0=(y==0).astype(float); return np.stack([p0,1.0-p0], axis=1)
        def __getattr__(self, name): return getattr(self.model, name)

    model = ScaledModel(base_model, scaler)

    # True positives (actual 1 and predicted 1)
    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}/{method}] no true positives. Skipping.")
        return

    # LIME explainer: use scaled train (like your original)
    X_train_scaled = scaler.transform(train[feat_cols].values)
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=train["target"].values,   # 1D labels
        feature_names=feat_cols,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    # DiCE explainer
    all_data = pd.concat([train[feat_cols+["target"]], test[feat_cols+["target"]]], axis=0, ignore_index=True)
    dice_data = dice_ml.Data(dataframe=all_data, continuous_features=feat_cols, outcome_name="target")
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    try:
        dice_explainer = Dice(dice_data, dice_model, method=method)
    except Exception as e:
        tqdm.write(f"[ERROR] Failed to create DiCE explainer with method '{method}': {e}")
        return

    # Optional: quantile-based permitted ranges (does not change top-5)
    permitted_range = _permitted_ranges_by_quantile(all_data, feat_cols, qlo=0.01, qhi=0.99)

    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "DiCE_all.csv"
    if overwrite and out_csv.exists():
        out_csv.unlink(missing_ok=True)

    results = []

    for idx in tqdm(tp_df.index.astype(int), desc=f"{project}/{model_type}/{method}", leave=False, disable=not verbose):
        x0 = test.loc[idx, feat_cols].astype(float)
        x0_df = x0.to_frame().T

        # ----- LIME top-5 EXACTLY like original -----
        # original calls ask for many features then slice to 5
        x0_scaled = scaler.transform(x0_df.values)
        lime_exp = lime_explainer.explain_instance(
            x0_scaled[0],
            model.predict_proba,              # ScaledModel handles scaling inside
            num_features=len(feat_cols),      # match original call
        )
        # original logic: class 1 map, slice first 5
        try:
            top_features = lime_exp.as_map()[1]          # list of (feat_idx, weight)
        except KeyError:
            # original would fail if class 1 map missing; we just skip this instance
            tqdm.write(f"[{project}/{model_type}/{method}] {idx}: LIME as_map()[1] missing; skipping.")
            continue

        top_features_index = [pair[0] for pair in top_features][:TOPK]
        if len(top_features_index) < TOPK:
            tqdm.write(f"[{project}/{model_type}/{method}] {idx}: fewer than 5 features; skipping.")
            continue
        top5_names = [feat_cols[i] for i in top_features_index]

        # (optional) also compute as_list()[:5] to mirror original, although we don't use the strings here
        _ = lime_exp.as_list()[:TOPK]

        # ----- DiCE restricted to those EXACT 5 features -----
        # Try with random_seed if supported; otherwise without.
        try:
            try:
                cf = dice_explainer.generate_counterfactuals(
                    x0_df,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    features_to_vary=top5_names,
                    permitted_range=permitted_range,
                    random_seed=SEED,
                )
            except TypeError as te:
                if "random_seed" in str(te):
                    cf = dice_explainer.generate_counterfactuals(
                        x0_df,
                        total_CFs=total_cfs,
                        desired_class="opposite",
                        features_to_vary=top5_names,
                        permitted_range=permitted_range,
                    )
                else:
                    raise te
        except Exception as e:
            tqdm.write(f"[{project}/{model_type}/{method}] {idx}: DiCE error: {e}")
            continue

        # Extract candidates
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
        except Exception:
            cf_df = None
        if cf_df is None or cf_df.empty:
            # No CFs under the strict top-5 constraint → skip (no methodology change).
            continue

        # Ensure all feature columns exist and order is consistent
        if "target" in cf_df.columns:
            cf_df = cf_df.drop(columns=["target"])
        for c in feat_cols:
            if c not in cf_df.columns:
                cf_df[c] = x0[c]
        cf_df = cf_df[feat_cols].astype(float)

        # Guarantee ≤5 changed features (should already hold with features_to_vary=top5)
        diffs = ~np.isclose(cf_df.values, x0.values[None, :], rtol=1e-7, atol=1e-7)
        changed_counts = diffs.sum(axis=1)
        keep_mask = (changed_counts > 0) & (changed_counts <= TOPK)
        if not np.any(keep_mask):
            continue
        cf_df = cf_df.loc[keep_mask].copy()

        # Verify flips to class 0
        probs = model.predict_proba(cf_df.values)
        preds = (probs[:, 1] >= 0.5).astype(int)
        flips_mask = (preds == 0)
        if not np.any(flips_mask):
            continue

        kept = cf_df.loc[flips_mask].copy()
        kept["proba0"] = probs[flips_mask, 0]
        kept["proba1"] = probs[flips_mask, 1]
        kept.insert(0, "candidate_id", np.arange(len(kept)))
        kept.insert(0, "test_idx", idx)
        results.append(kept)

    if results:
        out_df = pd.concat(results, axis=0, ignore_index=False)
        out_df.to_csv(out_csv, index=False)
        flipped = out_df["test_idx"].nunique()
        computed = len(out_df)
        tqdm.write(f"[OK] {project}/{model_type}/{method}: wrote {computed} flipped candidates "
                   f"(restricted to exact LIME top-5) for {flipped} TP(s) -> {out_csv}")
    else:
        tqdm.write(f"[{project}/{model_type}/{method}] no flipped candidates found under exact top-5 constraint.")


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="Generate DiCE counterfactuals (restricted to per-instance top-K LIME features)")
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated list: RandomForest,SVM,XGBoost,LightGBM,CatBoost")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated DiCE methods: random,kdtree,genetic (default: random)")
    ap.add_argument("--total_cfs", type=int, default=1,
                    help="How many CFs to request from DiCE per instance")
    ap.add_argument("--topk", type=int, default=5,
                    help="Number of LIME features to allow DiCE to vary (per instance)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing experiment files")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable verbose output")
    args = ap.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]

    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    valid_methods = ["random", "kdtree", "genetic"]
    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        print(f"ERROR: Invalid methods: {invalid_methods}")
        print(f"Valid methods are: {valid_methods}")
        return

    combos = [(p, m, method) for p in project_list for m in model_types for method in methods]

    print(f"Running {len(combos)} combinations:")
    print(f"  Projects: {project_list}")
    print(f"  Models: {model_types}")
    print(f"  Methods: {methods}")
    print(f"  LIME top-K: {args.topk}")
    print()

    for p, m, method in tqdm(combos, desc="Projects/Models/Methods", leave=True, disable=not args.verbose):
        generate_dice_flips_for_project(
            project=p,
            model_type=m,
            method=method,
            total_cfs=args.total_cfs,
            # topk=args.topk,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )

    print("Done!")


if __name__ == "__main__":
    main()
