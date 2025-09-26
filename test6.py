#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce the DiCE tutorial on your dataset/models.

- Works with: read_dataset(), get_model() from your codebase
- Supports methods: random, genetic, kdtree
- Select which rows to explain: true positives (default), specific test indices, or random test rows
- Optional: features_to_vary (list) and permitted_range (auto from quantiles or JSON)
- Saves long-format flipped CFs per project/model/method:
    experiments/{project}/{model_type}/{method}/DiCE_all.csv
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import dice_ml
from dice_ml import Dice

# your helpers
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED


# ----------------------------- utilities -----------------------------

class ScaledModel:
    """Wrap sklearn-like estimator with an internal StandardScaler fitted on train features."""
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
                p1 = proba[:, 0]; p0 = 1.0 - p1
                return np.stack([p0, p1], axis=1)
            if proba.ndim == 1:
                p1 = proba; p0 = 1.0 - p1
                return np.stack([p0, p1], axis=1)
        if hasattr(self.model, "decision_function"):
            s = np.clip(self.model.decision_function(Xs), -50, 50)
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


def permitted_range_from_quantiles(df: pd.DataFrame, cols: list[str], lo=0.01, hi=0.99) -> dict:
    """Build a DiCE permitted_range dict from quantiles over provided dataframe."""
    pr = {}
    for c in cols:
        ql, qh = float(df[c].quantile(lo)), float(df[c].quantile(hi))
        if not np.isfinite(ql) or not np.isfinite(qh) or ql == qh:
            ql, qh = float(df[c].min()), float(df[c].max())
        pr[c] = [ql, qh]
    return pr


def load_permitted_range(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warn] permitted_range file not found: {p}")
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[warn] failed to parse permitted_range JSON: {e}")
        return None


def parse_features_to_vary(csv_list: str | None) -> list[str] | None:
    if not csv_list:
        return None
    items = [s.strip() for s in csv_list.replace(",", " ").split() if s.strip()]
    return items or None


def choose_query_instances(project: str,
                           model_type: str,
                           train: pd.DataFrame,
                           test: pd.DataFrame,
                           mode: str,
                           k: int,
                           indices: list[int] | None) -> pd.DataFrame:
    """Return a DataFrame of query rows from test set (features only, no target)."""
    feat_cols = [c for c in test.columns if c != "target"]

    if indices:
        # specific indices from test
        good = [i for i in indices if i in test.index]
        if not good:
            print(f"[{project}/{model_type}] none of the requested indices are in test; falling back to mode={mode}")
        else:
            return test.loc[good, feat_cols]

    if mode == "tp":
        # True positives only (actual target=1 & predicted 1)
        model = get_model(project, model_type)
        tps = get_true_positives(model, train, test)
        if tps.empty:
            print(f"[{project}/{model_type}] no true positives; falling back to random")
            mode = "random"
        else:
            # Keep at most k rows (if k>0); else return all TPs
            if k > 0 and len(tps) > k:
                return tps.sample(n=k, random_state=SEED).loc[:, feat_cols]
            return tps.loc[:, feat_cols]

    if mode == "random":
        df = test.loc[:, feat_cols]
        if k > 0 and len(df) > k:
            return df.sample(n=k, random_state=SEED)
        return df

    # mode == "all"
    return test.loc[:, feat_cols]


def save_flipped_candidates(out_csv: Path,
                            all_records: list[pd.DataFrame],
                            verbose=True):
    if not all_records:
        if verbose:
            print(f"[save] no flipped CFs to write: {out_csv}")
        return
    out = pd.concat(all_records, axis=0, ignore_index=False)
    out.to_csv(out_csv, index=False)
    if verbose:
        print(f"[OK] wrote {len(out)} CF rows for {out['test_idx'].nunique()} instance(s) -> {out_csv}")


# ----------------------------- main per-project flow -----------------------------

def run_dice_for_project(
    project: str,
    model_type: str,
    method: str = "random",
    *,
    select: str = "tp",          # tp | random | all
    k: int = 10,                 # how many rows (tp/random modes). 0 = all in mode.
    test_indices: list[int] | None = None,
    total_cfs: int = 4,
    features_to_vary: list[str] | None = None,  # None = let DiCE vary all features
    permitted_range: dict | None = None,        # None = no constraint; or dict of {col: [lo,hi]}
    auto_quantiles: tuple[float, float] | None = None,  # e.g., (0.01, 0.99) to auto-build ranges
    overwrite: bool = True,
    verbose: bool = True,
):
    """Reproduce DiCE tutorial on your project/model with your data."""
    valid_methods = ["random", "genetic", "kdtree"]
    if method not in valid_methods:
        print(f"[ERROR] invalid method '{method}'. Choose from {valid_methods}")
        return

    ds = read_dataset()
    if project not in ds:
        print(f"[{project}/{model_type}/{method}] dataset not found. Skipping.")
        return

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    # Model + scaler wrapper
    base_model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)
    model = ScaledModel(base_model, scaler)

    # Which rows to explain?
    query_df = choose_query_instances(project, model_type, train, test, select, k, test_indices)
    if query_df is None or query_df.empty:
        print(f"[{project}/{model_type}/{method}] no query instances to explain.")
        return

    # DiCE data/model
    all_data = pd.concat([train[feat_cols + ["target"]], test[feat_cols + ["target"]]],
                         axis=0, ignore_index=True)
    dice_data = dice_ml.Data(dataframe=all_data,
                             continuous_features=feat_cols,  # your datasets are numeric
                             outcome_name="target")
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    try:
        explainer = Dice(dice_data, dice_model, method=method)
    except Exception as e:
        print(f"[ERROR] could not create Dice explainer with method={method}: {e}")
        return

    # Permitted range: explicit JSON wins; else optional auto-quantiles; else None
    if permitted_range is None and auto_quantiles is not None:
        lo, hi = auto_quantiles
        permitted_range = permitted_range_from_quantiles(all_data, feat_cols, lo, hi)

    # Output path
    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "DiCE_all.csv"
    if overwrite and out_csv.exists():
        out_csv.unlink(missing_ok=True)

    # Generate CFs
    records = []
    for idx, row in tqdm(query_df.iterrows(),
                         total=len(query_df),
                         desc=f"{project}/{model_type}/{method}",
                         leave=False,
                         disable=not verbose):
        x0 = row.astype(float)
        x0_df = x0.to_frame().T  # 1-row DF

        # Call DiCE (with or without features_to_vary/permitted_range)
        gen_kwargs = dict(
            query_instances=x0_df,
            total_CFs=total_cfs,
            desired_class="opposite",
        )
        if features_to_vary:
            gen_kwargs["features_to_vary"] = features_to_vary
        if permitted_range:
            gen_kwargs["permitted_range"] = permitted_range

        # random seed may not be supported for all methods
        try:
            cf = explainer.generate_counterfactuals(random_seed=SEED, **gen_kwargs)
        except TypeError as te:
            if "random_seed" in str(te):
                cf = explainer.generate_counterfactuals(**gen_kwargs)
            else:
                print(f"[{project}/{model_type}/{method}] {idx}: DiCE call error: {te}")
                continue
        except Exception as e:
            print(f"[{project}/{model_type}/{method}] {idx}: DiCE call error: {e}")
            continue

        # Extract and keep only successful flips to class 0
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
        except Exception:
            cf_df = None
        if cf_df is None or cf_df.empty:
            continue

        # Normalize columns/order
        if "target" in cf_df.columns:
            cf_df = cf_df.drop(columns=["target"])
        for c in feat_cols:
            if c not in cf_df.columns:
                cf_df[c] = x0[c]
        cf_df = cf_df[feat_cols].astype(float)

        # verify flips with our scaled model
        probs = model.predict_proba(cf_df.values)
        preds = (probs[:, 1] >= 0.5).astype(int)
        flips_mask = (preds == 0)
        if not np.any(flips_mask):
            continue

        kept = cf_df.loc[flips_mask].copy()
        kept["proba0"] = probs[flips_mask, 0]
        kept["proba1"] = probs[flips_mask, 1]
        kept.insert(0, "candidate_id", np.arange(len(kept)))
        kept.insert(0, "test_idx", int(idx))
        records.append(kept)

    # Save
    save_flipped_candidates(out_csv, records, verbose=verbose)


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="Reproduce DiCE tutorial on your datasets/models.")
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated list of model types to run")
    ap.add_argument("--methods", type=str, default="random,kdtree,genetic",
                    help="Comma-separated: random,kdtree,genetic")
    ap.add_argument("--select", type=str, default="tp", choices=["tp", "random", "all"],
                    help="Which rows to explain: true positives (default), random test rows, or all test rows")
    ap.add_argument("--k", type=int, default=10,
                    help="How many rows (for select=tp|random). 0 means 'all in that mode'.")
    ap.add_argument("--test_indices", type=str, default="",
                    help="Specific test indices (comma/space-separated). Overrides --select if present.")
    ap.add_argument("--total_cfs", type=int, default=4,
                    help="How many CFs to request per instance")
    ap.add_argument("--features_to_vary", type=str, default="",
                    help="Comma/space-separated feature names to vary. Leave blank to let DiCE vary all.")
    ap.add_argument("--permitted_range_json", type=str, default="",
                    help="Path to a JSON file of {feature: [lo,hi]} to constrain ranges (optional).")
    ap.add_argument("--auto_quantiles", type=str, default="",
                    help="Auto-build permitted_range from quantiles, e.g. '0.01,0.99'. Ignored if JSON provided.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing DiCE_all.csv")
    ap.add_argument("--verbose", action="store_true", help="Verbose progress")
    args = ap.parse_args()

    # projects
    ds = read_dataset()
    projects = list(sorted(ds.keys())) if args.project == "all" else [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]

    # models & methods
    model_types = [m.strip() for m in args.model_types.replace(",", " ").split() if m.strip()]
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]

    # selection
    test_indices = [int(x) for x in args.test_indices.replace(",", " ").split() if x.strip().isdigit()] if args.test_indices else None

    # features_to_vary
    ftv = parse_features_to_vary(args.features_to_vary)

    # permitted_range
    pr = load_permitted_range(args.permitted_range_json)
    aq = None
    if pr is None and args.auto_quantiles:
        try:
            lo, hi = [float(x) for x in args.auto_quantiles.split(",")]
            aq = (lo, hi)
        except Exception:
            print("[warn] failed to parse --auto_quantiles; expected 'lo,hi' like '0.01,0.99'")

    print(f"Projects: {projects}")
    print(f"Models:   {model_types}")
    print(f"Methods:  {methods}")
    print(f"Select:   {args.select} (k={args.k})  test_indices={test_indices}")
    print(f"Features_to_vary: {ftv if ftv else '(all)'}")
    print(f"Permitted_range:  {'JSON file' if pr else (f'quantiles {aq}' if aq else '(none)')}")
    print("")

    for project in projects:
        for model_type in model_types:
            for method in methods:
                run_dice_for_project(
                    project=project,
                    model_type=model_type,
                    method=method,
                    select=args.select,
                    k=args.k,
                    test_indices=test_indices,
                    total_cfs=args.total_cfs,
                    features_to_vary=ftv,
                    permitted_range=pr,
                    auto_quantiles=aq,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                )


if __name__ == "__main__":
    main()
