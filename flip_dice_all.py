#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate DiCE counterfactuals (k per true-positive instance) using different methods,
with a cap of maximum 5 features changed per counterfactual.
Verifies flips and saves successful candidates in long format.

Output CSV (per project/model/method):
  experiments/{project}/{model_type}/{method}/DiCE_all.csv

Columns:
  - test_idx: original test row index
  - candidate_id: 0..(k-1) per test_idx (after filtering)
  - <all feature columns> (no 'target')
  - proba0, proba1: model probabilities for class 0 and 1
  - num_features_changed: number of features changed
"""

import warnings
from argparse import ArgumentParser
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

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
    Provides predict and predict_proba for Dice verification.
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


def _permitted_range_from_train(train_df: pd.DataFrame, feat_cols):
    """Return {feature: [min_train, max_train]} for all features."""
    pr = {}
    for c in feat_cols:
        col = train_df[c].astype(float)
        pr[c] = [float(col.min()), float(col.max())]
    return pr


# ----------------------------- feature ranking helpers -----------------------------

def _quantile_grid(train: pd.DataFrame, feat: str, lo: float, hi: float, grid_points: int = 7):
    col = train[feat].astype(float).clip(lower=lo, upper=hi)
    qs = np.linspace(0.0, 1.0, max(3, int(grid_points)))
    vals = np.unique(np.quantile(col.values, qs, method="linear")).tolist()
    vals = [float(np.clip(v, lo, hi)) for v in vals]
    return vals


def _rank_features_single_gain(model: ScaledModel,
                               x0_df: pd.DataFrame,
                               feat_cols: List[str],
                               train: pd.DataFrame,
                               permitted_range: Dict[str, List[float]],
                               grid_points: int = 7) -> List[Tuple[str, float]]:
    """
    Per-instance ranking: for each feature, pick the value on a small grid that
    maximizes probability gain toward the *opposite* class. Returns
    [(feature, gain), ...] sorted desc by gain.
    """
    base_p = model.predict_proba(x0_df.values)[0]
    cur_label = int(np.argmax(base_p))
    target_class = 1 - cur_label
    base_tp = float(base_p[target_class])

    q = x0_df.iloc[0]
    rows = []
    spans = []
    for f in feat_cols:
        lo, hi = permitted_range[f]
        grid = _quantile_grid(train, f, lo, hi, grid_points)
        cur = float(q[f])
        cand = [v for v in grid if not np.isclose(v, cur, rtol=1e-9, atol=1e-12)]
        n = len(cand)
        if n == 0:
            spans.append((f, 0))
            continue
        for v in cand:
            r = q.copy()
            r[f] = v
            rows.append(r.values.astype(float))
        spans.append((f, n))

    if not rows:
        return [(f, 0.0) for f in feat_cols]

    X = np.vstack(rows)
    tps = model.predict_proba(X)[:, target_class]

    gains = []
    pos = 0
    for f, n in spans:
        if n == 0:
            gains.append((f, 0.0))
        else:
            best = float(np.max(tps[pos:pos + n]))
            gains.append((f, max(0.0, best - base_tp)))
            pos += n

    gains.sort(key=lambda t: t[1], reverse=True)
    return gains


def _best_feature_combos(gains: List[Tuple[str, float]],
                         k: int = 5,
                         pool_size: int = 15,
                         max_combos: int = 60) -> List[Tuple[str, ...]]:
    """
    Take top-N features by gain, form all size-k combos, score by sum of gains,
    return best few.
    """
    pool = [f for f, _ in gains[:max(1, min(pool_size, len(gains)))]]
    k = min(k, len(pool))
    if k == 0:
        return []
    all_combos = list(combinations(pool, k))
    score = {f: g for f, g in gains}
    all_combos.sort(key=lambda c: sum(score.get(f, 0.0) for f in c), reverse=True)
    return all_combos[:max_combos]


# ----------------------------- core per-project work -----------------------------

def generate_dice_flips_for_project(project: str,
                                    model_type: str,
                                    method: str = "random",
                                    total_cfs: int = 10,
                                    max_features_changed: int = 5,
                                    verbose: bool = True,
                                    overwrite: bool = True,
                                    always_find_cfs: bool = True):
    """
    For the given project/model/method:
      - find true positives on test
      - generate CFs (using DiCE method or custom K-feature search)
      - keep only candidates that:
           (a) actually flip to class 0 (strict threshold), and
           (b) change at most `max_features_changed` features (and at least 1)
      - save to experiments/{project}/{model_type}/{method}/DiCE_all_...csv
    """
    valid_methods = ["random", "kdtree", "genetic", "kfeature"]
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
    model = ScaledModel(base_model, scaler)

    # true positives (actual target=1 & predicted 1 on base_model)
    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}/{method}] no true positives. Skipping.")
        return

    # DiCE data/model
    all_data = pd.concat(
        [train[feat_cols + ["target"]], test[feat_cols + ["target"]]],
        axis=0, ignore_index=True
    )
    dice_data = dice_ml.Data(
        dataframe=all_data,
        continuous_features=feat_cols,  # all numeric in your setting
        outcome_name="target",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")

    # --- choose explainer ---
    explainer = None
    if method == "kfeature":
        try:
            from dice_ml.explainer_interfaces.dice_kfeature_search import DiceKFeatureSearch as Explainer
            explainer = Explainer(dice_data, dice_model)
        except Exception as e:
            tqdm.write(f("[ERROR] Could not import DiceKFeatureSearch: {e}"))
            return
    else:
        try:
            explainer = Dice(dice_data, dice_model, method=method)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to create DiCE explainer with method '{method}': {e}")
            return

    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"DiCE_all_{total_cfs}_max{max_features_changed}feat.csv"
    if overwrite and out_csv.exists():
        out_csv.unlink(missing_ok=True)

    results = []
    instances_without_cfs = []

    # permitted range from train (hard cap)
    permitted_range = _permitted_range_from_train(train, feat_cols)

    # -- helpers to normalize and filter candidates --
    def _normalize_cf_df(_cf_df: pd.DataFrame, x0: pd.Series) -> Optional[pd.DataFrame]:
        if _cf_df is None or _cf_df.empty:
            return None
        if "target" in _cf_df.columns:
            _cf_df = _cf_df.drop(columns=["target"])
        for c in feat_cols:
            if c not in _cf_df.columns:
                _cf_df[c] = x0[c]
        return _cf_df[feat_cols].astype(float)

    def _filter_and_pack(_cf_df: Optional[pd.DataFrame], x0: pd.Series, idx: int, label_pred: int) -> Optional[pd.DataFrame]:
        if _cf_df is None or _cf_df.empty:
            return None
        orig_vals = x0.values.astype(float)[None, :]
        cand_vals = _cf_df.values
        changed_mask_matrix = ~np.isclose(cand_vals, orig_vals, rtol=1e-7, atol=1e-7)
        changed_counts = changed_mask_matrix.sum(axis=1)
        allowed_mask = (changed_counts > 0) & (changed_counts <= max_features_changed)
        if not np.any(allowed_mask):
            return None
        _cf_df_allowed = _cf_df.iloc[allowed_mask.nonzero()[0]].copy()
        probs = model.predict_proba(_cf_df_allowed.values)
        preds = (probs[:, 1] >= 0.5).astype(int)
        flips_mask = (preds == 0) if label_pred == 1 else (preds == 1)
        if not np.any(flips_mask):
            return None
        kept = _cf_df_allowed.iloc[flips_mask.nonzero()[0]].copy()
        kept["proba0"] = probs[flips_mask, 0]
        kept["proba1"] = probs[flips_mask, 1]
        kept["num_features_changed"] = changed_counts[allowed_mask][flips_mask]
        kept.insert(0, "candidate_id", np.arange(len(kept)))
        kept.insert(0, "test_idx", idx)
        return kept

    for idx in tqdm(tp_df.index.astype(int),
                    desc=f"{project}/{model_type}/{method}",
                    leave=False,
                    disable=not verbose):
        x0 = test.loc[idx, feat_cols].astype(float)
        x0_df = x0.to_frame().T

        # original prediction on the wrapped/scaled model
        label_pred = int(model.predict(x0_df.values)[0])

        # ---- rank features and choose top-5 for the first attempt ----
        gains = _rank_features_single_gain(
            model=model,
            x0_df=x0_df,
            feat_cols=feat_cols,
            train=train,
            permitted_range=permitted_range,
            grid_points=10,
        )
        top5_feats = [f for f, _ in gains[:max_features_changed]]

        # ---- attempt 1: baseline (original random) restricted to top-5 ----
        try:
            if method == "kfeature":
                cf = explainer.generate_counterfactuals(
                    x0_df,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    features_to_vary="all",
                    permitted_range=permitted_range,
                    stopping_threshold=0.5,
                    max_changed_features=max_features_changed,
                    # sample_size=100000,
                    grid_points=10,
                    feature_pool_size=20,
                    restarts=24,
                    iter_limit=50,
                    random_seed=SEED,
                    verbose=False,
                    time_budget_sec=5.0,         # <-- hard cap per instance
    posthoc_max_seconds=2.0, 

                )
            else:
                if method == "random" and always_find_cfs:
                    # Expand sampling budget across a few attempts
                    for attempt in range(5):
                        try:
                            cf = explainer.generate_counterfactuals(
                                x0_df,
                                total_CFs=total_cfs,
                                desired_class="opposite",
                                features_to_vary=top5_feats,
                                permitted_range=permitted_range,
                                stopping_threshold=0.5,
                                sample_size=100000,
                                random_seed=SEED + attempt,
                                posthoc_sparsity_param=0.1,
                                verbose=False,
                            )
                            break
                        except TypeError:
                            cf = explainer.generate_counterfactuals(
                                x0_df,
                                total_CFs=total_cfs,
                                desired_class="opposite",
                                features_to_vary=top5_feats,
                                permitted_range=permitted_range,
                                stopping_threshold=0.5,
                                verbose=False,
                            )
                            break
                else:
                    cf = explainer.generate_counterfactuals(
                        x0_df,
                        total_CFs=total_cfs,
                        desired_class="opposite",
                        features_to_vary=top5_feats,
                        permitted_range=permitted_range,
                        stopping_threshold=0.5,
                        verbose=False,
                    )
        except Exception as e:
            tqdm.write(f"[{project}/{model_type}/{method}] DiCE error @ {idx}: {e}")
            instances_without_cfs.append(idx)
            continue

        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
        except Exception:
            cf_df = None

        kept = _filter_and_pack(_normalize_cf_df(cf_df, x0), x0, idx, label_pred)

        # ---- fallback: try several 5-feature combinations if no CFs were found ----
        if kept is None or kept.empty:
            combos5 = _best_feature_combos(gains, k=max_features_changed, pool_size=15, max_combos=60)
            found_any = False
            for subset in combos5:
                try:
                    if method == "random" and always_find_cfs:
                        for attempt in range(4):
                            try:
                                cf2 = explainer.generate_counterfactuals(
                                    x0_df,
                                    total_CFs=total_cfs,
                                    desired_class="opposite",
                                    features_to_vary=list(subset),
                                    permitted_range=permitted_range,
                                    stopping_threshold=0.5,
                                    sample_size=100000 * (2 ** attempt),
                                    random_seed=SEED + attempt,
                                    posthoc_sparsity_param=0.1,
                                    verbose=False,
                                )
                                break
                            except TypeError:
                                cf2 = explainer.generate_counterfactuals(
                                    x0_df,
                                    total_CFs=total_cfs,
                                    desired_class="opposite",
                                    features_to_vary=list(subset),
                                    permitted_range=permitted_range,
                                    stopping_threshold=0.5,
                                    verbose=False,
                                )
                                break
                    else:
                        cf2 = explainer.generate_counterfactuals(
                            x0_df,
                            total_CFs=total_cfs,
                            desired_class="opposite",
                            features_to_vary=list(subset),
                            permitted_range=permitted_range,
                            stopping_threshold=0.5,
                            verbose=False,
                        )
                except Exception as e:
                    tqdm.write(f"[{project}/{model_type}/{method}] subset {subset} error @ {idx}: {e}")
                    continue

                try:
                    cf2_df = cf2.cf_examples_list[0].final_cfs_df
                except Exception:
                    cf2_df = None

                kept2 = _filter_and_pack(_normalize_cf_df(cf2_df, x0), x0, idx, label_pred)
                if kept2 is not None and not kept2.empty:
                    kept = kept2
                    found_any = True
                    break

            if not found_any:
                instances_without_cfs.append(idx)
                continue

        # ---- record results (either baseline or fallback) ----
        results.append(kept)

    # write results
    if results:
        out_df = pd.concat(results, axis=0, ignore_index=False)
        out_df.to_csv(out_csv, index=False)
        flipped = out_df["test_idx"].nunique()
        computed = len(out_df)
        avg_features_changed = out_df["num_features_changed"].mean()
        tqdm.write(f"[OK] {project}/{model_type}/{method}: wrote {computed} candidates "
                   f"for {flipped}/{len(tp_df)} TP(s) (avg {avg_features_changed:.2f} features changed) -> {out_csv}")
        if instances_without_cfs:
            tqdm.write(f"  No valid CFs for {len(instances_without_cfs)} TP(s).")
    else:
        tqdm.write(f"[{project}/{model_type}/{method}] no candidates found with max {max_features_changed} features changed.")


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="Generate DiCE counterfactuals with maximum feature change limit")
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated list: RandomForest,SVM,XGBoost,LightGBM,CatBoost")
    ap.add_argument("--methods", type=str, default="random,kfeature",
                    help="Comma-separated DiCE methods: random,kdtree,genetic,kfeature")

    ap.add_argument("--total_cfs", type=int, default=100,
                    help="How many CFs to request from DiCE per instance")
    ap.add_argument("--max_features", type=int, default=5,
                    help="Maximum number of features that can be changed per CF")
    ap.add_argument("--always_find", action="store_true",
                    help="Expand search space exponentially to find valid CFs (maintains strict threshold)")
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

    valid_methods = ["random", "kdtree", "genetic", "kfeature"]
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
    print(f"  Max features changed: {args.max_features}")
    print(f"  Always find CFs: {args.always_find} (via search space expansion)")
    print()

    for p, m, method in tqdm(combos, desc="Projects/Models/Methods", leave=True, disable=not args.verbose):
        generate_dice_flips_for_project(
            project=p,
            model_type=m,
            method=method,
            total_cfs=args.total_cfs,
            max_features_changed=args.max_features,
            verbose=args.verbose,
            overwrite=args.overwrite,
            always_find_cfs=args.always_find,
        )

    print("Done!")


if __name__ == "__main__":
    main()
