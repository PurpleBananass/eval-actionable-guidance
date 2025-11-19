#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiCE + LIME with tiered permitted ranges:
  1) LIME top-K features_to_vary
  2) If none, weighted random K-subsets (P ∝ LIME weights)
  3) Repeat across a schedule of *tight → wider* permitted ranges
     (global quantiles or per-instance neighbor quantiles),
     ending with full [min, max] if needed.

Output:
  experiments/{project}/{model}/{method}/DiCE_all_{TOTAL}_max{K}feat.csv
"""

import sys
import io
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- LIME ---
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    print("ERROR: python package 'lime' is not installed. Install with: pip install lime")
    sys.exit(1)

# --- DiCE ---
import dice_ml
from dice_ml import Dice
from raiutils.exceptions import UserConfigValidationException

# --- your helpers (unchanged) ---
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED

np.random.seed(SEED)


# ----------------------------- wrappers & utils -----------------------------

class ModelWrapper:
    """Wrap base model to apply a StandardScaler fitted on TRAIN."""
    def __init__(self, model: Any, scaler: Optional[StandardScaler]):
        self.model = model
        self.scaler = scaler
        if hasattr(model, "classes_"):
            self.classes_ = model.classes_

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.scaler.transform(X) if self.scaler is not None else X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._prep(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._prep(X)
        if hasattr(self.model, "predict_proba"):
            P = self.model.predict_proba(Xp)
            if P.ndim == 1:
                p1 = P
                return np.stack([1.0 - p1, p1], axis=1)
            if P.ndim == 2 and P.shape[1] == 1:
                p1 = P[:, 0]
                return np.stack([1.0 - p1, p1], axis=1)
            return P
        if hasattr(self.model, "decision_function"):
            s = np.asarray(self.model.decision_function(Xp))
            s = np.clip(s, -50, 50)
            p1 = 1.0 / (1.0 + np.exp(-s))
            if p1.ndim == 1:
                return np.stack([1.0 - p1, p1], axis=1)
            p1r = p1[:, 0]
            return np.stack([1.0 - p1r, p1r], axis=1)
        y = self.model.predict(Xp)
        p0 = (y == 0).astype(float)
        return np.stack([p0, 1.0 - p0], axis=1)

    def __getattr__(self, name):
        return getattr(self.model, name)


def feature_minmax(train_df: pd.DataFrame, feat_cols: List[str]) -> Dict[str, List[float]]:
    out = {}
    for c in feat_cols:
        col = train_df[c].astype(float)
        out[c] = [float(col.min()), float(col.max())]
    return out


# ----------------------------- LIME helpers -----------------------------

def lime_importance_for_target(
    explainer: LimeTabularExplainer,
    model: ModelWrapper,
    x0_row: pd.Series,
    target_label: int,
    num_features: int,
    num_samples: int,
) -> np.ndarray:
    exp = explainer.explain_instance(
        data_row=x0_row.values.astype(float),
        predict_fn=model.predict_proba,
        labels=[target_label],
        num_features=num_features,
        num_samples=num_samples,
    )
    w = np.zeros(num_features, dtype=float)
    for fid, weight in exp.as_map().get(target_label, []):
        if 0 <= fid < num_features:
            w[int(fid)] = float(abs(weight))
    if not np.any(w > 0):
        w[:] = 1.0
    return w / (w.sum() + 1e-12)


def topk_from_weights(weights: np.ndarray, feature_names: Sequence[str], k: int) -> List[str]:
    idx = np.argsort(-weights)[: max(1, min(k, len(weights)))]
    return [feature_names[i] for i in idx]


def sample_weighted_k_subsets(
    weights: np.ndarray,
    feature_names: Sequence[str],
    k: int,
    num_subsets: int,
    rng: np.random.Generator,
) -> List[Tuple[str, ...]]:
    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    p = weights / weights.sum()
    n = len(feature_names)
    seen = set()
    out: List[Tuple[str, ...]] = []
    for _ in range(max(1, num_subsets)):
        choice = rng.choice(np.arange(n), size=min(k, n), replace=False, p=p)
        subset = tuple(sorted(feature_names[i] for i in choice))
        if subset in seen:
            continue
        seen.add(subset)
        out.append(subset)
    return out


# ----------------------------- permitted-range tiers -----------------------------

def parse_schedule(spec: str) -> List[Tuple[float, float]]:
    """
    "0.25:0.75,0.15:0.85,0.05:0.95,0.0:1.0" → [(0.25,0.75), ...]
    """
    tiers = []
    for part in spec.split(","):
        if not part.strip():
            continue
        a, b = part.split(":")
        tiers.append((float(a), float(b)))
    return tiers


def prange_global_quantiles(train: pd.DataFrame, feat_cols: List[str], qlo: float, qhi: float) -> Dict[str, List[float]]:
    out = {}
    for c in feat_cols:
        col = train[c].astype(float)
        lo = float(col.quantile(qlo))
        hi = float(col.quantile(qhi))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(col.min()), float(col.max())
        out[c] = [lo, hi]
    return out


def prange_neighbor_quantiles(train: pd.DataFrame,
                              feat_cols: List[str],
                              x0_row: pd.Series,
                              scaler: StandardScaler,
                              knn: NearestNeighbors,
                              qlo: float,
                              qhi: float,
                              ensure_contains_x0: bool = True) -> Dict[str, List[float]]:
    """
    Build per-instance ranges from the k-nearest neighbors (in z-space).
    """
    Xz = scaler.transform(train[feat_cols].values.astype(float))
    x0z = scaler.transform(x0_row.values.reshape(1, -1))
    _, idx = knn.kneighbors(x0z, return_distance=True)
    neigh = train[feat_cols].iloc[idx[0]].astype(float)

    out = {}
    for c in feat_cols:
        col = neigh[c].values
        lo = float(np.quantile(col, qlo, method="linear"))
        hi = float(np.quantile(col, qhi, method="linear"))
        # robust fallback
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            col_all = train[c].astype(float).values
            lo = float(np.quantile(col_all, qlo, method="linear"))
            hi = float(np.quantile(col_all, qhi, method="linear"))
        # include current value if requested
        if ensure_contains_x0:
            x0v = float(x0_row[c])
            lo = min(lo, x0v)
            hi = max(hi, x0v)
        out[c] = [lo, hi]
    return out


# ----------------------------- DiCE call (quiet) -----------------------------

def call_dice_generate(explainer, x0_df, features_to_vary, total_cfs, permitted_range,
                       stopping_threshold=0.5, verbose=False, echo_dice=False, extra_kwargs=None,
                       require_permitted=True):
    """Call DiCE while capturing its prints; return cf object or None."""
    kwargs = dict(
        total_CFs=int(total_cfs),
        desired_class="opposite",
        features_to_vary=list(features_to_vary),
        permitted_range=permitted_range,
        stopping_threshold=float(stopping_threshold),
        verbose=bool(verbose),
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    buf = io.StringIO()

    def _try():
        with redirect_stdout(buf), redirect_stderr(buf):
            return explainer.generate_counterfactuals(x0_df, **kwargs)

    try:
        obj = _try()
    except TypeError:
        if require_permitted:
            return None  # respect the tiered ranges (don't silently drop)
        kwargs.pop("permitted_range", None)
        try:
            obj = _try()
        except (UserConfigValidationException, Exception):
            obj = None
    except (UserConfigValidationException, Exception):
        obj = None
    finally:
        msg = buf.getvalue().strip()
        if echo_dice and msg:
            tqdm.write(msg, file=sys.stderr)
    return obj


def normalize_cf_df(cf_obj: Any, feat_cols: List[str], x0: pd.Series) -> Optional[pd.DataFrame]:
    try:
        cf_df = cf_obj.cf_examples_list[0].final_cfs_df
    except Exception:
        return None
    if cf_df is None or cf_df.empty:
        return None
    if "target" in cf_df.columns:
        cf_df = cf_df.drop(columns=["target"])
    for c in feat_cols:
        if c not in cf_df.columns:
            cf_df[c] = float(x0[c])
    return cf_df[feat_cols].astype(float)


def filter_valid_cfs(model: ModelWrapper, x0: pd.Series, cf_df: Optional[pd.DataFrame], k: int) -> Optional[pd.DataFrame]:
    if cf_df is None or cf_df.empty:
        return None
    x0v = x0.values.astype(float)[None, :]
    X = cf_df.values.astype(float)
    changed = ~np.isclose(X, x0v, rtol=1e-7, atol=1e-7)
    changed_cnt = changed.sum(axis=1)
    span_ok = (changed_cnt > 0) & (changed_cnt <= k)
    if not np.any(span_ok):
        return None
    cf_ok = cf_df.iloc[np.where(span_ok)[0]].copy()

    base_label = int(model.predict(x0v)[0])
    preds = model.predict(cf_ok.values)
    flip_mask = preds != base_label
    if not np.any(flip_mask):
        return None
    cf_ok = cf_ok.iloc[np.where(flip_mask)[0]].copy()
    prob = model.predict_proba(cf_ok.values)
    cf_ok["proba0"] = prob[:, 0]
    cf_ok["proba1"] = prob[:, 1]
    cf_ok["num_features_changed"] = changed_cnt[span_ok][flip_mask]
    return cf_ok


# ----------------------------- config -----------------------------

@dataclass
class GenCfg:
    max_features: int
    total_cfs: int
    lime_samples: int = 4000
    fallback_subsets: int = 50
    flip_threshold: float = 0.5
    seed: int = SEED
    # DiCE knobs
    dice_sample_size: int = 5000      # for method="random"
    dice_restarts: int = 24           # for method="kfeature"
    dice_grid_points: int = 10        # for method="kfeature"
    # Range tiers
    range_strategy: str = "global"    # 'global' or 'neighbor'
    range_schedule: str = "0.25:0.75,0.15:0.85,0.05:0.95,0.0:1.0"
    neighbor_k: int = 200
    ensure_contains_x0: bool = True


# ----------------------------- runner -----------------------------

def _out_path(project: str, model_type: str, method: str, total_cfs: int, max_features: int) -> Path:
    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"DiCE_all_{total_cfs}_max{max_features}feat.csv"


def generate_cf_for_project(project: str,
                            model_type: str,
                            method: str,
                            total_cfs: int,
                            max_features: int,
                            verbose: bool = True,
                            overwrite: bool = True,
                            cfg_overrides: Dict[str, Any] = None):

    valid_methods = ["random", "kdtree", "genetic", "kfeature"]
    if method not in valid_methods:
        tqdm.write(f"[{project}/{model_type}/{method}] Unsupported method '{method}'. Choose from {valid_methods}")
        return

    ds = read_dataset()
    if project not in ds:
        tqdm.write(f"[{project}/{model_type}] dataset not found. Skipping.")
        return

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    base_model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values.astype(float))
    model = ModelWrapper(base_model, scaler=scaler)

    # Keep your TP computation exactly as-is
    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}] no true positives. Skipping.")
        return

    # DiCE data/model
    all_data = pd.concat([train[feat_cols + ["target"]], test[feat_cols + ["target"]]], axis=0)
    dice_data = dice_ml.Data(dataframe=all_data, continuous_features=feat_cols, outcome_name="target")
    dice_model = dice_ml.Model(model=model, backend="sklearn")

    if method == "kfeature":
        try:
            from dice_ml.explainer_interfaces.dice_kfeature_search import DiceKFeatureSearch as KExplainer
            explainer = KExplainer(dice_data, dice_model)
        except Exception as e:
            tqdm.write(f"[ERROR] Could not init DiceKFeatureSearch: {e}")
            return
    else:
        try:
            explainer = Dice(dice_data, dice_model, method=method)
        except Exception as e:
            tqdm.write(f"[ERROR] Could not init DiCE explainer ('{method}'): {e}")
            return

    # LIME explainer on TRAIN
    lime_explainer = LimeTabularExplainer(
        training_data=train[feat_cols].values.astype(float),
        feature_names=feat_cols,
        class_names=["neg", "pos"],
        mode="classification",
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=SEED,
    )

    # config
    cfg = GenCfg(max_features=max_features, total_cfs=total_cfs)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    # neighbor index for range_strategy='neighbor'
    knn = None
    if cfg.range_strategy == "neighbor":
        Xz = scaler.transform(train[feat_cols].values.astype(float))
        knn = NearestNeighbors(
            n_neighbors=min(cfg.neighbor_k, max(1, len(Xz))),
            metric="euclidean"
        ).fit(Xz)

    # min-max (last fallback tier)
    minmax = feature_minmax(train, feat_cols)
    # parsed schedule
    tiers = parse_schedule(cfg.range_schedule)

    out_path = _out_path(project, model_type, method, total_cfs, max_features)
    if overwrite and out_path.exists():
        out_path.unlink(missing_ok=True)

    results = []
    misses = 0
    rng = np.random.default_rng(cfg.seed)

    # DiCE per-method kwargs
    extra_kwargs = {}
    if method == "random":
        extra_kwargs["sample_size"] = int(cfg.dice_sample_size)
    elif method == "kfeature":
        extra_kwargs.update(dict(restarts=int(cfg.dice_restarts),
                                 grid_points=int(cfg.dice_grid_points)))

    for idx in tqdm(tp_df.index.astype(int),
                    desc=f"{project}/{model_type}/{method} (DiCE+LIME, tiered ranges)",
                    leave=False, disable=not verbose):
        x0_row = test.loc[idx, feat_cols].astype(float)
        x0_df = x0_row.to_frame().T
        base_label = int(model.predict(x0_df.values)[0])
        target_label = 1 - base_label

        # LIME weights toward target
        w = lime_importance_for_target(
            explainer=lime_explainer,
            model=model,
            x0_row=x0_row,
            target_label=target_label,
            num_features=len(feat_cols),
            num_samples=cfg.lime_samples,
        )
        topk_feats = topk_from_weights(w, feat_cols, cfg.max_features)

        kept = None

        # ----- iterate across permitted-range tiers -----
        for (qlo, qhi) in tiers + [(0.0, 1.0)]:  # ensure final min-max attempt
            if cfg.range_strategy == "neighbor":
                prange = prange_neighbor_quantiles(
                    train, feat_cols, x0_row, scaler, knn, qlo, qhi,
                    ensure_contains_x0=cfg.ensure_contains_x0
                )
            else:
                prange = prange_global_quantiles(train, feat_cols, qlo, qhi)

            # 1) Top-K attempt under this tier
            cf_obj = call_dice_generate(
                explainer=explainer,
                x0_df=x0_df,
                features_to_vary=topk_feats,
                total_cfs=cfg.total_cfs,
                permitted_range=prange,
                stopping_threshold=cfg.flip_threshold,
                verbose=False,
                echo_dice=False,
                extra_kwargs=extra_kwargs,
                require_permitted=True,  # don't drop the prange
            )
            cf_df = normalize_cf_df(cf_obj, feat_cols, x0_row)
            kept = filter_valid_cfs(model, x0_row, cf_df, cfg.max_features)

            # 2) Weighted K-subset fallback (same tier)
            if kept is None or kept.empty:
                subsets = sample_weighted_k_subsets(w, feat_cols, cfg.max_features, cfg.fallback_subsets, rng)
                for subset in subsets:
                    cf_obj2 = call_dice_generate(
                        explainer=explainer,
                        x0_df=x0_df,
                        features_to_vary=list(subset),
                        total_cfs=cfg.total_cfs,
                        permitted_range=prange,
                        stopping_threshold=cfg.flip_threshold,
                        verbose=False,
                        echo_dice=False,
                        extra_kwargs=extra_kwargs,
                        require_permitted=True,
                    )
                    cf_df2 = normalize_cf_df(cf_obj2, feat_cols, x0_row)
                    kept2 = filter_valid_cfs(model, x0_row, cf_df2, cfg.max_features)
                    if kept2 is not None and not kept2.empty:
                        kept = kept2
                        break

            if kept is not None and not kept.empty:
                break  # success at this tier; stop widening

        if kept is None or kept.empty:
            misses += 1
            continue

        kept = kept.copy()
        kept.insert(0, "candidate_id", np.arange(len(kept)))
        kept.insert(0, "test_idx", int(idx))
        results.append(kept)

    if results:
        out_df = pd.concat(results, axis=0, ignore_index=True)
        out_df = out_df[["test_idx", "candidate_id"] + feat_cols + ["proba0", "proba1", "num_features_changed"]]
        out_df.to_csv(out_path, index=False)
        uniq = out_df["test_idx"].nunique()
        tqdm.write(
            f"[OK] {project}/{model_type}/{method}: wrote {len(out_df)} rows across "
            f"{uniq} TP(s) → {out_path} | misses={misses}"
        )
    else:
        tqdm.write(f"[{project}/{model_type}/{method}] no candidates found. misses={misses}")


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="DiCE+LIME with tiered permitted ranges (tight → wide)")
    ap.add_argument("--project", type=str, default="all")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated: random,kdtree,genetic,kfeature")
    ap.add_argument("--total_cfs", type=int, default=1)
    ap.add_argument("--max_features", type=int, default=5)
    ap.add_argument("--lime_samples", type=int, default=4000)
    ap.add_argument("--fallback_subsets", type=int, default=50)
    ap.add_argument("--flip_threshold", type=float, default=0.5)

    # DiCE knobs
    ap.add_argument("--dice_sample_size", type=int, default=5000)  # random
    ap.add_argument("--dice_restarts", type=int, default=24)       # kfeature
    ap.add_argument("--dice_grid_points", type=int, default=10)    # kfeature

    # Range tiers
    ap.add_argument("--range_strategy", type=str, choices=["global", "neighbor"], default="neighbor")
    ap.add_argument("--range_schedule", type=str, default="0.25:0.75,0.15:0.85,0.05:0.95,0.0:1.0")
    ap.add_argument("--neighbor_k", type=int, default=200)
    ap.add_argument("--ensure_contains_x0", action="store_true")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    projects = read_dataset()
    project_list = list(sorted(projects.keys())) if args.project == "all" else \
                   [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    cfg_overrides = dict(
        lime_samples=args.lime_samples,
        fallback_subsets=args.fallback_subsets,
        flip_threshold=args.flip_threshold,
        dice_sample_size=args.dice_sample_size,
        dice_restarts=args.dice_restarts,
        dice_grid_points=args.dice_grid_points,
        range_strategy=args.range_strategy,
        range_schedule=args.range_schedule,
        neighbor_k=args.neighbor_k,
        ensure_contains_x0=bool(args.ensure_contains_x0),
    )

    print(f"Running: {len(project_list)} projects × {len(args.model_types.split(','))} models × {len(methods)} methods")
    print(f"Range strategy: {args.range_strategy} | schedule: {args.range_schedule}")
    print(f"Output template: experiments/{{project}}/{{model}}/{{method}}/DiCE_all_{args.total_cfs}_max{args.max_features}feat.csv\n")

    for p in tqdm(project_list, desc="Projects", disable=not args.verbose):
        for mt in [m.strip() for m in args.model_types.split(",") if m.strip()]:
            for md in methods:
                generate_cf_for_project(
                    project=p,
                    model_type=mt,
                    method=md,
                    total_cfs=args.total_cfs,
                    max_features=args.max_features,
                    verbose=args.verbose,
                    overwrite=args.overwrite,
                    cfg_overrides=cfg_overrides,
                )

    print("Done!")


if __name__ == "__main__":
    main()
