#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project-level LIME importance + K-feature-capped counterfactuals (no DiCE).

Outputs:
- LIME summary per project/model:
    ./evaluations/lime/{project}/{model_type}/lime_summary.csv
- Counterfactuals per project/model (compatible with your downstream scripts):
    experiments/{project}/{model_type}/kfeature/DiCE_all_{TOTAL}_max{K}feat.csv
      columns: test_idx, candidate_id, <features...>, proba0, proba1, num_features_changed

Assumptions:
- Your helpers exist: read_dataset(), get_model(), get_true_positives()
- Binary classification; models may/may not expose predict_proba
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Your helpers
from data_utils import read_dataset, get_model, get_true_positives
try:
    from hyparams import SEED
except Exception:
    SEED = 42

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception as e:
    raise RuntimeError("Please install LIME: pip install lime") from e


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

class ProbModelWrapper:
    """Provide predict_proba(X)->[N,2] for any sklearn-like classifier."""
    def __init__(self, base_model):
        self.m = base_model

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.m, "predict_proba"):
            p = self.m.predict_proba(X)
            if p.ndim == 1:
                p1 = p
                return np.stack([1 - p1, p1], axis=1)
            if p.shape[1] == 2:
                return p
            p1 = np.max(p, axis=1)
            return np.stack([1 - p1, p1], axis=1)
        if hasattr(self.m, "decision_function"):
            s = self.m.decision_function(X)
            if np.ndim(s) > 1:
                s = s[:, 0]
            p1 = self._sigmoid(s)
            return np.stack([1 - p1, p1], axis=1)
        y = self.m.predict(X)
        p1 = (y == 1).astype(float)
        return np.stack([1 - p1, p1], axis=1)


@dataclass
class LimeConfig:
    discretizer: str = "entropy"
    feature_selection: str = "lasso_path"
    num_samples: int = 5000
    seed: int = SEED
    eval_cap: Optional[int] = 200   # number of instances to aggregate over (None = all TPs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------------------
# Project-level LIME importance
# --------------------------------------------------------------------------------------

def compute_project_lime_importance(train_df: pd.DataFrame,
                                    test_df: pd.DataFrame,
                                    model,
                                    *,
                                    only_true_positives: bool = True,
                                    cfg: LimeConfig = LimeConfig()
                                    ) -> pd.DataFrame:
    """
    Aggregate LIME importance across many instances (by default true positives),
    returning a single project-level ranking.

    Returns DataFrame: ['feature','mean_abs_weight','mean_weight','fraction_selected','count']
    """
    feat_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feat_cols].astype(float).values
    y_train = train_df["target"].astype(int).values

    if only_true_positives:
        tp_df = get_true_positives(model, train_df, test_df)
        X_eval_df = test_df.loc[tp_df.index, feat_cols].astype(float)
    else:
        X_eval_df = test_df[feat_cols].astype(float)

    if cfg.eval_cap is not None and len(X_eval_df) > cfg.eval_cap:
        X_eval_df = X_eval_df.iloc[: cfg.eval_cap]

    if len(X_eval_df) == 0:
        return pd.DataFrame(columns=["feature","mean_abs_weight","mean_weight","fraction_selected","count"])

    # Scale for LIME explainer; we pass inverse_transform into predict_fn
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    feature_names = feat_cols

    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=y_train,
        feature_names=feature_names,
        discretizer=cfg.discretizer,
        feature_selection=cfg.feature_selection,
        random_state=cfg.seed,
        mode="classification",
        class_names=[0, 1],
    )
    model_p = ProbModelWrapper(model)

    # Collect per-instance maps, then aggregate
    rows = []
    # Precompute predictions to set label we explain = opposite of predicted class
    P_eval = model_p.predict_proba(X_eval_df.values)
    y_pred = (P_eval[:, 1] >= 0.5).astype(int)
    for i, (idx, x_raw) in enumerate(X_eval_df.iterrows()):
        x_scaled = scaler.transform(x_raw.values.reshape(1, -1))[0]
        label_to_explain = 1 - int(y_pred[i])      # opposite-class explanation

        exp = explainer.explain_instance(
            data_row=x_scaled,
            predict_fn=lambda Xs: model_p.predict_proba(scaler.inverse_transform(Xs)),
            labels=[label_to_explain],
            num_samples=cfg.num_samples,
            num_features=len(feature_names),
        )
        fmap = dict(exp.as_map()[label_to_explain])  # {feature_index: weight}
        for j, w in fmap.items():
            rows.append((feature_names[j], float(w)))

    if not rows:
        return pd.DataFrame(columns=["feature","mean_abs_weight","mean_weight","fraction_selected","count"])

    df = pd.DataFrame(rows, columns=["feature","weight"])
    grouped = df.groupby("feature", as_index=False).agg(
        mean_abs_weight=("weight", lambda a: np.mean(np.abs(a))),
        mean_weight=("weight", "mean"),
        count=("weight", "count"),
    )
    n_instances = int(len(X_eval_df))
    grouped["fraction_selected"] = grouped["count"] / max(1, n_instances)
    grouped = grouped.sort_values(["mean_abs_weight","fraction_selected"], ascending=[False, False], kind="stable")
    return grouped.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Counterfactual generation using top-K features from project-level LIME
# --------------------------------------------------------------------------------------

@dataclass
class CFConfig:
    top_k: int = 5                 # K-feature cap
    total_cfs: int = 5             # max CFs to save per instance
    grid_points: int = 11          # per-feature grid for proposals (train quantiles)
    iter_limit: int = 60           # greedy steps cap
    restarts: int = 8              # additional K-subset restarts (weighted by importance)
    flip_threshold: float = 0.5    # target prob threshold
    seed: int = SEED


def _train_range_and_grids(train_col: np.ndarray, grid_points: int) -> Tuple[float, float, np.ndarray]:
    lo = float(np.min(train_col))
    hi = float(np.max(train_col))
    qs = np.linspace(0.0, 1.0, max(3, int(grid_points)))
    vals = np.unique(np.quantile(train_col.astype(float), qs, method="linear"))
    vals = np.clip(vals, lo, hi)
    return lo, hi, vals


def _best_1step_move(model_p: ProbModelWrapper,
                     x0: np.ndarray,
                     current: np.ndarray,
                     target_class: int,
                     feat_idxs: List[int],
                     grids: List[np.ndarray],
                     lohi: List[Tuple[float,float]]) -> Tuple[Optional[np.ndarray], float]:
    """
    Evaluate all 1-feature moves across given feature indices.
    Return (best_candidate, best_gain) or (None, 0.0).
    """
    base = float(model_p.predict_proba(current.reshape(1, -1))[0, target_class])
    best_gain, best_vec = 0.0, None
    for j, fidx in enumerate(feat_idxs):
        lo, hi = lohi[j]
        for v in grids[j]:
            if np.isclose(v, current[fidx], rtol=1e-9, atol=1e-12):
                continue
            cand = current.copy()
            cand[fidx] = float(np.clip(v, lo, hi))
            pt = float(model_p.predict_proba(cand.reshape(1, -1))[0, target_class])
            gain = pt - base
            if gain > best_gain + 1e-12:
                best_gain, best_vec = gain, cand
    return best_vec, best_gain


def _greedy_cf_search(model_p: ProbModelWrapper,
                      x0: np.ndarray,
                      target_class: int,
                      feat_idxs: List[int],
                      train_df: pd.DataFrame,
                      grid_points: int,
                      iter_limit: int) -> Optional[np.ndarray]:
    """
    Greedy coordinate ascent over selected features using train-based quantile grids.
    """
    # Build per-feature grids from training data ranges/quantiles
    grids = []
    lohi = []
    for fidx in feat_idxs:
        lo, hi, vals = _train_range_and_grids(train_df.iloc[:, fidx].values, grid_points)
        # include original point to help polish/monotonicity
        vals = np.unique(np.append(vals, x0[fidx]))
        grids.append(vals)
        lohi.append((lo, hi))

    cur = x0.copy()
    for _ in range(iter_limit):
        nxt, gain = _best_1step_move(model_p, x0, cur, target_class, feat_idxs, grids, lohi)
        if nxt is None or gain <= 1e-12:
            break
        cur = nxt
        pt = float(model_p.predict_proba(cur.reshape(1, -1))[0, target_class])
        if pt >= 0.5:
            return cur
    return None

import numpy as np
from sklearn.neighbors import NearestNeighbors

def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def snap_to_quantiles(x, train_col, n=11):
    lo, hi = float(np.min(train_col)), float(np.max(train_col))
    qs = np.linspace(0.0, 1.0, max(3, int(n)))
    grid = np.unique(np.quantile(train_col.astype(float), qs, method="linear"))
    grid = np.clip(grid, lo, hi)
    # snap each scalar to nearest grid value
    return grid[np.argmin(np.abs(grid - x))]

def anchored_line_search(x0, z, feat_idx, model, X_train, target_class,
                         flip_thresh=0.5, n_quant=11, max_iter=20):
    """Bisection on the line from x0 to z projected to feat_idx; returns closest flip or None."""
    x0 = x0.astype(float).copy()
    d = z - x0
    mask = np.zeros_like(x0, dtype=bool); mask[feat_idx] = True
    d[~mask] = 0.0

    lo, hi = 0.0, 1.0
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x = x0 + mid * d
        # snap active coords to nearest train quantiles and clip to train range
        for j in np.where(mask)[0]:
            x[j] = snap_to_quantiles(x[j], X_train[:, j], n_quant)
            x[j] = np.clip(x[j], np.min(X_train[:, j]), np.max(X_train[:, j]))
        proba = model.predict_proba(x.reshape(1, -1))[0]
        p_t = float(proba[target_class])
        if p_t >= flip_thresh:
            best = x.copy()
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-3:
            break
    return best

def ksparse_trust_region_cf(x0, model, X_train, target_class, feat_idx_topK,
                            k_neighbors=20, trust_r_init=0.5, trust_r_min=0.05,
                            max_iters=60, n_quant=11, flip_thresh=0.5, seed=42):
    """
    Model-agnostic K-sparse trust-region CF with anchor seeding.
    x0: (d,) ndarray
    model: object with predict_proba(X)->[N,2]
    X_train: (N,d)
    target_class: 0 or 1
    feat_idx_topK: list[int] candidate feature indices (|S|>=K; we will keep K each step)
    """
    rng = np.random.default_rng(seed)
    d = x0.shape[0]

    # 1) Anchor seeding: nearest neighbors of target class in train
    # If labels not available here, pass in prefiltered negatives; otherwise approximate with predicted class.
    P_tr = model.predict_proba(X_train)
    y_tr = (P_tr[:, 1] >= 0.5).astype(int)
    anchor_pool = X_train[y_tr == target_class]
    if len(anchor_pool) == 0:
        anchor_pool = X_train  # fallback

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(anchor_pool))).fit(anchor_pool)
    _, idxs = nn.kneighbors(x0.reshape(1, -1))
    anchors = anchor_pool[idxs[0]]

    # Try bisection toward a few nearest anchors
    for z in anchors[:min(5, len(anchors))]:
        # cf = anchored_line_search(x0, z, feat_idx_topK, model, X_train, flip_thresh, n_quant)
        cf = anchored_line_search(
            x0, z, feat_idx_topK, model, X_train,
            target_class=target_class,       # <-- pass the class explicitly
            flip_thresh=flip_thresh,
            n_quant=n_quant
        )

        if cf is not None:
            return cf  # already a very close flip

    # 2) K-sparse trust-region SPSA
    x = x0.copy()
    trust_r = trust_r_init
    lam = 1.0  # penalty multiplier
    K = len(feat_idx_topK)

    def loss(x):
        p1 = float(model.predict_proba(x.reshape(1, -1))[0, 1])
        # we drive toward target_class with hinge on the opposite probability
        p_t = p1 if target_class == 1 else 1.0 - p1
        hinge = max(0.0, flip_thresh - p_t)
        # z-score-ish distance using train std
        std = np.std(X_train, axis=0) + 1e-12
        dist = np.sum(np.abs((x - x0) / std))
        return lam * hinge + 0.05 * dist, p_t

    std = np.std(X_train, axis=0) + 1e-12

    for it in range(max_iters):
        # SPSA gradient estimate restricted to candidate coords
        delta = np.zeros_like(x)
        # Random support over feat_idx_topK but keep exactly K nonzeros (K-sparse)
        support = np.array(feat_idx_topK, dtype=int)
        # Rademacher +/-1
        delta[support] = rng.choice([-1.0, 1.0], size=len(support))
        step = trust_r * std  # scale by feature scale
        x_plus  = x + step * delta
        x_minus = x - step * delta

        # snap to quantiles & clip to train range
        for arr in (x_plus, x_minus):
            for j in support:
                arr[j] = snap_to_quantiles(arr[j], X_train[:, j], n_quant)
                arr[j] = np.clip(arr[j], np.min(X_train[:, j]), np.max(X_train[:, j]))

        f_plus,  p_t_plus  = loss(x_plus)
        f_minus, p_t_minus = loss(x_minus)
        # SPSA gradient (finite-difference along random direction)
        g_hat = (f_plus - f_minus) / (2.0 * trust_r + 1e-12)
        # Move opposite to gradient sign on K coords
        direction = -np.sign(g_hat)  # scalar: sign of 1D directional derivative
        # convert back into vector direction = delta * sign
        grad_vec = direction * delta

        # project to K-sparse: already K-sparse because we limit to 'support'
        x_new = x + 0.5 * trust_r * std * grad_vec

        # snap & clip
        for j in support:
            x_new[j] = snap_to_quantiles(x_new[j], X_train[:, j], n_quant)
            x_new[j] = np.clip(x_new[j], np.min(X_train[:, j]), np.max(X_train[:, j]))

        f_new, p_t_new = loss(x_new)
        f_cur, p_t_cur = loss(x)

        if f_new <= f_cur:
            x = x_new
            # success: slightly expand trust region
            trust_r = min(1.5 * trust_r, 1.0)
        else:
            # failure: shrink trust region
            trust_r = max(trust_r * 0.5, trust_r_min)
            lam *= 1.05  # increase pressure to flip

        if p_t_new >= flip_thresh:
            # quick polish: try snapping changed coords toward original
            changed = np.where(~np.isclose(x, x0, rtol=1e-7, atol=1e-7))[0]
            for j in changed:
                trial = x.copy()
                trial[j] = snap_to_quantiles(x0[j], X_train[:, j], n_quant)
                trial[j] = np.clip(trial[j], np.min(X_train[:, j]), np.max(X_train[:, j]))
                p_t_trial = float(model.predict_proba(trial.reshape(1, -1))[0, target_class])
                if p_t_trial >= flip_thresh:
                    x = trial
            return x

    return None  # no flip found

def generate_cfs_with_kcap(train_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           model,
                           lime_summary: pd.DataFrame,
                           *,
                           cf_cfg: CFConfig = CFConfig()) -> pd.DataFrame:
    """
    For each true positive in test_df, pick top-K features from project LIME and try, in order:
      1) Anchor line search toward nearest target-class train anchors (closest flips)
      2) K-sparse trust-region SPSA (model-agnostic)
      3) Greedy 1-step ascent (last resort)
    Returns a long DataFrame with CF rows.
    """
    feat_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feat_cols].astype(float).values
    X_test_df = test_df[feat_cols].astype(float)

    # feature ranking from project-level LIME
    ranked = [f for f in lime_summary["feature"].tolist() if f in feat_cols]
    if not ranked:
        return pd.DataFrame(columns=["test_idx","candidate_id"] + feat_cols + ["proba0","proba1","num_features_changed"])

    topK_feats = ranked[: min(cf_cfg.top_k, len(ranked))]
    topK_idx = [feat_cols.index(f) for f in topK_feats]

    # model wrapper
    model_p = ProbModelWrapper(model)

    # true positives in test split
    tp_df = get_true_positives(model, train_df, test_df)
    if tp_df.empty:
        return pd.DataFrame(columns=["test_idx","candidate_id"] + feat_cols + ["proba0","proba1","num_features_changed"])

    # Build an anchor pool once: nearest training points predicted as target class
    P_tr = model_p.predict_proba(X_train)
    y_tr_pred = (P_tr[:, 1] >= 0.5).astype(int)

    rng = np.random.default_rng(cf_cfg.seed)
    rows = []

    # importance weights for restarts (safe reindex; fill missing with tiny value)
    if not lime_summary.empty:
        w_series = lime_summary.set_index("feature")["mean_abs_weight"].reindex(feat_cols).fillna(1e-9)
        weights = (w_series.values / np.sum(w_series.values)).astype(float)
    else:
        weights = np.ones(len(feat_cols), dtype=float) / max(1, len(feat_cols))

    from sklearn.neighbors import NearestNeighbors

    for n_i, idx in enumerate(tp_df.index.astype(int)):
        x0 = X_test_df.loc[idx].values.astype(float)

        # target class is opposite of current prediction
        p0 = model_p.predict_proba(x0.reshape(1, -1))[0]
        y_pred = int(p0[1] >= 0.5)
        target = 1 - y_pred  # for TPs this should be 0

        # anchors of target class
        anchor_pool = X_train[y_tr_pred == target]
        if len(anchor_pool) == 0:
            anchor_pool = X_train  # fallback if no target-class anchors predicted
        nn = NearestNeighbors(n_neighbors=min(20, len(anchor_pool))).fit(anchor_pool)
        _, nn_idx = nn.kneighbors(x0.reshape(1, -1))
        anchors = anchor_pool[nn_idx[0]]

        found: List[np.ndarray] = []

        # ---- Attempt 1: anchor line-search on top-K ----
        for z in anchors[:min(8, len(anchors))]:
            cf = anchored_line_search(x0, z, topK_idx, model_p, X_train,
                                      target_class=target,
                                      flip_thresh=cf_cfg.flip_threshold,
                                      n_quant=cf_cfg.grid_points,
                                      max_iter=30)
            if cf is not None:
                found.append(cf)
                if len(found) >= cf_cfg.total_cfs:
                    break

        # ---- Attempt 2: importance-weighted K-subset restarts with trust-region (if still needed) ----
        r = 0
        while len(found) < cf_cfg.total_cfs and r < max(0, cf_cfg.restarts - 1):
            subset = rng.choice(np.arange(len(feat_cols)),
                                size=min(cf_cfg.top_k, len(feat_cols)),
                                replace=False, p=weights)
            # Avoid identical to topK to get diversity
            if set(subset.tolist()) == set(topK_idx) and cf_cfg.restarts > 1:
                r += 1
                continue

            cf_tr = ksparse_trust_region_cf(
                x0=x0, model=model_p, X_train=X_train, target_class=target,
                feat_idx_topK=subset.tolist(),
                k_neighbors=20, trust_r_init=0.4, trust_r_min=0.05,
                max_iters=60, n_quant=cf_cfg.grid_points, flip_thresh=cf_cfg.flip_threshold,
                seed=int(rng.integers(0, 2**31-1))
            )
            if cf_tr is not None:
                found.append(cf_tr)
            r += 1

        # ---- Attempt 3: greedy fallback on top-K (guarantee a try) ----
        if len(found) == 0:
            cf_g = _greedy_cf_search(
                model_p, x0, target, topK_idx, train_df[feat_cols], cf_cfg.grid_points, cf_cfg.iter_limit
            )
            if cf_g is not None:
                found.append(cf_g)

        # Pack rows
        cand_id = 0
        for cf in found[: cf_cfg.total_cfs]:
            proba = model_p.predict_proba(cf.reshape(1, -1))[0]
            changed = ~np.isclose(cf, x0, rtol=1e-7, atol=1e-7)
            num_changed = int(np.sum(changed))
            if num_changed == 0 or num_changed > cf_cfg.top_k:
                continue
            row = {
                "test_idx": int(idx),
                "candidate_id": int(cand_id),
                "proba0": float(proba[0]),
                "proba1": float(proba[1]),
                "num_features_changed": int(num_changed),
            }
            for c, v in zip(feat_cols, cf.astype(float)):
                row[c] = float(v)
            rows.append(row)
            cand_id += 1

    if not rows:
        return pd.DataFrame(columns=["test_idx","candidate_id"] + feat_cols + ["proba0","proba1","num_features_changed"])
    out = pd.DataFrame(rows)
    out = out[["test_idx","candidate_id"] + feat_cols + ["proba0","proba1","num_features_changed"]]
    return out


# --------------------------------------------------------------------------------------
# Orchestrator / CLI
# --------------------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Project-level LIME importance + K-capped CF generation")
    ap.add_argument("--projects", type=str, default="all", help="Project name(s) or 'all'")
    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    # LIME
    ap.add_argument("--lime_samples", type=int, default=5000, help="LIME num_samples per instance")
    ap.add_argument("--lime_eval_cap", type=int, default=200, help="Instances to aggregate for project-level LIME (TPs)")
    # CF
    ap.add_argument("--top_k", type=int, default=5, help="Max features changed per CF")
    ap.add_argument("--total_cfs", type=int, default=5, help="Max CFs per TP to save")
    ap.add_argument("--grid_points", type=int, default=11, help="Quantile grid points per feature")
    ap.add_argument("--iter_limit", type=int, default=60, help="Greedy steps cap")
    ap.add_argument("--restarts", type=int, default=8, help="Extra K-subset restarts (importance-weighted)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ds = read_dataset()
    if args.projects == "all":
        project_list = list(sorted(ds.keys()))
    else:
        project_list = [p.strip() for p in args.projects.replace(",", " ").split() if p.strip()]
    model_types = [m.strip() for m in args.models.replace(",", " ").split() if m.strip()]

    print(f"Running for {len(project_list)} projects × {len(model_types)} models")
    for project in project_list:
        train, test = ds[project]
        feat_cols = [c for c in train.columns if c != "target"]

        for model_type in model_types:
            model = get_model(project, model_type)

            # 1) Project-level LIME importance (aggregate over TPs)
            lime_cfg = LimeConfig(
                num_samples=int(args.lime_samples),
                seed=SEED,
                eval_cap=args.lime_eval_cap
            )
            summary = compute_project_lime_importance(train, test, model, only_true_positives=True, cfg=lime_cfg)
            out_dir_lime = os.path.join("evaluations", "lime", project, model_type)
            ensure_dir(out_dir_lime)
            sum_path = os.path.join(out_dir_lime, "lime_summary.csv")
            summary.to_csv(sum_path, index=False)

            if args.verbose:
                print(f"[LIME] {project}/{model_type}: kept {len(summary)} features in summary → {sum_path}")
                if not summary.empty:
                    print(summary.head(10).to_string(index=False))

            # 2) Counterfactuals using top-K features from LIME
            cf_cfg = CFConfig(
                top_k=int(args.top_k),
                total_cfs=int(args.total_cfs),
                grid_points=int(args.grid_points),
                iter_limit=int(args.iter_limit),
                restarts=int(args.restarts),
                flip_threshold=0.5,
                seed=SEED,
            )
            cf_df = generate_cfs_with_kcap(train, test, model, summary, cf_cfg=cf_cfg)

            out_dir_cf = os.path.join("experiments", project, model_type, "kfeature")
            ensure_dir(out_dir_cf)
            out_csv = os.path.join(out_dir_cf, f"DiCE_all_{cf_cfg.total_cfs}_max{cf_cfg.top_k}feat.csv")
            cf_df.to_csv(out_csv, index=False)

            print(f"[CF] {project}/{model_type}: wrote {len(cf_df)} rows across "
                  f"{(cf_df['test_idx'].nunique() if not cf_df.empty else 0)} TP(s) → {out_csv}")

if __name__ == "__main__":
    main()
