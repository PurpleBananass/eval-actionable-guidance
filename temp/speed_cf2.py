#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIME-only CF generator — minimal build (baseline enumeration + flip-preserving refine + fallback + rescue)

Design goals (defaults tuned for low Mahalanobis without hurting flip rate):
- Keep the original enumeration-based search guided by LIME feature weights.
- Preserve flips via a light refine step (drop unnecessary edits + binary search toward original).
- Use a stochastic fallback only if enumeration fails.
- Use a small 1D/2D rescue for stubborn instances.
- Default to *local* value pools (neighbor quantiles) and avoid dataset extremes.

Output:
  experiments/{project}/{model}/{method}/DiCE_all_{TOTAL}_max{K}feat.csv
"""

from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, List, Optional, Tuple
from itertools import combinations, product

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# for tree-like detection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    print("ERROR: python package 'lime' is not installed. Install with: pip install lime")
    sys.exit(1)

# project helpers
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED


# ----------------------------- utils -----------------------------

def round_key(x: NDArray[np.float_], n: int = 12) -> Tuple:
    return tuple(np.round(x.astype(float), n).tolist())

def mad(arr: NDArray[np.float_], c: float = 1.4826) -> float:
    """Median Absolute Deviation (robust scale) with Gaussian consistency factor."""
    med = np.median(arr)
    return float(c * np.median(np.abs(arr - med)) + 1e-12)

class UnitTransformer:
    """Robust z-units: (x - median) / MAD computed on TRAIN."""
    def __init__(self, X_train: NDArray[np.float_]):
        self.median = np.median(X_train, axis=0)
        self.mad = np.array([mad(X_train[:, i]) for i in range(X_train.shape[1])])

    def to_units(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        return (X - self.median) / self.mad

    def from_units(self, U: NDArray[np.float_]) -> NDArray[np.float_]:
        return U * self.mad + self.median

class ModelWrapper:
    """Wrap a binary classifier; optionally scale with StandardScaler fitted on TRAIN features."""
    def __init__(self, model: Any, scaler: Optional[StandardScaler]):
        self.model = model
        self.scaler = scaler

    def _prep(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        return self.scaler.transform(X) if self.scaler is not None else X

    def predict_proba(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        Xp = self._prep(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xp)
            if proba.ndim == 1:
                p1 = proba
                return np.stack([1 - p1, p1], axis=1)
            return proba
        if hasattr(self.model, "decision_function"):
            m = self.model.decision_function(Xp)
            if np.ndim(m) == 1:
                p1 = 1.0 / (1.0 + np.exp(-m))
                return np.stack([1 - p1, p1], axis=1)
        y = self.model.predict(Xp)
        p0 = (y == 0).astype(float)
        return np.stack([p0, 1.0 - p0], axis=1)

    def predict_label(self, X: NDArray[np.float_]) -> NDArray[np.int_]:
        Xp = self._prep(X)
        try:
            y = self.model.predict(Xp)
            return np.asarray(y, dtype=int)
        except Exception:
            P = self.predict_proba(X)
            return (P[:, 1] >= 0.5).astype(int)

class NeighborIndex:
    def __init__(self, X_units: NDArray[np.float_], k: int = 300):
        self.nn = NearestNeighbors(n_neighbors=min(k, max(1, len(X_units))), metric="euclidean").fit(X_units)
        self.Xu = X_units

    def neighbors(self, x_units: NDArray[np.float_]) -> NDArray[np.float_]:
        _, idx = self.nn.kneighbors(x_units.reshape(1, -1), return_distance=True)
        return self.Xu[idx[0]]


# ----------------------------- config -----------------------------

@dataclass
class GenCfg:
    # cardinality controls
    max_features: int
    total_cfs: int

    # search pool size (defaults tuned to your “working” recipe)
    neighbor_k: int = 500
    quantiles_n: int = 41
    per_feature_cap: int = 64
    include_extremes: bool = False  # keep pools local; reduces Mahalanobis
    feature_pool_factor: float = 1.2
    feature_pool_abs_max: Optional[int] = None

    # budgets
    instance_time_sec: float = 240.0
    eval_cap: int = 800_000

    # enumeration throttles
    max_product_per_combo: int = 4000
    max_combos_per_k: int = 3000
    top_per_feature_k_gt1: int = 8

    # stochastic fallback
    random_iters: int = 800_000
    rand_subset_min: int = 1
    rand_subset_max: Optional[int] = None  # None → use max_features

    # refine (flip-preserving)
    refine: bool = True
    refine_binary_steps: int = 36
    refine_drop_passes: int = 5

    # rescue pass
    rescue_enable: bool = True
    rescue_k_features: int = 3
    rescue_bisect_steps: int = 28

    # LIME
    lime_samples: int = 8000
    seed: int = SEED


# ----------------------------- core -----------------------------

class MinChangeCF_LIME:
    """
    Pipeline:
      1) LIME-guided enumeration by Hamming weight over local value pools (neighbor quantiles).
      2) Flip-preserving refine: drop unnecessary edits, then 1D binary search back toward original.
      3) Stochastic fallback if enumeration fails.
      4) Small 1D/2D rescue for stubborn cases.

    No hard radius gates, no history/delta guidance, no gain-ranking reorder.
    """

    def __init__(self,
                 model: ModelWrapper,
                 X_train: NDArray[np.float_],
                 feat_names: List[str],
                 lime_explainer: LimeTabularExplainer,
                 cfg: GenCfg):
        self.model = model
        self.cfg = cfg
        self.feat_names = feat_names
        self.Xtr = X_train.astype(float)

        self.ut = UnitTransformer(self.Xtr)
        self.train_min = np.min(self.Xtr, axis=0).astype(float)
        self.train_max = np.max(self.Xtr, axis=0).astype(float)
        Xu = self.ut.to_units(self.Xtr)
        self.nn_all = NeighborIndex(Xu, k=max(50, cfg.neighbor_k))
        self.rng = np.random.default_rng(cfg.seed)

        self.lime = lime_explainer
        self.last_diag: Dict[str, Any] = {}

    # ---- helpers ----
    def _predict_is_target0(self, x: np.ndarray) -> bool:
        """
        EXACT flip rule (as requested):
        A candidate is a flip iff predict_proba(... )[:, 0] >= 0.5
        """
        p0 = float(self.model.predict_proba(x.reshape(1, -1))[0, 0])
        return p0 >= 0.5

    def _lime_importance(self, x0_row: pd.Series, d: int) -> np.ndarray:
        exp = self.lime.explain_instance(
            data_row=x0_row.values.astype(float),
            predict_fn=self.model.predict_proba,
            labels=[0],  # target = column 0
            num_features=d,
            num_samples=self.cfg.lime_samples,
        )
        weights = np.zeros(d, dtype=float)
        for fid, w in exp.as_map().get(0, []):
            if 0 <= fid < d:
                weights[int(fid)] = float(abs(w))
        if np.all(weights <= 0):
            weights = np.ones(d, dtype=float)
        return weights / (weights.sum() + 1e-12)

    def _neighbor_quantiles(self, x_orig: np.ndarray, j: int, n: int) -> np.ndarray:
        x_u = self.ut.to_units(x_orig.reshape(1, -1))[0]
        Xn_u = self.nn_all.neighbors(x_u)
        Xn = self.ut.from_units(Xn_u) if len(Xn_u) > 0 else self.Xtr
        lo, hi = float(self.train_min[j]), float(self.train_max[j])
        col = np.clip(Xn[:, j].astype(float), lo, hi)
        qs = np.linspace(0.0, 1.0, max(3, int(n)))
        vals = np.unique(np.quantile(col, qs, method="linear"))
        vals = np.clip(vals, lo, hi)
        if self.cfg.include_extremes:
            vals = np.unique(np.concatenate([vals, [lo, hi]]))
        return vals

    def _sorted_by_unit_closeness(self, j: int, x0_val: float, candidates: Iterable[float]) -> List[float]:
        cand = sorted(set(float(c) for c in candidates))
        med, madj = float(self.ut.median[j]), float(self.ut.mad[j])
        x0u = (x0_val - med) / madj
        return sorted(cand, key=lambda v: abs(((v - med) / madj) - x0u))

    def _build_value_pools(self, x0: np.ndarray, use_idx: List[int], weights: np.ndarray) -> Dict[int, List[float]]:
        pools: Dict[int, List[float]] = {}
        for j in use_idx:
            grid = list(self._neighbor_quantiles(x0, j, self.cfg.quantiles_n))
            ordered = self._sorted_by_unit_closeness(j, float(x0[j]), grid)
            ordered = [v for v in ordered if not np.isclose(v, float(x0[j]))]
            lo, hi = float(self.train_min[j]), float(self.train_max[j])
            ordered = [float(np.clip(v, lo, hi)) for v in ordered]
            # dedup preserving order
            dedup, seen = [], set()
            for v in ordered:
                k = round(v, 12)
                if k in seen:
                    continue
                seen.add(k)
                dedup.append(v)
            pools[j] = dedup[: max(1, int(self.cfg.per_feature_cap))]
        return pools

    # ---------------- refinement (flip-preserving) ----------------
    def _refine_flip(self, x0: np.ndarray, cf: np.ndarray) -> np.ndarray:
        if not self.cfg.refine:
            return cf.copy()
        z = cf.copy()

        # A) drop changed dims back to original if flip holds
        for _ in range(max(1, int(self.cfg.refine_drop_passes))):
            changed = np.where(~np.isclose(z, x0, rtol=1e-7, atol=1e-7))[0]
            if changed.size == 0:
                break
            order = list(changed[np.argsort(np.abs(z[changed] - x0[changed]))])
            improved = False
            for j in order:
                x_try = z.copy()
                x_try[j] = float(x0[j])
                if self._predict_is_target0(x_try):
                    z = x_try
                    improved = True
            if not improved:
                break

        # B) 1D binary search toward original on remaining changed dims
        changed = np.where(~np.isclose(z, x0, rtol=1e-7, atol=1e-7))[0]
        for j in changed:
            a = float(x0[j])
            b = float(z[j])
            if not self._predict_is_target0(z):
                continue
            low, high = a, b
            best = high
            x_test = z.copy()
            x_test[j] = a
            if self._predict_is_target0(x_test):
                z[j] = a
                continue
            for _ in range(max(1, int(self.cfg.refine_binary_steps))):
                mid = 0.5 * (low + high)
                if np.isclose(mid, high, rtol=1e-10, atol=1e-12):
                    break
                x_mid = z.copy()
                lo, hi = float(self.train_min[j]), float(self.train_max[j])
                x_mid[j] = float(np.clip(mid, lo, hi))
                if self._predict_is_target0(x_mid):
                    best = x_mid[j]
                    high = mid
                    z[j] = best
                else:
                    low = mid

        return z

    # ---------------- main explain ----------------
    def explain(self, x0_row: pd.Series, x0: NDArray[np.float_]) -> List[NDArray[np.float_]]:
        self.last_diag = {}
        start_time = time.time()
        d = x0.shape[0]

        weights = self._lime_importance(x0_row, d)
        ranked = list(np.argsort(-weights))

        # Prepare pools
        pool_size = int(max(1, round(self.cfg.feature_pool_factor * self.cfg.max_features)))
        if self.cfg.feature_pool_abs_max is not None:
            pool_size = min(pool_size, int(self.cfg.feature_pool_abs_max))
        pool_size = min(pool_size, len(ranked))
        order = list(ranked[:pool_size])
        pools = self._build_value_pools(x0, order, weights)

        flips: List[NDArray[np.float_]] = []
        seen = set()
        evals = 0

        def push_cf(z: np.ndarray):
            key = round_key(z, 12)
            if key in seen:
                return False
            seen.add(key)
            flips.append(z.astype(float))
            return True

        # 1) Enumeration by Hamming weight
        for k in range(1, self.cfg.max_features + 1):
            if time.time() - start_time >= self.cfg.instance_time_sec or evals >= self.cfg.eval_cap or len(flips) >= self.cfg.total_cfs:
                break

            combos_tried = 0
            for idx_tuple in combinations(order, k):
                if time.time() - start_time >= self.cfg.instance_time_sec or evals >= self.cfg.eval_cap or len(flips) >= self.cfg.total_cfs:
                    break
                if combos_tried >= self.cfg.max_combos_per_k:
                    break
                if any(j not in pools or len(pools[j]) == 0 for j in idx_tuple):
                    continue

                value_lists = []
                for j in idx_tuple:
                    base = float(x0[j])
                    vals = [base] + pools[j]
                    if k >= 2:
                        vals = vals[: (1 + self.cfg.top_per_feature_k_gt1)]
                    # dedup preserving order
                    dedup, sv = [], set()
                    for v in vals:
                        key = round(v, 12)
                        if key in sv:
                            continue
                        sv.add(key)
                        dedup.append(v)
                    value_lists.append(dedup)

                # throttle product
                def prod_len(vls):
                    p = 1
                    for vv in vls:
                        p *= max(1, len(vv))
                    return p

                while prod_len(value_lists) > self.cfg.max_product_per_combo:
                    i = int(np.argmax([len(v) for v in value_lists]))
                    if len(value_lists[i]) > 1:
                        value_lists[i] = value_lists[i][:-1]
                    else:
                        break

                for tup in product(*value_lists):
                    x = x0.copy()
                    changed = False
                    for i, j in enumerate(idx_tuple):
                        v = float(tup[i])
                        if not np.isclose(v, float(x0[j])):
                            changed = True
                        lo, hi = float(self.train_min[j]), float(self.train_max[j])
                        x[j] = float(np.clip(v, lo, hi))
                    if not changed:
                        continue

                    evals += 1
                    if self._predict_is_target0(x):
                        z = self._refine_flip(x0, x)
                        push_cf(z)
                        if len(flips) >= self.cfg.total_cfs:
                            self.last_diag = dict(phase="enum", evals=evals, reason="hit")
                            return flips[: self.cfg.total_cfs]
                combos_tried += 1

        if len(flips) >= self.cfg.total_cfs:
            return flips[: self.cfg.total_cfs]

        # 2) Stochastic fallback
        probs_full = np.ones(x0.shape[0], dtype=float) / max(1, x0.shape[0])
        kmin = max(1, int(self.cfg.rand_subset_min))
        kmax = int(self.cfg.rand_subset_max or self.cfg.max_features)
        kmax = max(kmin, min(kmax, self.cfg.max_features))
        features_all = np.arange(x0.shape[0], dtype=int)

        it = 0
        while it < self.cfg.random_iters:
            if time.time() - start_time >= self.cfg.instance_time_sec or evals >= self.cfg.eval_cap or len(flips) >= self.cfg.total_cfs:
                break
            it += 1
            k = int(self.rng.integers(low=kmin, high=kmax + 1))
            k = min(k, features_all.size)
            if k <= 0:
                continue
            idx = self.rng.choice(features_all, size=k, replace=False, p=probs_full)

            x = x0.copy()
            valid = True
            for j in idx:
                cand_vals = pools.get(int(j))
                if not cand_vals:
                    grid = list(self._neighbor_quantiles(x0, int(j), max(11, self.cfg.quantiles_n // 2)))
                    ordered = self._sorted_by_unit_closeness(int(j), float(x0[int(j)]), grid)
                    ordered = [v for v in ordered if not np.isclose(v, float(x0[int(j)]))]
                    lo, hi = float(self.train_min[int(j)]), float(self.train_max[int(j)])
                    cand_vals = [float(np.clip(v, lo, hi)) for v in ordered][: max(8, min(32, self.cfg.per_feature_cap))]
                    if self.cfg.include_extremes:
                        cand_vals = list(dict.fromkeys(([lo] + cand_vals + [hi])))
                    if not cand_vals:
                        valid = False
                        break
                    pools[int(j)] = cand_vals
                # pick near-head 80%, random 15%, far-extreme 5%
                rnum = self.rng.random()
                head = min(5, len(cand_vals))
                if rnum < 0.80:
                    pick_idx = int(self.rng.integers(low=0, high=head))
                elif rnum < 0.95:
                    pick_idx = int(self.rng.integers(low=0, high=len(cand_vals)))
                else:
                    pick_idx = 0 if abs(cand_vals[0] - x0[j]) > abs(cand_vals[-1] - x0[j]) else (len(cand_vals) - 1)
                v = float(cand_vals[pick_idx])
                lo, hi = float(self.train_min[int(j)]), float(self.train_max[int(j)])
                x[int(j)] = float(np.clip(v, lo, hi))

            if not valid:
                continue

            evals += 1
            if self._predict_is_target0(x):
                z = self._refine_flip(x0, x)
                push_cf(z)
                self.last_diag = dict(phase="fallback", evals=evals, iters=it, reason="hit")
                return flips[: self.cfg.total_cfs]

        # 3) Final rescue (1D/2D bisection)
        if (not flips) and self.cfg.rescue_enable:
            z = self._rescue_minimal_flip(x0, weights=self._lime_importance(x0_row, x0.shape[0]))
            if z is not None:
                z = self._refine_flip(x0, z)
                push_cf(z)
                self.last_diag = dict(phase="rescue", reason="hit")
                return flips[: self.cfg.total_cfs]

        self.last_diag = dict(phase="done", reason="no-cf", evals=evals)
        return []

    # ---------------- rescue helpers ----------------
    def _rescue_minimal_flip(self, x0: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
        ranked = list(np.argsort(-weights))
        K = int(min(max(1, self.cfg.rescue_k_features), len(ranked)))

        # 1-D lines
        for j in ranked[:K]:
            grid = self._neighbor_quantiles(x0, j, max(31, 9))
            vals = [v for v in grid if not np.isclose(v, float(x0[j]))]
            lo, hi = float(self.train_min[j]), float(self.train_max[j])
            vals = [float(np.clip(v, lo, hi)) for v in vals]
            vals = sorted(vals, key=lambda v: abs(v - float(x0[j])))
            for tval in vals[:8] + ([vals[0], vals[-1]] if len(vals) >= 2 else []):
                a, b = float(x0[j]), float(tval)
                xa = x0.copy()
                xb = x0.copy(); xb[j] = b
                if not self._predict_is_target0(xb):
                    continue
                # bisection back toward a
                lo_t, hi_t = 0.0, 1.0
                best = xb.copy()
                for _ in range(max(8, int(self.cfg.rescue_bisect_steps))):
                    mid = 0.5 * (lo_t + hi_t)
                    xm = xa.copy(); xm[j] = a + mid * (b - a)
                    xm = np.clip(xm, self.train_min, self.train_max)
                    if self._predict_is_target0(xm):
                        best = xm; hi_t = mid
                    else:
                        lo_t = mid
                    if np.isclose(hi_t, lo_t, rtol=1e-10, atol=1e-12):
                        break
                return best

        # simple 2-D pairs
        for j1, j2 in combinations(ranked[:K], 2):
            base = x0.copy()
            v1s = self._neighbor_quantiles(x0, j1, 9)
            v2s = self._neighbor_quantiles(x0, j2, 9)
            v1s = [v for v in v1s if not np.isclose(v, float(base[j1]))]
            v2s = [v for v in v2s if not np.isclose(v, float(base[j2]))]
            for v1 in v1s[:4]:
                for v2 in v2s[:4]:
                    x = base.copy()
                    lo1, hi1 = float(self.train_min[j1]), float(self.train_max[j1])
                    lo2, hi2 = float(self.train_min[j2]), float(self.train_max[j2])
                    x[j1] = float(np.clip(v1, lo1, hi1))
                    x[j2] = float(np.clip(v2, lo2, hi2))
                    if self._predict_is_target0(x):
                        return x
        return None


# ----------------------------- I/O helpers -----------------------------

def _out_path(project: str, model_type: str, method: str, total_cfs: int, max_features: int) -> Path:
    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"DiCE_all_{total_cfs}_max{max_features}feat.csv"


# ----------------------------- driver -----------------------------

def _is_tree_like(base_model: Any) -> bool:
    """Detect common tree/boosting libraries to avoid external scaling."""
    if isinstance(base_model, (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)):
        return True
    mod = type(base_model).__module__.lower()
    name = type(base_model).__name__.lower()
    if "xgboost" in mod or "xgb" in mod:
        return True
    if "lightgbm" in mod or "lgbm" in mod:
        return True
    if "catboost" in mod or "catboost" in name:
        return True
    return False

def _external_scaler_for(base_model: Any, X_train: np.ndarray) -> Optional[StandardScaler]:
    """
    Use an external StandardScaler only if:
      - model is NOT a Pipeline, and
      - model is NOT tree/boosting-like (RF/ET/GBDT, XGBoost/LightGBM/CatBoost).
    Scaling tree inputs at inference when the model was trained on raw features
    reduces flip rate substantially; we skip it.
    """
    if isinstance(base_model, Pipeline):
        return None
    if _is_tree_like(base_model):
        return None
    return StandardScaler().fit(X_train)

def generate_cf_for_project(project: str,
                            model_type: str,
                            method: str,
                            total_cfs: int,
                            max_features: int,
                            verbose: bool = True,
                            overwrite: bool = True,
                            cfg_overrides: Dict[str, Any] = None):
    valid_methods = ["kfeature"]
    if method not in valid_methods:
        tqdm.write(f"[{project}/{model_type}/{method}] Unsupported method '{method}'. Use 'kfeature'.")
        return

    ds = read_dataset()
    if project not in ds:
        tqdm.write(f"[{project}/{model_type}] dataset not found. Skipping.")
        return
    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    base_model = get_model(project, model_type)
    scaler = _external_scaler_for(base_model, train[feat_cols].values)
    model = ModelWrapper(base_model, scaler=scaler)

    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}] no true positives. Skipping.")
        return

    lime_explainer = LimeTabularExplainer(
        training_data=train[feat_cols].values.astype(float),
        feature_names=feat_cols,
        class_names=["neg", "pos"],
        mode="classification",
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=SEED,
    )

    cfg = GenCfg(max_features=max_features, total_cfs=total_cfs)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    expl = MinChangeCF_LIME(model,
                            train[feat_cols].values,
                            feat_names=feat_cols,
                            lime_explainer=lime_explainer,
                            cfg=cfg)

    out_path = _out_path(project, model_type, method, total_cfs, max_features)
    if overwrite and out_path.exists():
        out_path.unlink(missing_ok=True)

    rows = []
    misses = 0
    for idx in tqdm(tp_df.index.astype(int),
                    desc=f"{project}/{model_type}/{method} (minimal)",
                    leave=False,
                    disable=not verbose):
        x0_row = test.loc[idx, feat_cols].astype(float)
        x0 = x0_row.values

        cfs = expl.explain(x0_row, x0)
        if not cfs:
            misses += 1
            d = expl.last_diag if isinstance(expl.last_diag, dict) else {}
            tqdm.write(
                f"[MISS] {project}/{model_type}/{method} idx={int(idx)} "
                f"reason={d.get('reason','unknown')} phase={d.get('phase','?')} "
                f"evals={d.get('evals','?')}"
            )
            continue

        cnt = 0
        for cf in cfs:
            proba = model.predict_proba(cf.reshape(1, -1))[0]
            changed = ~np.isclose(cf, x0, rtol=1e-7, atol=1e-7)
            num_changed = int(np.sum(changed))
            if num_changed == 0:
                continue
            rec = {
                "test_idx": int(idx),
                "candidate_id": int(cnt),
                **{c: float(v) for c, v in zip(feat_cols, cf.astype(float))},
                "proba0": float(proba[0]),
                "proba1": float(proba[1]),
                "num_features_changed": num_changed,
            }
            rows.append(rec)
            cnt += 1
            if cnt >= cfg.total_cfs:
                break

    if rows:
        out_df = pd.DataFrame(rows)
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
    ap = ArgumentParser(description="LIME-only CFs — minimal (enumeration + refine + fallback + rescue)")
    ap.add_argument("--project", type=str, default="all")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost")
    ap.add_argument("--methods", type=str, default="kfeature")
    ap.add_argument("--total_cfs", type=int, default=1)
    ap.add_argument("--max_features", type=int, default=5)

    # pool & budgets (defaults = your working recipe)
    ap.add_argument("--neighbor_k", type=int, default=500)
    ap.add_argument("--quantiles_n", type=int, default=41)
    ap.add_argument("--per_feature_cap", type=int, default=64)
    ap.add_argument("--feature_pool_factor", type=float, default=1.2)
    ap.add_argument("--include_extremes", action="store_true")  # opt-in; default False
    ap.add_argument("--instance_time_sec", type=float, default=240.0)
    ap.add_argument("--eval_cap", type=int, default=800_000)

    # enumeration throttles
    ap.add_argument("--max_product_per_combo", type=int, default=4000)
    ap.add_argument("--max_combos_per_k", type=int, default=3000)
    ap.add_argument("--top_per_feature_k_gt1", type=int, default=8)

    # fallback iterations
    ap.add_argument("--random_iters", type=int, default=800_000)
    ap.add_argument("--rand_subset_min", type=int, default=1)
    ap.add_argument("--rand_subset_max", type=int, default=0, help="0 → use max_features")

    # refine & rescue
    ap.add_argument("--refine_binary_steps", type=int, default=36)
    ap.add_argument("--refine_drop_passes", type=int, default=5)
    ap.add_argument("--rescue_k_features", type=int, default=3)
    ap.add_argument("--rescue_bisect_steps", type=int, default=28)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    projects = read_dataset()
    project_list = list(sorted(projects.keys())) if args.project == "all" else \
                   [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]
    model_types = [m.strip() for m in args.model_types.replace(",", " ").split() if m.strip()]
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]

    cfg_overrides = dict(
        neighbor_k=args.neighbor_k,
        quantiles_n=args.quantiles_n,
        per_feature_cap=args.per_feature_cap,
        include_extremes=bool(args.include_extremes),  # default False unless passed
        feature_pool_factor=args.feature_pool_factor,

        instance_time_sec=args.instance_time_sec,
        eval_cap=args.eval_cap,

        max_product_per_combo=args.max_product_per_combo,
        max_combos_per_k=args.max_combos_per_k,
        top_per_feature_k_gt1=args.top_per_feature_k_gt1,

        random_iters=args.random_iters,
        rand_subset_min=args.rand_subset_min,
        rand_subset_max=(None if int(args.rand_subset_max) == 0 else int(args.rand_subset_max)),

        refine_binary_steps=args.refine_binary_steps,
        refine_drop_passes=args.refine_drop_passes,

        rescue_k_features=args.rescue_k_features,
        rescue_bisect_steps=args.rescue_bisect_steps,
    )

    print(f"Running LIME-only CFs (minimal) for {len(project_list)} projects × {len(model_types)} models")
    print(f"Outputs: experiments/{{project}}/{{model}}/kfeature/DiCE_all_{args.total_cfs}_max{args.max_features}feat.csv\n")

    for p in tqdm(project_list, desc="Projects", disable=not args.verbose):
        for mt in model_types:
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

if __name__ == "__main__":
    main()
