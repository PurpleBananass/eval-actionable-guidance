#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIME-only CF generator (same strategy; faster + larger/smarter pools + failure diagnostics).

Key fix:
- When building candidates we now clip **only the edited features** to [train_min, train_max].
  The old global clamp (clipping the whole vector) could silently change unedited features.

Strategy (unchanged):
  1) Enumerate by Hamming weight K = 1..max_features (closest / highest-gain values first,
     early-stop on first flip at each K).
  2) Single stochastic fallback: sample subsets k in [rand_subset_min, rand_subset_max] (<= Kmax),
     features ∝ LIME importance, values biased to closest candidates with a small extreme trickle.

Outputs (DiCE-compatible):
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
from tqdm import tqdm

# LIME (exit early if missing)
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    print("ERROR: python package 'lime' is not installed. Install with: pip install lime")
    sys.exit(1)

# project helpers (must exist in your codebase)
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED


# ----------------------------- utils -----------------------------

def _renorm_subset_probs(full_probs: np.ndarray, subset_idx: np.ndarray) -> np.ndarray:
    p = np.asarray(full_probs, dtype=float)[subset_idx]
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(p) / max(1, len(p))
    return p / s

def round_key(x: NDArray[np.float_], n: int = 12) -> Tuple:
    return tuple(np.round(x.astype(float), n).tolist())

def mad(arr: NDArray[np.float_], c: float = 1.4826) -> float:
    med = np.median(arr)
    return float(c * np.median(np.abs(arr - med)) + 1e-12)

class UnitTransformer:
    """robust z-units: (x - median) / MAD over TRAIN."""
    def __init__(self, X_train: NDArray[np.float_]):
        self.median = np.median(X_train, axis=0)
        self.mad = np.array([mad(X_train[:, i]) for i in range(X_train.shape[1])])

    def to_units(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        return (X - self.median) / self.mad

    def from_units(self, U: NDArray[np.float_]) -> NDArray[np.float_]:
        return U * self.mad + self.median

class ModelWrapper:
    """Wrap a binary classifier; scale with StandardScaler fitted on TRAIN features."""
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

class NeighborIndex:
    def __init__(self, X_units: NDArray[np.float_], k: int = 300):
        self.nn = NearestNeighbors(n_neighbors=min(k, max(1, len(X_units))), metric="euclidean").fit(X_units)
        self.Xu = X_units

    def neighbors(self, x_units: NDArray[np.float_]) -> NDArray[np.float_]:
        _, idx = self.nn.kneighbors(x_units.reshape(1, -1), return_distance=True)
        return self.Xu[idx[0]]

class BatchEval:
    """Batch up candidate evaluations to amortize model overhead while preserving order."""
    def __init__(self, model: ModelWrapper, batch_size: int = 4096):
        self.model = model
        self.batch_size = int(batch_size)
        self.buf_x: List[np.ndarray] = []
        self.buf_tag: List[Any] = []
        self.out: List[Tuple[np.ndarray, Any]] = []

    def push(self, x: np.ndarray, tag: Any = None):
        self.buf_x.append(x.copy())
        self.buf_tag.append(tag)
        if len(self.buf_x) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buf_x:
            return
        X = np.stack(self.buf_x, axis=0)
        P = self.model.predict_proba(X)
        for tag, p in zip(self.buf_tag, P):
            self.out.append((p, tag))
        self.buf_x.clear()
        self.buf_tag.clear()

    def results(self) -> List[Tuple[np.ndarray, Any]]:
        self.flush()
        out = self.out
        self.out = []
        return out

def _alloc_caps(weights: np.ndarray,
                idx_order: List[int],
                total_cap: int,
                min_cap: int,
                max_cap: int) -> Dict[int, int]:
    """Distribute a total pool budget across features proportional to weights."""
    if total_cap is None or total_cap <= 0:
        return {j: max_cap for j in idx_order}
    w = np.array([weights[j] for j in idx_order], dtype=float)
    w = w / (w.sum() + 1e-12)
    raw = np.maximum(min_cap, np.round(w * total_cap)).astype(int)
    raw = np.minimum(raw, max_cap)
    while raw.sum() > total_cap:
        i = int(np.argmax(raw))
        if raw[i] > min_cap:
            raw[i] -= 1
        else:
            break
    while raw.sum() < total_cap:
        i = int(np.argmax(w))
        if raw[i] < max_cap:
            raw[i] += 1
        else:
            break
    return {j: int(c) for j, c in zip(idx_order, raw)}


# ----------------------------- config -----------------------------

@dataclass
class GenCfg:
    max_features: int
    total_cfs: int
    flip_threshold: float = 0.5

    # search pool size
    neighbor_k: int = 300
    quantiles_n: int = 41             # neighbor quantiles per feature
    global_quantiles_n: int = 0       # 0 disables; union with train-wide quantiles if >0
    per_feature_cap: int = 64
    include_extremes: bool = True

    # pooled cap allocated by LIME weights (optional—set None/0 to disable)
    pool_cap_total: Optional[int] = None
    min_cap: int = 4
    max_cap: int = 64

    # budgets
    instance_time_sec: float = 240.0
    eval_cap: int = 800_000

    # enumeration throttles
    max_product_per_combo: int = 6000
    max_combos_per_k: int = 4000
    top_per_feature_k_gt1: int = 8

    # broadened enumeration feature pool
    feature_pool_factor: float = 3.0
    feature_pool_abs_max: Optional[int] = None

    # single stochastic fallback
    random_iters: int = 200_000
    rand_subset_min: int = 1
    rand_subset_max: Optional[int] = None

    # batching
    batch_size: int = 4096
    fallback_batch_size: int = 4096

    # LIME
    lime_samples: int = 4000
    seed: int = SEED

    # value ranking
    rank_by_gain: bool = True


# ----------------------------- core (LIME-only) -----------------------------

class MinChangeCF_LIME:
    """
    1) Enumerate by Hamming weight (K=1..Kmax), closest / highest-gain values first (batched eval, early-stop).
    2) Single random sampling fallback (features ∝ LIME, values biased to close ones + tiny extremes; batched).
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
        self.nn = NeighborIndex(Xu, k=cfg.neighbor_k)
        self.rng = np.random.default_rng(cfg.seed)

        self.lime = lime_explainer
        self.last_diag: Dict[str, Any] = {}

    # ---- LIME importance for target label ----
    def _lime_importance(self, x0_row: pd.Series, d: int, target_label: int) -> np.ndarray:
        exp = self.lime.explain_instance(
            data_row=x0_row.values.astype(float),
            predict_fn=self.model.predict_proba,
            labels=[target_label],
            num_features=d,
            num_samples=self.cfg.lime_samples,
        )
        weights = np.zeros(d, dtype=float)
        m = exp.as_map().get(target_label, [])
        for fid, w in m:
            if 0 <= fid < d:
                weights[int(fid)] = float(abs(w))
        if np.all(weights <= 0):
            weights = np.ones(d, dtype=float)
        return weights / (weights.sum() + 1e-12)

    # ---- neighbor + optional global quantiles & ordering ----
    def _neighbor_quantiles(self, x_orig: np.ndarray, j: int, n: int) -> np.ndarray:
        x_u = self.ut.to_units(x_orig.reshape(1, -1))[0]
        Xn_u = self.nn.neighbors(x_u)
        Xn = self.ut.from_units(Xn_u) if len(Xn_u) > 0 else self.Xtr
        lo, hi = float(self.train_min[j]), float(self.train_max[j])
        col = np.clip(Xn[:, j].astype(float), lo, hi)

        qs = np.linspace(0.0, 1.0, max(3, int(n)))
        vals = np.unique(np.quantile(col, qs, method="linear"))
        vals = np.clip(vals, lo, hi)

        if self.cfg.global_quantiles_n and self.cfg.global_quantiles_n > 0:
            qs_g = np.linspace(0.0, 1.0, int(self.cfg.global_quantiles_n))
            glob = np.unique(np.quantile(self.Xtr[:, j].astype(float), qs_g, method="linear"))
            vals = np.unique(np.clip(np.concatenate([vals, glob]), lo, hi))

        if self.cfg.include_extremes:
            vals = np.unique(np.concatenate([vals, [lo, hi]]))
        return vals

    def _sorted_by_unit_closeness(self, j: int, x0_val: float, candidates: Iterable[float]) -> List[float]:
        med, madj = float(self.ut.median[j]), float(self.ut.mad[j])
        x0u = (x0_val - med) / madj
        uniq = sorted(set(float(c) for c in candidates))
        return sorted(uniq, key=lambda v: abs(((v - med) / madj) - x0u))

    def _build_value_pools(self, x0: np.ndarray, use_idx: List[int], weights: np.ndarray) -> Dict[int, List[float]]:
        if self.cfg.pool_cap_total:
            cap_by_feat = _alloc_caps(weights, use_idx,
                                      total_cap=int(self.cfg.pool_cap_total),
                                      min_cap=int(self.cfg.min_cap),
                                      max_cap=int(self.cfg.max_cap))
        else:
            cap_by_feat = {j: int(self.cfg.per_feature_cap) for j in use_idx}

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
                if k in seen: continue
                seen.add(k); dedup.append(v)
            cap = max(1, int(cap_by_feat.get(j, self.cfg.per_feature_cap)))
            # ensure extremes remain if requested
            if self.cfg.include_extremes and len(dedup) > 0:
                lo_v, hi_v = lo, hi
                keep = dedup[: cap]
                if lo_v in dedup and lo_v not in keep:
                    keep = ([lo_v] + keep)[:cap]
                if hi_v in dedup and hi_v not in keep:
                    if len(keep) >= cap:
                        keep[-1] = hi_v
                    else:
                        keep.append(hi_v)
                pools[j] = keep
            else:
                pools[j] = dedup[: cap]
        return pools

    def _rank_pools_by_gain(self,
                            x0: np.ndarray,
                            pools: Dict[int, List[float]],
                            target_class: int) -> Dict[int, List[float]]:
        if not self.cfg.rank_by_gain or not pools:
            return pools
        be = BatchEval(self.model, batch_size=self.cfg.batch_size)
        p0_t = float(self.model.predict_proba(x0.reshape(1, -1))[0, target_class])

        for j, vals in pools.items():
            for v in vals:
                x = x0.copy()
                # per-feature clip on edit
                lo, hi = float(self.train_min[j]), float(self.train_max[j])
                x[j] = float(np.clip(float(v), lo, hi))
                be.push(x, tag=(j, v))

        gains: Dict[int, List[Tuple[float, float]]] = {}
        for (p, tag) in be.results():
            j, v = tag
            gain = float(p[target_class]) - p0_t
            gains.setdefault(j, []).append((gain, float(v)))

        for j, items in gains.items():
            base = float(x0[j])
            items.sort(key=lambda g: (-g[0], abs(g[1] - base)))
            pools[j] = [v for (g, v) in items]
        return pools

    # ---- main explain ----
    def explain(self, x0_row: pd.Series, x0: NDArray[np.float_], y_pred: int) -> List[NDArray[np.float_]]:
        self.last_diag = {}
        target_class = 1 - int(y_pred)
        start_time = time.time()
        d = x0.shape[0]

        # LIME importance → order & weights
        weights = self._lime_importance(x0_row, d, target_class)

        # BROADENED FEATURE POOL for enumeration
        ranked = np.argsort(-weights)
        pool_size = int(max(1, round(self.cfg.feature_pool_factor * self.cfg.max_features)))
        if self.cfg.feature_pool_abs_max is not None:
            pool_size = min(pool_size, int(self.cfg.feature_pool_abs_max))
        pool_size = min(pool_size, len(ranked))
        order = list(ranked[:pool_size])

        pools = self._build_value_pools(x0, order, weights)
        pools = self._rank_pools_by_gain(x0, pools, target_class)

        # enumeration K=1..Kmax (batched eval)
        flips: List[NDArray[np.float_]] = []
        seen = set()
        evals = 0
        total_combos = 0
        reason = "exhausted"

        for k in range(1, self.cfg.max_features + 1):
            if time.time() - start_time >= self.cfg.instance_time_sec:
                reason = "timeout-enum"; break
            if evals >= self.cfg.eval_cap:
                reason = "evalcap-enum"; break
            if len(flips) >= self.cfg.total_cfs:
                reason = "got-enough"; break

            combos_tried = 0
            for idx_tuple in combinations(order, k):
                if time.time() - start_time >= self.cfg.instance_time_sec:
                    reason = "timeout-enum"; break
                if evals >= self.cfg.eval_cap:
                    reason = "evalcap-enum"; break
                if len(flips) >= self.cfg.total_cfs:
                    reason = "got-enough"; break
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
                    dedup, sv = [], set()
                    for v in vals:
                        key = round(v, 12)
                        if key in sv: continue
                        sv.add(key); dedup.append(v)
                    value_lists.append(dedup)

                # throttle product size
                def prod_len(vls):
                    p = 1
                    for vv in vls: p *= max(1, len(vv))
                    return p
                while prod_len(value_lists) > self.cfg.max_product_per_combo:
                    i = int(np.argmax([len(v) for v in value_lists]))
                    if len(value_lists[i]) > 1:
                        value_lists[i] = value_lists[i][:-1]
                    else:
                        break

                be = BatchEval(self.model, batch_size=self.cfg.batch_size)
                cand_vecs: List[np.ndarray] = []

                tried = 0
                for tup in product(*value_lists):
                    if time.time() - start_time >= self.cfg.instance_time_sec:
                        reason = "timeout-enum"; break
                    if evals >= self.cfg.eval_cap:
                        reason = "evalcap-enum"; break
                    if len(flips) >= self.cfg.total_cfs:
                        reason = "got-enough"; break
                    tried += 1
                    if tried > self.cfg.max_product_per_combo:
                        break

                    changed = any(not np.isclose(tup[i], float(x0[idx_tuple[i]]))
                                  for i in range(len(idx_tuple)))
                    if not changed:
                        continue

                    x = x0.copy()
                    for i, j in enumerate(idx_tuple):
                        v = float(tup[i])
                        lo, hi = float(self.train_min[j]), float(self.train_max[j])
                        # per-feature clip ONLY here
                        x[j] = float(np.clip(v, lo, hi))

                    be.push(x, tag=None)
                    cand_vecs.append(x)
                    evals += 1

                    if len(be.buf_x) >= self.cfg.batch_size:
                        for (p, _), xvec in zip(be.results(), cand_vecs):
                            if p[1] < self.cfg.flip_threshold:
                                key = round_key(xvec, 12)
                                if key not in seen:
                                    seen.add(key)
                                    flips.append(xvec.astype(float))
                                    break
                        cand_vecs = []
                        if flips: break

                total_combos += 1

                if not flips and cand_vecs:
                    for (p, _), xvec in zip(be.results(), cand_vecs):
                        if p[1] < self.cfg.flip_threshold:
                            key = round_key(xvec, 12)
                            if key not in seen:
                                seen.add(key)
                                flips.append(xvec.astype(float))
                                break

                combos_tried += 1
                if flips:
                    break

            if flips:
                break

        if flips:
            self.last_diag = dict(phase="enum", evals=evals,
                                  elapsed=time.time() - start_time,
                                  reason="hit", total_combos=total_combos)
            return flips[: self.cfg.total_cfs]

        # SINGLE stochastic fallback (batched)
        probs_full = weights / (weights.sum() + 1e-12)
        kmin = max(1, int(self.cfg.rand_subset_min))
        kmax = int(self.cfg.rand_subset_max or self.cfg.max_features)
        kmax = max(kmin, min(kmax, self.cfg.max_features))

        features_all = np.arange(x0.shape[0], dtype=int)
        if features_all.size == 0:
            self.last_diag = dict(phase="fallback", evals=evals,
                                  elapsed=time.time() - start_time, reason="no-features")
            return []

        p_sub = _renorm_subset_probs(probs_full, features_all)

        be = BatchEval(self.model, batch_size=self.cfg.fallback_batch_size)
        cand_vecs: List[np.ndarray] = []
        it = 0
        reason = "exhausted-fallback"
        while it < self.cfg.random_iters:
            if time.time() - start_time >= self.cfg.instance_time_sec:
                reason = "timeout-fallback"; break
            if evals >= self.cfg.eval_cap:
                reason = "evalcap-fallback"; break

            batch_fill = min(self.cfg.fallback_batch_size, self.cfg.random_iters - it)
            for _ in range(batch_fill):
                it += 1
                k = int(self.rng.integers(low=kmin, high=kmax + 1))
                k = min(k, features_all.size)
                if k <= 0:
                    continue
                idx = self.rng.choice(features_all, size=k, replace=False, p=p_sub)

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
                            valid = False; break
                        pools[int(j)] = cand_vals

                    r = self.rng.random()
                    head = min(5, len(cand_vals))
                    if r < 0.80:
                        pick_idx = int(self.rng.integers(low=0, high=head))
                    elif r < 0.95:
                        pick_idx = int(self.rng.integers(low=0, high=len(cand_vals)))
                    else:
                        pick_idx = 0 if abs(cand_vals[0] - x0[j]) > abs(cand_vals[-1] - x0[j]) else (len(cand_vals) - 1)

                    v = float(cand_vals[pick_idx])
                    lo, hi = float(self.train_min[int(j)]), float(self.train_max[int(j)])
                    # per-feature clip ONLY here
                    x[int(j)] = float(np.clip(v, lo, hi))

                if not valid:
                    continue

                be.push(x, tag=None)
                cand_vecs.append(x)
                evals += 1

            for (p, _), xvec in zip(be.results(), cand_vecs):
                if p[1] < self.cfg.flip_threshold:
                    self.last_diag = dict(phase="fallback", evals=evals, iters=it,
                                          elapsed=time.time() - start_time, reason="hit")
                    return [xvec.astype(float)]
            cand_vecs = []

        self.last_diag = dict(phase="fallback", evals=evals, iters=it,
                              elapsed=time.time() - start_time, reason=reason)
        return []


# ----------------------------- I/O helpers -----------------------------

def _out_path(project: str, model_type: str, method: str, total_cfs: int, max_features: int) -> Path:
    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"DiCE_all_{total_cfs}_max{max_features}feat.csv"


# ----------------------------- driver -----------------------------

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
    scaler = StandardScaler().fit(train[feat_cols].values)
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

    expl = MinChangeCF_LIME(model, train[feat_cols].values, feat_cols, lime_explainer, cfg)

    out_path = _out_path(project, model_type, method, total_cfs, max_features)
    if overwrite and out_path.exists():
        out_path.unlink(missing_ok=True)

    rows = []
    misses = 0
    for idx in tqdm(tp_df.index.astype(int),
                    desc=f"{project}/{model_type}/{method} (LIME same-strategy)",
                    leave=False,
                    disable=not verbose):
        x0_row = test.loc[idx, feat_cols].astype(float)
        x0 = x0_row.values
        p = model.predict_proba(x0.reshape(1, -1))[0]
        y_pred = int(p[1] >= 0.5)  # TP sanity

        cfs = expl.explain(x0_row, x0, y_pred)
        if not cfs:
            misses += 1
            d = expl.last_diag if isinstance(expl.last_diag, dict) else {}
            tqdm.write(
                f"[MISS] {project}/{model_type}/{method} idx={int(idx)} "
                f"reason={d.get('reason','unknown')} phase={d.get('phase','?')} "
                f"evals={d.get('evals','?')} iters={d.get('iters','-')} "
                f"elapsed={d.get('elapsed','?'):.2f}s"
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
                "proba0": float(proba[0]),
                "proba1": float(proba[1]),
                "num_features_changed": num_changed,
            }
            for c, v in zip(feat_cols, cf.astype(float)):
                rec[c] = float(v)
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
    ap = ArgumentParser(description="LIME-only CFs (same strategy; faster + larger/smarter pools/batches + diagnostics)")
    ap.add_argument("--project", type=str, default="all")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost")
    ap.add_argument("--methods", type=str, default="kfeature")
    ap.add_argument("--total_cfs", type=int, default=1)
    ap.add_argument("--max_features", type=int, default=5)

    # pool size & budgets
    ap.add_argument("--neighbor_k", type=int, default=800)
    ap.add_argument("--quantiles_n", type=int, default=81)
    ap.add_argument("--global_quantiles_n", type=int, default=0)
    ap.add_argument("--per_feature_cap", type=int, default=128)
    ap.add_argument("--include_extremes", action="store_true")

    # pooled cap allocation
    ap.add_argument("--pool_cap_total", type=int, default=0, help="0 disables allocation by weights")
    ap.add_argument("--min_cap", type=int, default=4)
    ap.add_argument("--max_cap", type=int, default=64)

    ap.add_argument("--instance_time_sec", type=float, default=240.0)
    ap.add_argument("--eval_cap", type=int, default=2000000)

    # enumeration throttles
    ap.add_argument("--max_product_per_combo", type=int, default=4000)
    ap.add_argument("--max_combos_per_k", type=int, default=3000)
    ap.add_argument("--top_per_feature_k_gt1", type=int, default=8)

    # broadened enumeration feature pool
    ap.add_argument("--feature_pool_factor", type=float, default=1.6)
    ap.add_argument("--feature_pool_abs_max", type=int, default=0, help="0 → no hard cap")

    # fallback iterations
    ap.add_argument("--random_iters", type=int, default=800000)
    ap.add_argument("--rand_subset_min", type=int, default=5)
    ap.add_argument("--rand_subset_max", type=int, default=0, help="0 → use max_features")

    # batching
    ap.add_argument("--batch_size", type=int, default=16384)
    ap.add_argument("--fallback_batch_size", type=int, default=16384)

    # LIME
    ap.add_argument("--lime_samples", type=int, default=8000)

    # ranking toggle
    ap.add_argument("--no_rank_by_gain", action="store_true", help="Disable reordering pools by target-prob gain")

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
        global_quantiles_n=args.global_quantiles_n,
        per_feature_cap=args.per_feature_cap,
        include_extremes=bool(args.include_extremes),
        pool_cap_total=(None if int(args.pool_cap_total) == 0 else int(args.pool_cap_total)),
        min_cap=args.min_cap,
        max_cap=args.max_cap,
        instance_time_sec=args.instance_time_sec,
        eval_cap=args.eval_cap,
        max_product_per_combo=args.max_product_per_combo,
        max_combos_per_k=args.max_combos_per_k,
        top_per_feature_k_gt1=args.top_per_feature_k_gt1,
        feature_pool_factor=args.feature_pool_factor,
        feature_pool_abs_max=(None if int(args.feature_pool_abs_max) == 0 else int(args.feature_pool_abs_max)),
        random_iters=args.random_iters,
        rand_subset_min=args.rand_subset_min,
        rand_subset_max=(None if int(args.rand_subset_max) == 0 else int(args.rand_subset_max)),
        lime_samples=args.lime_samples,
        batch_size=args.batch_size,
        fallback_batch_size=args.fallback_batch_size,
        rank_by_gain=(not args.no_rank_by_gain),
    )

    print(f"Running LIME-only (same strategy) for {len(project_list)} projects × {len(model_types)} models")
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
