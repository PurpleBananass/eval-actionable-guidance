#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nearest-Neighbor Subset CFs (batched, expansion; ≤K by default + history-aware ranking + history-refine)

Goal
-----
Find the closest counterfactual(s) for each test TP by changing at most K features
(K = max_features), where edited values come from TRAIN neighbor points.

We make misses very rare by progressively expanding the search:
  - Increase neighbor_k up to all train points
  - Increase the LIME top-pool up to all features
  - Increase combinations per neighbor
  - (Fallback) Mixed-donor mode: each edited feature may take its value from a
    different neighbor (still from TRAIN), enabling a much larger search space.

History-aware improvements:
  - Feasibility prior blends historical movement magnitudes with LIME for pool.
  - Sign-consistency filter (optional).
  - History-aware ranking: distance + λ * (1 - max cosine to history) for the
    same edited subset.

Distance options (for ranking)
------------------------------
--distance:
  - unit_l2   : Euclidean in robust unit space (median/MAD on TRAIN). [default]
  - euclidean : Euclidean in z-scored space (TRAIN mean/std).
  - raw_l2    : Euclidean in original feature space.

Default behavior: **≤K edits** (change fewer features when possible).
Pass `--exact_k` to force **exactly K**.

Output
------
experiments/{project}/{model}/CF_all.csv
Columns: test_idx, candidate_id, feature columns, proba0, proba1, num_features_changed, dist_unit_l2
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from tqdm import tqdm

# ---- LIME (optional; falls back to uniform weights if missing) ----
try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAVE_LIME = True
except Exception:
    _HAVE_LIME = False

# ---- project helpers ----
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED


# ============================ utils ============================

def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.reshape(-1)


def _coerce_index(idx, d: int) -> np.ndarray:
    if isinstance(idx, tuple):
        if len(idx) == 1:
            arr = np.asarray(idx[0], dtype=int).ravel()
        else:
            arr = np.concatenate([np.asarray(i, dtype=int).ravel() for i in idx])
    else:
        arr = np.asarray(idx, dtype=int).ravel()
    arr = arr[(arr >= 0) & (arr < d)]
    return np.unique(arr)


def mad(arr: NDArray[np.float_], c: float = 1.4826) -> float:
    med = np.median(arr)
    return float(c * np.median(np.abs(arr - med)) + 1e-12)


class UnitTransformer:
    """Robust units: (x - median) / MAD computed on TRAIN."""
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
    """KNN over unit space (supports dynamic k in kneighbors)."""
    def __init__(self, X_units: NDArray[np.float_], k: int = 300):
        self.nn = NearestNeighbors(
            n_neighbors=min(k, max(1, len(X_units))),
            metric="euclidean"
        ).fit(X_units)
        self.Xu = X_units

    def neighbor_indices(self, x_units: NDArray[np.float_], k: Optional[int] = None) -> NDArray[np.int_]:
        if k is None:
            _, idx = self.nn.kneighbors(x_units.reshape(1, -1), return_distance=True)
        else:
            k = min(int(k), len(self.Xu))
            _, idx = self.nn.kneighbors(x_units.reshape(1, -1), n_neighbors=k, return_distance=True)
        return idx[0]


# ============================ config ============================

@dataclass
class GenCfg:
    # Edit budget (≤K by default; use --exact_k to force exactly K)
    max_features: int
    total_cfs: int
    exact_k: bool = False   # << default is "at most K"

    # LIME + feasibility prior
    lime_samples: int = 4000
    top_pool: int = 16
    feasibility_alpha: float = 0.5

    # neighbors & batching
    neighbor_k: int = 200
    max_combos_per_k: int = 4000
    batch_size: int = 4096

    # expansion strategy
    expand_rounds: int = 4
    mixed_donor_after_round: int = 2
    mixed_donor_reps_per_subset: int = 4
    mixed_donor_max_candidates: int = 50000

    # budgets
    instance_time_sec: float = 600.0
    eval_cap: int = 5_000_000

    # distances
    distance: str = "unit_l2"  # 'unit_l2' | 'euclidean' | 'raw_l2'

    # history-aware ranking
    history_weight: float = 0.15
    sign_consistency: bool = True
    sign_consistency_frac: float = 0.60

    # history refine
    refine_history: bool = True
    refine_history_steps: int = 12
    refine_history_topk: int = 10

    # seed
    seed: int = SEED


# ============================ LIME helpers ============================

def lime_weights_or_uniform(
    explainer: Optional["LimeTabularExplainer"],
    model: ModelWrapper,
    x0_row: pd.Series,
    num_features: int,
    num_samples: int,
) -> np.ndarray:
    if _HAVE_LIME and explainer is not None:
        try:
            exp = explainer.explain_instance(
                data_row=x0_row.values.astype(float),
                predict_fn=model.predict_proba,
                labels=[0],
                num_features=num_features,
                num_samples=num_samples,
            )
            w = np.zeros(num_features, dtype=float)
            for fid, weight in exp.as_map().get(0, []):
                if 0 <= fid < num_features:
                    w[int(fid)] = float(abs(weight))
            if not np.any(w > 0):
                w[:] = 1.0
            return w / (w.sum() + 1e-12)
        except Exception:
            pass
    w = np.ones(num_features, dtype=float)
    return w / w.sum()


def topk_features(weights: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, len(weights))))
    return np.argsort(-weights)[:k]


# ============================ history-aware helpers ============================

def _max_history_cosine_for_cf(hist_deltas_full: np.ndarray,
                               x0: np.ndarray, cf: np.ndarray,
                               topk: int = 10) -> float:
    changed = ~np.isclose(cf, x0, rtol=1e-7, atol=1e-12)
    if hist_deltas_full.size == 0 or not np.any(changed):
        return 0.0
    d_vec = (cf - x0)[changed]
    H = hist_deltas_full[:, changed]
    keep = np.any(np.abs(H) > 0, axis=1)
    H = H[keep]
    if H.size == 0:
        return 0.0
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    dn = d_vec / (np.linalg.norm(d_vec) + 1e-12)
    cos = Hn @ dn
    if cos.size == 0:
        return 0.0
    order = np.argsort(-cos)
    return float(np.max(cos[order[:max(1, topk)]]))


# ============================ history-aligned refine ============================

def _select_hist_dirs_for_changed(
    hist_deltas_full: np.ndarray,
    changed_mask: np.ndarray,
    d_vec: np.ndarray,
    topk: int,
) -> List[np.ndarray]:
    if hist_deltas_full.size == 0 or not np.any(changed_mask):
        return []

    H = hist_deltas_full[:, changed_mask]
    keep = np.any(np.abs(H) > 0, axis=1)
    H = H[keep]
    if H.size == 0:
        return []

    d = d_vec.astype(float)
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        return []

    H_norms = np.linalg.norm(H, axis=1)
    nz = H_norms > 0
    if not np.any(nz):
        return []
    Hn = H[nz] / (H_norms[nz][:, None] + 1e-12)
    dn = d / (d_norm + 1e-12)
    cos = Hn @ dn

    order = np.argsort(-cos)
    top = H[nz][order[:max(1, int(topk))]]

    aligned = []
    for h in top:
        if np.dot(h, d) < 0:
            h = -h
        aligned.append(h.astype(float))
    return aligned


def _refine_toward_history(
    x0: np.ndarray,
    cf: np.ndarray,
    model: "ModelWrapper",
    hist_deltas_full: np.ndarray,
    refine_steps: int = 12,
    topk: int = 1,
) -> np.ndarray:
    z = cf.astype(float).copy()
    changed_mask = ~np.isclose(z, x0, rtol=1e-7, atol=1e-12)
    if not np.any(changed_mask):
        return z

    d_vec = (z - x0)[changed_mask]
    cand_dirs = _select_hist_dirs_for_changed(hist_deltas_full, changed_mask, d_vec, topk)
    if not cand_dirs:
        return z

    best = z
    best_alpha = 0.0
    for h in cand_dirs:
        target = x0.copy()
        target[changed_mask] = x0[changed_mask] + h

        lo, hi = 0.0, 1.0
        last_good_alpha = 0.0
        for _ in range(refine_steps):
            mid = 0.5 * (lo + hi)
            cand = (1.0 - mid) * z + mid * target
            if model.predict_label(cand.reshape(1, -1))[0] == 0:
                last_good_alpha = mid
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 1e-6:
                break

        if last_good_alpha > best_alpha:
            best_alpha = last_good_alpha
            best = (1.0 - best_alpha) * z + best_alpha * target

    return best


# ============================ distance helpers ============================

def _distance_unit_l2(ut: UnitTransformer, x0: np.ndarray, X: np.ndarray) -> np.ndarray:
    x0u = ut.to_units(x0.reshape(1, -1))
    Xu = ut.to_units(X)
    diff = Xu - x0u
    return np.sqrt(np.sum(diff * diff, axis=1))

def _distance_euclidean_z(z_mean: np.ndarray, z_std: np.ndarray, x0: np.ndarray, X: np.ndarray) -> np.ndarray:
    zstd = np.where(z_std > 0, z_std, 1.0)
    x0z = (x0.reshape(1, -1) - z_mean) / zstd
    Xz  = (X - z_mean) / zstd
    diff = Xz - x0z
    return np.sqrt(np.sum(diff * diff, axis=1))

def _distance_raw_l2(x0: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = X - x0.reshape(1, -1)
    return np.sqrt(np.sum(diff * diff, axis=1))

def _distance_any(kind: str,
                  ut: UnitTransformer,
                  z_mean: Optional[np.ndarray],
                  z_std: Optional[np.ndarray],
                  x0: np.ndarray,
                  X: np.ndarray) -> np.ndarray:
    if kind == "unit_l2":
        return _distance_unit_l2(ut, x0, X)
    elif kind == "euclidean":
        assert z_mean is not None and z_std is not None, "z-stats not provided for euclidean distance"
        return _distance_euclidean_z(z_mean, z_std, x0, X)
    elif kind == "raw_l2":
        return _distance_raw_l2(x0, X)
    else:
        raise ValueError(f"Unknown distance kind: {kind}")


# ============================ subset size scheduling ============================

def _subset_sizes_for_round(K: int, exact_k: bool) -> List[int]:
    """If exact_k: [K]; else: [1,2,...,K] (favor smaller via per-size budget)."""
    return [K] if exact_k else list(range(1, K + 1))

def _alloc_budget_per_size(sizes: List[int], total_budget: int) -> Dict[int, int]:
    """
    Split 'total_budget' across 'sizes' with a bias toward smaller sizes.
    Weight w_s = 1/s.
    """
    if total_budget <= 0:
        return {s: 0 for s in sizes}
    weights = np.array([1.0 / s for s in sizes], dtype=float)
    weights = weights / weights.sum()
    counts = np.floor(weights * total_budget).astype(int)
    # ensure at least 1 per size if possible
    for i, s in enumerate(sizes):
        if total_budget >= len(sizes) and counts[i] == 0:
            counts[i] = 1
    # fix rounding to sum to total_budget
    diff = total_budget - int(counts.sum())
    i = 0
    while diff > 0:
        counts[i % len(counts)] += 1
        diff -= 1
        i += 1
    return {s: int(c) for s, c in zip(sizes, counts)}


# ============================ core search ============================

def _build_candidates_from_subset(
    x0: np.ndarray,
    xn: np.ndarray,
    subsets: List[Tuple[int, ...]],
) -> np.ndarray:
    x0_1d = _as_1d(x0)
    xn_1d = _as_1d(xn)
    d = x0_1d.size

    C = np.repeat(x0_1d.reshape(1, -1), len(subsets), axis=0)
    for r, sub in enumerate(subsets):
        idx = _coerce_index(sub, d)
        if idx.size:
            C[r, idx] = xn_1d[idx]
    return C


def _sample_combinations(pool: Sequence[int], s: int, max_count: int, rng: np.random.Generator) -> List[Tuple[int, ...]]:
    n = len(pool)
    total = math.comb(n, s)
    if total <= max_count:
        return list(combinations(pool, s))
    out: List[Tuple[int, ...]] = []
    seen: set = set()
    trials = 0
    need = max_count
    pool_arr = np.array(pool, dtype=int)
    while len(out) < need and trials < need * 40:
        trials += 1
        choice = tuple(sorted(rng.choice(pool_arr, size=s, replace=False).tolist()))
        if choice in seen:
            continue
        seen.add(choice)
        out.append(choice)
    return out


def _build_candidates_mixed_donors(
    x0: np.ndarray,
    donors: np.ndarray,
    subsets: List[Tuple[int, ...]],
    reps_per_subset: int,
    max_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    d = x0.shape[0]
    out = []
    cap = int(max_candidates)
    for sub in subsets:
        idx = _coerce_index(sub, d)
        if idx.size == 0:
            continue
        for _ in range(reps_per_subset):
            z = x0.copy()
            donor_idxs = rng.integers(low=0, high=len(donors), size=idx.size)
            z[idx] = donors[donor_idxs, idx]
            out.append(z)
            if len(out) >= cap:
                return np.asarray(out, dtype=float)
    return np.asarray(out, dtype=float) if out else np.empty((0, d), dtype=float)


def find_cfs_for_instance(
    model: ModelWrapper,
    ut: UnitTransformer,
    knn: NeighborIndex,
    x0_row: pd.Series,
    x0: np.ndarray,
    feat_cols: List[str],
    cfg: GenCfg,
    lime_explainer: Optional["LimeTabularExplainer"],
    rng: np.random.Generator,
    train_units: np.ndarray,
    hist_deltas_full: np.ndarray,
    z_mean: Optional[np.ndarray],
    z_std: Optional[np.ndarray],
    hist_sign: Optional[np.ndarray],
) -> pd.DataFrame:
    d = x0.shape[0]
    K = int(cfg.max_features)
    assert K >= 1, "max_features must be >= 1"

    # LIME for ordering
    weights = lime_weights_or_uniform(
        explainer=lime_explainer,
        model=model,
        x0_row=x0_row,
        num_features=d,
        num_samples=cfg.lime_samples,
    )

    # Blend with feasibility prior
    if hist_deltas_full.size:
        hist_abs = np.mean(np.abs(hist_deltas_full), axis=0)
        hist_std = np.std(hist_deltas_full, axis=0)
        feas = hist_abs + hist_std
        if np.any(feas > 0):
            w_hist = feas / (np.mean(feas) + 1e-12)
            weights = weights * (1.0 + float(cfg.feasibility_alpha) * w_hist)
            weights = weights / (weights.sum() + 1e-12)

    # base pool
    base_pool_k = max(K, int(cfg.top_pool))
    base_pool_idx = topk_features(weights, base_pool_k).tolist()

    # time / eval tracking
    start_t = time.time()
    evals = 0

    # expansion schedules
    n_train = len(train_units)
    neigh_sched = [
        min(cfg.neighbor_k, n_train),
        min(int(cfg.neighbor_k * 2.5), n_train),
        min(int(cfg.neighbor_k * 10), n_train),
        n_train,
    ]
    pool_sched = [
        min(base_pool_k, d),
        min(max(base_pool_k * 2, K), d),
        min(max(base_pool_k * 4, K), d),
        d,
    ]
    combos_sched = [
        cfg.max_combos_per_k,
        max(cfg.max_combos_per_k * 5, 20000),
        max(cfg.max_combos_per_k * 20, 80000),
        max(cfg.max_combos_per_k * 50, 200000),
    ]

    kept_rows: List[Dict[str, Any]] = []

    # Prepare x0 location in unit space once
    x0u = ut.to_units(x0.reshape(1, -1))[0]

    def _append_kept(C: np.ndarray):
        nonlocal kept_rows, evals
        if C.size == 0:
            return
        P = model.predict_proba(C)
        evals += C.shape[0]
        flips = (P[:, 0] >= 0.5)  # target becomes class 0
        if not np.any(flips):
            return
        C_flip = C[flips]
        P_flip = P[flips]

        # ≤K (or ==K if exact_k)
        changed_counts = np.sum(~np.isclose(C_flip, x0, rtol=1e-7, atol=1e-12), axis=1)
        if cfg.exact_k:
            valid = (changed_counts == K)
        else:
            valid = (changed_counts >= 1) & (changed_counts <= K)
        if not np.any(valid):
            return

        C_sel = C_flip[valid]
        P_sel = P_flip[valid]
        cc_sel = changed_counts[valid].astype(int)

        # distances
        dist = _distance_any(cfg.distance, ut, z_mean, z_std, x0, C_sel)

        # sign-consistency (optional)
        if cfg.sign_consistency and hist_sign is not None and hist_deltas_full.size:
            delta = C_sel - x0.reshape(1, -1)
            edited = ~np.isclose(delta, 0.0, rtol=1e-7, atol=1e-12)
            sgn = np.sign(delta)
            match = (sgn == hist_sign.reshape(1, -1)) | (hist_sign.reshape(1, -1) == 0)
            num_edited = np.sum(edited, axis=1)
            frac = np.where(num_edited > 0, np.sum(match & edited, axis=1) / (num_edited + 1e-12), 0.0)
            keep_mask = frac >= float(cfg.sign_consistency_frac)
            if not np.any(keep_mask):
                return
            C_sel = C_sel[keep_mask]; P_sel = P_sel[keep_mask]; dist = dist[keep_mask]; cc_sel = cc_sel[keep_mask]

        if C_sel.shape[0] == 0:
            return

        # history-aware re-ranking
        if hist_deltas_full.size and cfg.history_weight > 0.0:
            mis = np.empty(C_sel.shape[0], dtype=float)
            for r in range(C_sel.shape[0]):
                maxcos = _max_history_cosine_for_cf(
                    hist_deltas_full, x0, C_sel[r],
                    topk=max(1, int(cfg.refine_history_topk))
                )
                mis[r] = 1.0 - max(0.0, min(1.0, maxcos))
            # Primary: fewer edits; Secondary: distance; Tertiary: higher p0; Then: history penalty
            score = dist + float(cfg.history_weight) * mis
            order = np.lexsort((score, -P_sel[:, 0], dist, cc_sel))
        else:
            order = np.lexsort((-P_sel[:, 0], dist, cc_sel))  # fewer edits primary

        C_sel = C_sel[order]; P_sel = P_sel[order]; dist = dist[order]; cc_sel = cc_sel[order]

        for r in range(C_sel.shape[0]):
            cf = C_sel[r]
            proba = P_sel[r]
            d0 = float(dist[r])
            k_used = int(cc_sel[r])

            # refine (never worsen; keep ≤K rule)
            if cfg.refine_history and hist_deltas_full.size:
                cf2 = _refine_toward_history(
                    x0=x0,
                    cf=cf,
                    model=model,
                    hist_deltas_full=hist_deltas_full,
                    refine_steps=cfg.refine_history_steps,
                    topk=max(1, int(cfg.refine_history_topk)),
                )
                if model.predict_label(cf2.reshape(1, -1))[0] == 0:
                    edits = int(np.sum(~np.isclose(cf2, x0, rtol=1e-7, atol=1e-12)))
                    if (cfg.exact_k and edits == K) or ((not cfg.exact_k) and (1 <= edits <= K)):
                        d2 = float(_distance_any(cfg.distance, ut, z_mean, z_std, x0, cf2.reshape(1, -1))[0])
                        if d2 <= d0:
                            cf = cf2
                            proba = model.predict_proba(cf.reshape(1, -1))[0]
                            d0 = d2
                            k_used = edits

            rec = {
                **{c: float(v) for c, v in zip(feat_cols, cf.astype(float))},
                "proba0": float(proba[0]),
                "proba1": float(proba[1]),
                "num_features_changed": int(k_used),
                "dist_unit_l2": float(d0),
            }
            kept_rows.append(rec)
            if len(kept_rows) >= cfg.total_cfs:
                break

    # expansion rounds
    rounds = int(max(1, cfg.expand_rounds))
    for rd in range(rounds):
        if len(kept_rows) >= cfg.total_cfs:
            break
        if (time.time() - start_t) >= cfg.instance_time_sec or evals >= cfg.eval_cap:
            break

        neighbor_k_r = neigh_sched[min(rd, len(neigh_sched) - 1)]
        pool_k_r = pool_sched[min(rd, len(pool_sched) - 1)]
        combos_k_r = combos_sched[min(rd, len(combos_sched) - 1)]

        # neighbors
        n_idx = knn.neighbor_indices(x0u, k=neighbor_k_r)
        donors = ut.from_units(knn.Xu[n_idx])

        # feature pool
        pool_idx = topk_features(weights, max(K, pool_k_r)).tolist()

        # subset-size schedule and per-size budgets
        sizes = _subset_sizes_for_round(K, cfg.exact_k)
        per_size_budget = _alloc_budget_per_size(sizes, combos_k_r)

        # ----- Stage A: single-donor search (per neighbor) -----
        bs = max(1, int(cfg.batch_size))
        for s in sizes:
            if len(kept_rows) >= cfg.total_cfs:
                break
            subs = _sample_combinations(pool_idx, s, per_size_budget[s], rng)
            if not subs:
                continue
            for ni in range(donors.shape[0]):
                if len(kept_rows) >= cfg.total_cfs:
                    break
                if (time.time() - start_t) >= cfg.instance_time_sec or evals >= cfg.eval_cap:
                    break
                xn = donors[ni]
                for lo in range(0, len(subs), bs):
                    hi = min(lo + bs, len(subs))
                    C = _build_candidates_from_subset(x0, xn, subs[lo:hi])
                    _append_kept(C)
                    if len(kept_rows) >= cfg.total_cfs:
                        break
                    if (time.time() - start_t) >= cfg.instance_time_sec or evals >= cfg.eval_cap:
                        break

        if len(kept_rows) >= cfg.total_cfs:
            break
        if (time.time() - start_t) >= cfg.instance_time_sec or evals >= cfg.eval_cap:
            break

        # ----- Stage B: mixed-donor fallback -----
        if rd >= int(cfg.mixed_donor_after_round):
            for s in sizes:
                if len(kept_rows) >= cfg.total_cfs:
                    break
                subs = _sample_combinations(pool_idx, s, per_size_budget[s], rng)
                if not subs:
                    continue
                produced = 0
                chunk = max(1, min(20000, cfg.mixed_donor_max_candidates))
                while produced < cfg.mixed_donor_max_candidates:
                    if len(kept_rows) >= cfg.total_cfs:
                        break
                    if (time.time() - start_t) >= cfg.instance_time_sec or evals >= cfg.eval_cap:
                        break
                    reps = int(max(1, cfg.mixed_donor_reps_per_subset))
                    max_subsets_this = max(1, chunk // reps)
                    subs_batch = subs[:max_subsets_this]
                    subs = subs[max_subsets_this:] + subs_batch
                    Cmix = _build_candidates_mixed_donors(
                        x0=x0,
                        donors=donors,
                        subsets=subs_batch,
                        reps_per_subset=reps,
                        max_candidates=chunk,
                        rng=rng,
                    )
                    if Cmix.size == 0:
                        break
                    _append_kept(Cmix)
                    produced += Cmix.shape[0]

    # prepare DF
    if not kept_rows:
        return pd.DataFrame()

    df = pd.DataFrame(kept_rows)
    # de-dup identical solutions (same feature vector)
    uniq_cols = feat_cols
    df = df.drop_duplicates(subset=uniq_cols, keep="first").reset_index(drop=True)

    # final sort: fewer edits → distance → -proba0
    df = df.sort_values(by=["num_features_changed", "dist_unit_l2", "proba0"],
                        ascending=[True, True, False])
    return df.head(cfg.total_cfs)


# ============================ I/O helpers & driver ============================

def _is_tree_like(base_model: Any) -> bool:
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
    if isinstance(base_model, Pipeline):
        return None
    if _is_tree_like(base_model):
        return None
    return StandardScaler().fit(X_train)


def _out_path(project: str, model_type: str, method: str, total_cfs: int, max_features: int) -> Path:
    out_dir = Path(EXPERIMENTS) / project / model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "CF_all.csv"


def run_project(project: str,
                model_type: str,
                method: str,
                total_cfs: int,
                max_features: int,
                verbose: bool,
                overwrite: bool,
                cfg_overrides: Dict[str, Any]):

    ds = read_dataset()
    if project not in ds:
        tqdm.write(f"[{project}/{model_type}] dataset not found. Skipping.")
        return

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    base_model = get_model(project, model_type)
    scaler = _external_scaler_for(base_model, train[feat_cols].values.astype(float))
    model = ModelWrapper(base_model, scaler=scaler)

    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}] no true positives. Skipping.")
        return

    # unit transformer & KNN on TRAIN
    Xtr = train[feat_cols].values.astype(float)
    ut = UnitTransformer(Xtr)
    Xu = ut.to_units(Xtr)
    knn = NeighborIndex(Xu, k=max(64, cfg_overrides.get("neighbor_k", 200)))

    # z-stats for --distance euclidean
    z_mean = Xtr.mean(axis=0)
    z_std = Xtr.std(axis=0)

    # Historical delta pool (TRAIN↔TEST overlaps)
    common_idx = train.index.intersection(test.index)
    hist_deltas = (test.loc[common_idx, feat_cols].values.astype(float)
                   - train.loc[common_idx, feat_cols].values.astype(float))
    nz_mask = np.any(np.abs(hist_deltas) > 0, axis=1)
    hist_deltas = hist_deltas[nz_mask]
    hist_sign = np.sign(np.nanmean(hist_deltas, axis=0)) if hist_deltas.size else np.zeros(Xtr.shape[1], dtype=float)

    # LIME explainer (optional)
    lime_explainer = None
    if _HAVE_LIME:
        lime_explainer = LimeTabularExplainer(
            training_data=Xtr,
            feature_names=feat_cols,
            class_names=["neg", "pos"],
            mode="classification",
            discretize_continuous=False,
            sample_around_instance=True,
            random_state=SEED,
        )

    cfg = GenCfg(
        max_features=max_features,
        total_cfs=total_cfs,
        exact_k=cfg_overrides.get("exact_k", False),  # << default ≤K
        lime_samples=cfg_overrides.get("lime_samples", 4000),
        top_pool=cfg_overrides.get("top_pool", 16),
        feasibility_alpha=cfg_overrides.get("feasibility_alpha", 0.5),
        neighbor_k=cfg_overrides.get("neighbor_k", 200),
        max_combos_per_k=cfg_overrides.get("max_combos_per_k", 4000),
        batch_size=cfg_overrides.get("batch_size", 4096),
        expand_rounds=cfg_overrides.get("expand_rounds", 4),
        mixed_donor_after_round=cfg_overrides.get("mixed_donor_after_round", 2),
        mixed_donor_reps_per_subset=cfg_overrides.get("mixed_donor_reps_per_subset", 4),
        mixed_donor_max_candidates=cfg_overrides.get("mixed_donor_max_candidates", 50000),
        instance_time_sec=cfg_overrides.get("instance_time_sec", 600.0),
        eval_cap=cfg_overrides.get("eval_cap", 5_000_000),
        distance=cfg_overrides.get("distance", "unit_l2"),
        history_weight=cfg_overrides.get("history_weight", 0.15),
        sign_consistency=cfg_overrides.get("sign_consistency", True),
        sign_consistency_frac=cfg_overrides.get("sign_consistency_frac", 0.60),
        refine_history=cfg_overrides.get("refine_history", True),
        refine_history_steps=cfg_overrides.get("refine_history_steps", 12),
        refine_history_topk=cfg_overrides.get("refine_history_topk", 10),
        seed=cfg_overrides.get("seed", SEED),
    )

    out_path = _out_path(project, model_type, method, total_cfs, max_features)
    if overwrite and out_path.exists():
        out_path.unlink(missing_ok=True)

    rng = np.random.default_rng(cfg.seed)
    results = []
    misses = 0

    for idx in tqdm(tp_df.index.astype(int),
                    desc=f"{project}/{model_type}/{method} (≤K={cfg.max_features}{' | EXACT-K' if cfg.exact_k else ''}, dist={cfg.distance})",
                    leave=False, disable=not verbose):
        x0_row = test.loc[idx, feat_cols].astype(float)
        x0 = x0_row.values.astype(float)

        kept = find_cfs_for_instance(
            model=model,
            ut=ut,
            knn=knn,
            x0_row=x0_row,
            x0=x0,
            feat_cols=feat_cols,
            cfg=cfg,
            lime_explainer=lime_explainer,
            rng=rng,
            train_units=Xu,
            hist_deltas_full=hist_deltas,
            z_mean=z_mean,
            z_std=z_std,
            hist_sign=hist_sign,
        )

        if kept is None or kept.empty:
            misses += 1
            continue

        kept = kept.copy()
        kept.insert(0, "candidate_id", np.arange(len(kept)))
        kept.insert(0, "test_idx", int(idx))
        results.append(kept)

    if results:
        out_df = pd.concat(results, axis=0, ignore_index=True)
        cols = ["test_idx", "candidate_id"] + feat_cols + ["proba0", "proba1", "num_features_changed", "dist_unit_l2"]
        out_df = out_df[cols]
        out_df.to_csv(out_path, index=False)
        uniq = out_df["test_idx"].nunique()
        tqdm.write(
            f"[OK] {project}/{model_type}/{method}: wrote {len(out_df)} rows across "
            f"{uniq} TP(s) → {out_path} | misses={misses}"
        )
    else:
        tqdm.write(f"[{project}/{model_type}/{method}] no candidates found. misses={misses}")


# ============================ CLI ============================

def main():
    ap = ArgumentParser(description="Nearest-Neighbor Subset CFs (expansion, ≤K by default, history-aware ranking + refine)")
    ap.add_argument("--project", type=str, default="all")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost")
    ap.add_argument("--method", type=str, default="subset-nn")
    ap.add_argument("--total_cfs", type=int, default=1)
    ap.add_argument("--max_features", type=int, default=5)

    # Exact-K toggle (default OFF → ≤K)
    ap.add_argument("--exact_k", action="store_true", help="Enforce exactly K (=max_features) edited features")

    # LIME / neighbors / batching
    ap.add_argument("--lime_samples", type=int, default=12000)
    ap.add_argument("--top_pool", type=int, default=128)
    ap.add_argument("--feasibility_alpha", type=float, default=0.5)
    ap.add_argument("--neighbor_k", type=int, default=2500)
    ap.add_argument("--max_combos_per_k", type=int, default=120000)
    ap.add_argument("--batch_size", type=int, default=16384)

    # expansion
    ap.add_argument("--expand_rounds", type=int, default=6)
    ap.add_argument("--mixed_donor_after_round", type=int, default=1)
    ap.add_argument("--mixed_donor_reps_per_subset", type=int, default=12)
    ap.add_argument("--mixed_donor_max_candidates", type=int, default=300000)

    # budgets
    ap.add_argument("--instance_time_sec", type=float, default=600.0)
    ap.add_argument("--eval_cap", type=int, default=20000000)

    ap.add_argument("--distance", type=str, default="euclidean", choices=["unit_l2", "euclidean", "raw_l2"])

    # history-aware re-ranking
    ap.add_argument("--history_weight", type=float, default=0.15)
    ap.add_argument("--sign_consistency", dest="sign_consistency", action="store_true")
    ap.add_argument("--no_sign_consistency", dest="sign_consistency", action="store_false")
    ap.set_defaults(sign_consistency=True)
    ap.add_argument("--sign_consistency_frac", type=float, default=0.60)

    # history refine flags (default ON; --no_refine_history to disable)
    ap.add_argument("--refine_history", dest="refine_history", action="store_true")
    ap.add_argument("--no_refine_history", dest="refine_history", action="store_false")
    ap.set_defaults(refine_history=True)
    ap.add_argument("--refine_history_steps", type=int, default=24)
    ap.add_argument("--refine_history_topk", type=int, default=32)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    projects = read_dataset()
    project_list = list(sorted(projects.keys())) if args.project == "all" else \
                   [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]
    model_types = [m.strip() for m in args.model_types.replace(",", " ").split() if m.strip()]

    cfg_overrides = dict(
        exact_k=bool(args.exact_k),  # default False → ≤K
        lime_samples=args.lime_samples,
        top_pool=args.top_pool,
        feasibility_alpha=args.feasibility_alpha,
        neighbor_k=args.neighbor_k,
        max_combos_per_k=args.max_combos_per_k,
        batch_size=args.batch_size,
        expand_rounds=args.expand_rounds,
        mixed_donor_after_round=args.mixed_donor_after_round,
        mixed_donor_reps_per_subset=args.mixed_donor_reps_per_subset,
        mixed_donor_max_candidates=args.mixed_donor_max_candidates,
        instance_time_sec=args.instance_time_sec,
        eval_cap=args.eval_cap,
        distance=args.distance,
        history_weight=args.history_weight,
        sign_consistency=args.sign_consistency,
        sign_consistency_frac=args.sign_consistency_frac,
        refine_history=args.refine_history,
        refine_history_steps=args.refine_history_steps,
        refine_history_topk=args.refine_history_topk,
    )

    print(f"Running subset-NN CFs (≤K={args.max_features}{' | EXACT-K' if args.exact_k else ''}) "
          f"for {len(project_list)} projects × {len(model_types)} models")
    print(f"Distance metric: {args.distance}")
    print(f"Output: experiments/{{project}}/{{model}}/CF_all.csv\n")

    for p in tqdm(project_list, desc="Projects", disable=not args.verbose):
        for mt in model_types:
            run_project(
                project=p,
                model_type=mt,
                method=args.method,
                total_cfs=args.total_cfs,
                max_features=args.max_features,
                verbose=args.verbose,
                overwrite=args.overwrite,
                cfg_overrides=cfg_overrides,
            )


if __name__ == "__main__":
    main()
