#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, PROPOSED_CHANGES, SEED

np.random.seed(SEED)

# ---------- Utilities ----------

def _features_only(df_or_row, label="target"):
    if isinstance(df_or_row, pd.Series):
        return df_or_row[df_or_row.index != label]
    return df_or_row.loc[:, df_or_row.columns != label]

def _load_flips_long_or_wide(fpath: Path) -> pd.DataFrame | None:
    if not fpath.exists() or fpath.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(fpath)
        # normalize "wide" (index=TP) to have test_idx column
        if "test_idx" not in df.columns:
            # if there's an index column captured in CSV, try it
            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "test_idx"})
            # or if no explicit column, try to treat implicit index as test_idx
            if "test_idx" not in df.columns and df.index.name is not None:
                df = df.reset_index().rename(columns={df.index.name: "test_idx"})
            elif "test_idx" not in df.columns:
                # assume first col is the index if it looks integer-ish
                try:
                    maybe_idx = df.columns[0]
                    if pd.api.types.is_integer_dtype(df[maybe_idx]) or all(df[maybe_idx].astype(str).str.isnumeric()):
                        df = df.rename(columns={maybe_idx: "test_idx"})
                except Exception:
                    pass
        return df
    except Exception:
        return None

def _predict_proba1(model, X):
    """Return P(y=1) if available; fallback to decision function -> sigmoid; else label."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        elif p.ndim == 1:
            return p  # already prob(1)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.clip(s, -50, 50)
        return 1.0 / (1.0 + np.exp(-s))
    y = model.predict(X)  # 0/1 labels
    return y.astype(float)

def _nearest_negative_distances(trainX_scaled, trainy, x_scaled, k=10):
    """Distances in standardized space to nearest class-0 training points."""
    mask0 = (trainy == 0)
    if mask0.sum() == 0:
        return np.array([])
    nn = NearestNeighbors(n_neighbors=min(k, mask0.sum()), metric="euclidean")
    nn.fit(trainX_scaled[mask0])
    d, _ = nn.kneighbors(x_scaled.reshape(1, -1), return_distance=True)
    return d.ravel()

def _random_box_sampling(model, scaler, original_row, train_df, features_to_vary=None,
                         n_samples=20000, keep_fraction_fixed=True):
    """
    Sample uniformly in the train min-max box over selected features.
    Non-varied features are kept at original values.
    Returns: (min_prob1, frac_pred0, any_flip_bool)
    """
    feat_cols = list(train_df.columns)
    X0 = original_row.values.astype(float)
    X0 = X0.reshape(1, -1)

    mins = train_df.min().values.astype(float)
    maxs = train_df.max().values.astype(float)

    if features_to_vary is None or len(features_to_vary) == 0:
        vary_idx = np.arange(len(feat_cols))
    else:
        vary_idx = np.array([feat_cols.index(f) for f in features_to_vary if f in feat_cols])
        if vary_idx.size == 0:
            vary_idx = np.arange(len(feat_cols))

    batch = 2000
    probs1_min = 1.0
    total = 0
    flipped = 0

    for start in range(0, n_samples, batch):
        m = min(batch, n_samples - start)
        M = np.repeat(X0, m, axis=0)
        U = np.random.uniform(size=(m, vary_idx.size))
        M[:, vary_idx] = mins[vary_idx] + U * (maxs[vary_idx] - mins[vary_idx])

        P = _predict_proba1(model, scaler.transform(M))
        probs1_min = min(probs1_min, float(P.min()))
        flipped += int((P < 0.5).sum())
        total += m

    frac_pred0 = flipped / total if total else 0.0
    return probs1_min, frac_pred0, flipped > 0

def _estimate_local_gradient(model, scaler, x_row, eps=1e-2, max_feats=30):
    """
    Finite-difference gradient of prob(1) wrt features at x.
    This is a heuristic; for tree models it’s piecewise-flat and may be 0.
    """
    feats = x_row.index.tolist()
    x = x_row.values.astype(float)
    x_scaled = scaler.transform([x])[0]
    p0 = _predict_proba1(model, np.array([x_scaled]))[0]

    grads = np.zeros(x.shape[0], dtype=float)
    # budget sanity
    idxs = np.arange(len(x))
    if len(x) > max_feats:
        np.random.shuffle(idxs)
        idxs = idxs[:max_feats]

    for j in idxs:
        x_pert = x.copy()
        # scale-aware epsilon
        sd = scaler.scale_[j] if hasattr(scaler, "scale_") else 1.0
        step = eps * (sd if sd > 0 else 1.0)
        x_pert[j] += step
        p1 = _predict_proba1(model, scaler.transform([x_pert]))[0]
        grads[j] = (p1 - p0) / step

    return p0, grads, feats

# ---------- Diagnostics per instance ----------

def diagnose_one(project, model_type, tp_idx, features_to_vary=None, n_samples=20000):
    ds = read_dataset()
    train, test = ds[project]
    feat_cols = list(_features_only(train).columns)
    model = get_model(project, model_type)
    scaler = StandardScaler().fit(_features_only(train).values)

    # Original point (must be a TP according to your pipeline)
    x_row = _features_only(test.loc[int(tp_idx)]).astype(float)
    x_scaled = scaler.transform([x_row.values])[0]
    p1 = _predict_proba1(model, np.array([x_scaled]))[0]

    # Training set distances to nearest class-0
    trainX = _features_only(train).values.astype(float)
    trainy = train["target"].values.astype(int)
    trainX_scaled = scaler.transform(trainX)
    d0 = _nearest_negative_distances(trainX_scaled, trainy, x_scaled, k=10)

    # Random box sampling within train min-max over allowed features
    p1_min, frac0, any_flip = _random_box_sampling(
        model, scaler, x_row, _features_only(train), features_to_vary, n_samples=n_samples
    )

    # Local gradient estimate (heuristic)
    p1_here, grads, feat_names = _estimate_local_gradient(model, scaler, x_row, eps=1e-2)

    print(f"\n[{project}/{model_type}] diagnose TP {tp_idx}")
    print(f"- Original prob(1): {p1:.4f}")
    if d0.size:
        print(f"- Nearest class-0 distance (std space): min={d0.min():.3f}, median={np.median(d0):.3f}")
    else:
        print("- No class-0 points in training set (unexpected).")
    print(f"- Random box (train min–max) over {'ALL features' if not features_to_vary else features_to_vary}:")
    print(f"    min prob(1) seen = {p1_min:.4f}, fraction predicted 0 = {frac0:.4f}, any flip? {any_flip}")
    # Show top local “most decreasing” features (negative gradient reduces prob(1))
    top_k = 8
    order = np.argsort(grads)  # most negative first
    print(f"- Local gradient approx: top-{top_k} features that *decrease* prob(1) if increased:")
    for j in order[:top_k]:
        print(f"    {feat_names[j]} : grad={grads[j]:+.4e}")

    # Quick verdict
    if not any_flip:
        print("\nVerdict: No class-0 found in the sampled box → likely causes:")
        print("  • Allowed ranges too tight (train min–max still not enough near this point), or")
        print("  • Model’s decision region around this TP is very ‘stubborn’ (prob(1) ≫ 0.5).")
        print("Suggestions: widen ranges; allow more features; increase DiCE samples/iters; try genetic method.")
    else:
        print("\nVerdict: There exist class-0s in the box, so DiCE likely missed them due to search effort/params.")
        print("Suggestions: increase total_CFs / iterations / sample_size; different seeds; enable more features to vary.")

# ---------- Find failing TPs from your DiCE outputs ----------

def find_failed_tp_indices(project, model_type) -> list[int]:
    """Return TP indices that do NOT appear in experiments/<proj>/<model>/DiCE_all.csv (i.e., no CF saved)."""
    ds = read_dataset()
    train, test = ds[project]
    model = get_model(project, model_type)

    tp_df = get_true_positives(model, train, test)
    tp_idxs = list(tp_df.index.astype(int))

    exp_csv = Path(EXPERIMENTS) / project / model_type / "DiCE_all.csv"
    flips = _load_flips_long_or_wide(exp_csv)
    if flips is None or flips.empty:
        return tp_idxs  # nothing saved -> treat all TPs as failing

    if "test_idx" in flips.columns:
        flipped_tp = set(flips["test_idx"].astype(int).unique())
    else:
        # wide format (one row per flipped TP) -> index is TP (or first column after cleanup)
        # try to infer
        if "Unnamed: 0" in flips.columns:
            flipped_tp = set(flips["Unnamed: 0"].astype(int).unique())
        elif flips.index.name is not None:
            flipped_tp = set(flips.index.astype(int).unique())
        else:
            # last resort: assume there is a column named like the index
            flipped_tp = set()

    failed = [i for i in tp_idxs if i not in flipped_tp]
    return failed

# ---------- CLI ----------

def main():
    ap = ArgumentParser()
    ap.add_argument("--project", required=True, help="e.g., derby@0")
    ap.add_argument("--model_type", required=True, help="RandomForest, SVM, XGBoost, LightGBM, CatBoost")
    ap.add_argument("--idxs", type=str, default="", help="Space-separated TP indices to diagnose; if empty, auto-detect failures")
    ap.add_argument("--features_to_vary", type=str, default="", help="Comma-separated features to vary in the sampling box (default: all)")
    ap.add_argument("--samples", type=int, default=20000, help="Random samples for the box probe")
    args = ap.parse_args()

    if args.idxs.strip():
        indices = [int(x) for x in args.idxs.split()]
    else:
        indices = find_failed_tp_indices(args.project, args.model_type)
        if len(indices) == 0:
            print("No failing TPs detected (all TPs have a saved CF).")
            return

    if args.features_to_vary.strip():
        ftv = [f.strip() for f in args.features_to_vary.split(",") if f.strip()]
    else:
        ftv = None

    for idx in indices:
        diagnose_one(args.project, args.model_type, idx, features_to_vary=ftv, n_samples=args.samples)

if __name__ == "__main__":
    main()
