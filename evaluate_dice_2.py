#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DiCE/kfeatures evaluation with caching of 'best' candidates.

What this script does:
- RQ1: Strictly verifies flips by re-predicting with the model.
- RQ3: Compares counterfactual deltas to historical deltas (Mahalanobis or cosine),
       selecting the 'best' candidate per test_idx (lowest distance). Results are cached.
- Implications: Computes total scaled change using either computed best/first or cached selections.

Key fixes vs previous version:
- Properly detect changed features by comparing the candidate to the ORIGINAL row (never to itself),
  whether the flip CSV has a full feature vector or only a subset.
- Selected cache stores the full feature vector + min/max/mean stats and is reusable across runs.

Selected cache file (per model/method/params):
  ./evaluations/feasibility/mahalanobis/selected/
    {ABBR}_DiCE_{method}_{distance}_{selection}_cf{total}_max{K}.csv
"""

import math
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from tabulate import tabulate

from hyparams import PROPOSED_CHANGES, EXPERIMENTS  # PROPOSED_CHANGES unused; kept for parity
from data_utils import read_dataset, get_model, get_true_positives


# ----------------------------- config / helpers -----------------------------

MODEL_ABBR = {
    "SVM": "SVM",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

DEFAULT_GROUPS = [
    ["activemq@0", "activemq@1", "activemq@2", "activemq@3"],
    ["camel@0", "camel@1", "camel@2"],
    ["derby@0", "derby@1"],
    ["groovy@0", "groovy@1"],
    ["hbase@0", "hbase@1"],
    ["hive@0", "hive@1"],
    ["jruby@0", "jruby@1", "jruby@2"],
    ["lucene@0", "lucene@1", "lucene@2"],
    ["wicket@0", "wicket@1"],
]

# Where selected choices are stored
SELECTED_DIR = "./evaluations/feasibility/mahalanobis/selected"

# --- add these helpers near the top (under SELECTED_DIR) ---

def _is_pipeline(model) -> bool:
    return hasattr(model, "steps") or hasattr(model, "named_steps")

def _needs_scaling(model, model_type: str) -> bool:
    """
    Scale only if training likely used scaling and the model isn't a Pipeline already.
    """
    if _is_pipeline(model):
        return False
    mt = (model_type or "").lower()
    return mt in {"svm", "svc", "linearsvc", "logisticregression", "knn", "kneighborsclassifier"}

def _prepare_X_for_predict(model, model_type: str, train_X: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Mirror training-time preprocessing:
      - If Pipeline → pass through raw X (pipeline applies transforms).
      - Else if model_type needs scaling → fit StandardScaler on train_X and transform X.
      - Else → return raw X.
    """
    if _needs_scaling(model, model_type):
        scaler = StandardScaler().fit(train_X)
        return scaler.transform(X)
    return X

def _negative_label(model) -> int:
    classes = list(getattr(model, "classes_", []))
    if classes:
        try:
            return sorted(classes)[0]
        except Exception:
            pass
    return 0

def _negative_label(model) -> int:
    """
    Pick the model's 'negative' label robustly.
    - If classes_ is {0,1} → 0
    - Else (e.g., {-1, 1}) → min(classes_)
    - Else fallback → 0
    """
    classes = list(getattr(model, "classes_", []))
    if classes:
        if 0 in classes and 1 in classes:
            return 0
        try:
            return sorted(classes)[0]
        except Exception:
            return 0
    return 0


def _selected_index_file_param(
    model_type: str,
    method: str,
    total_cfs: int,
    max_features: int,
    distance: str,
    selection_strategy: str,
) -> Path:
    abbr = MODEL_ABBR.get(model_type, model_type)
    fname = f"{abbr}_DiCE_{method}_{distance}_{selection_strategy}_cf{total_cfs}_max{max_features}.csv"
    return Path(SELECTED_DIR) / fname


def _dice_flip_path(
    project: str,
    model_type: str,
    method: str,
    total_cfs: int,
    max_features: int,
) -> Path:
    """
    New generator writes: DiCE_all_{total_cfs}_max{max_features}feat.csv
    under experiments/{project}/{model_type}/{method}/
    """
    filename = f"DiCE_all_{total_cfs}_max{max_features}feat.csv"
    return Path(EXPERIMENTS) / project / model_type / method / filename


def _feature_cols(df: pd.DataFrame) -> list[str]:
    non_feats = {
        "test_idx",
        "candidate_id",
        "proba0",
        "proba1",
        "num_features_changed",
        "target",
        # sometimes appear in caches:
        "project", "min", "max", "mean",
    }
    return [c for c in df.columns if c not in non_feats]


def _features_only(df_or_row, label="target"):
    if isinstance(df_or_row, pd.Series):
        return df_or_row[df_or_row.index != label]
    return df_or_row.loc[:, df_or_row.columns != label]


def _load_flips_long(flip_path: Path, feature_cols: list[str]) -> pd.DataFrame | None:
    """
    Return a long-format DataFrame with potentially multiple rows per test_idx.
    Keeps only feature columns + 'test_idx' (+ 'candidate_id' if present).
    """
    if not flip_path.exists() or flip_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(flip_path)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    # Ensure test_idx exists or reconstruct from index
    if "test_idx" not in df.columns:
        try:
            df = pd.read_csv(flip_path, index_col=0).reset_index().rename(columns={"index": "test_idx"})
        except Exception:
            return None

    keep = ["test_idx"] + ([c for c in ("candidate_id",) if c in df.columns]) + [c for c in feature_cols if c in df.columns]
    df = df.loc[:, [c for c in keep if c in df.columns]].copy()

    df["test_idx"] = pd.to_numeric(df["test_idx"], errors="coerce")
    df = df.dropna(subset=["test_idx"]).copy()
    df["test_idx"] = df["test_idx"].astype(int)

    if "candidate_id" in df.columns:
        df = df.sort_values(["test_idx", "candidate_id"], kind="stable")
    else:
        df = df.sort_values(["test_idx"], kind="stable")

    return df


def generate_all_combinations(data):
    feature_values = [data[feature] for feature in data]
    combinations = list(product(*feature_values))
    return pd.DataFrame(combinations, columns=data.keys())


def normalized_mahalanobis_distance(df, x, y):
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0

    standardized_df = (df - df.mean()) / df.std()

    x_standardized = [
        (x[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns
    ]
    y_standardized = [
        (y[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns
    ]

    cov_matrix = np.cov(standardized_df.T)
    if cov_matrix.ndim == 0:
        inv_cov_matrix = np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)

    min_vector = np.array([min(df[feature]) for feature in df.columns])
    max_vector = np.array([max(df[feature]) for feature in df.columns])

    min_vector_standardized = [
        (min_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]
    max_vector_standardized = [
        (max_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]

    max_vector_distance = mahalanobis(min_vector_standardized, max_vector_standardized, inv_cov_matrix)
    normalized_distance = distance / max_vector_distance if max_vector_distance != 0 else 0
    return normalized_distance


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        print(vec1, vec2)  # keep baseline behavior
        return 0
    return dot_product / (norm_vec1 * norm_vec2)


def cosine_all(df, x):
    return [cosine_similarity(x, row) for _, row in df.iterrows()]


def mahalanobis_all(df, x):
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0

    standardized_df = (df - df.mean()) / df.std()
    x_standardized = [
        (x[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns
    ]

    cov_matrix = np.cov(standardized_df.T)
    if cov_matrix.ndim == 0:
        inv_cov_matrix = np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    min_vector = np.array([min(df[feature]) for feature in df.columns])
    max_vector = np.array([max(df[feature]) for feature in df.columns])

    min_vector_standardized = [
        (min_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]
    max_vector_standardized = [
        (max_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]

    max_vector_distance = mahalanobis(min_vector_standardized, max_vector_standardized, inv_cov_matrix)

    distances = []
    for _, y in df.iterrows():
        y_standardized = [
            (y[feature] - df[feature].mean()) / df[feature].std()
            for feature in df.columns
        ]
        distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)
        distances.append(distance / max_vector_distance if max_vector_distance != 0 else 0)
    return distances


# ----------------------------- RQ1: Flip rates (strict verify) -----------------------------

def _count_flips_for_project_model(
    project: str,
    model_type: str,
    method: str,
    total_cfs: int,
    max_features: int,
) -> tuple[int, int, int, int]:
    ds = read_dataset()
    if project not in ds:
        return 0, 0, 0, 0

    train, test = ds[project]
    feat_cols = list(_features_only(test).columns)

    model = get_model(project, model_type)

    # Build TP index set (y_true=1 & y_pred=1)
    # IMPORTANT: predict on the same representation the model expects.
    X_train = train[feat_cols].values
    X_test  = test[feat_cols].values
    X_test_for_pred = _prepare_X_for_predict(model, model_type, X_train, X_test)
    y_test_pred = model.predict(X_test_for_pred)
    tp_idx_set = set(test.index[(test["target"].values == 1) & (y_test_pred == 1)].astype(int))
    tp_count = len(tp_idx_set)

    flip_path = _dice_flip_path(project, model_type, method, total_cfs, max_features)
    flips = _load_flips_long(flip_path, feat_cols)
    if flips is None or flips.empty or tp_count == 0:
        return 0, 0, tp_count, 0

    # keep only TPs
    flips = flips[flips["test_idx"].astype(int).isin(tp_idx_set)]
    if flips.empty:
        return 0, 0, tp_count, 0

    fcols = [c for c in _feature_cols(flips) if c in feat_cols]
    flipped_set = set()
    invalid_rows = 0
    neg_label = _negative_label(model)

    for t, group in flips.groupby("test_idx"):
        # reconstruct full (raw) feature matrix from original + candidate deltas
        orig = test.loc[int(t), feat_cols].astype(float)
        full_rows = []
        for _, row in group[fcols].astype(float).iterrows():
            v = orig.copy()
            v[fcols] = row.values
            full_rows.append(v.values)
        X_use_raw = np.asarray(full_rows)

        # Prepare exactly as training expected, then predict labels
        X_use = _prepare_X_for_predict(model, model_type, X_train, X_use_raw)
        preds = np.asarray(model.predict(X_use)).reshape(-1)

        # flipped if ANY candidate == negative label (e.g., 0 or -1)
        did_flip = bool(np.any(preds == neg_label))
        if did_flip:
            flipped_set.add(int(t))

        # diagnostics: count candidates that did NOT flip
        invalid_rows += int(np.sum(preds != neg_label))

    flipped = len(flipped_set)
    computed = flips["test_idx"].astype(int).nunique()
    return flipped, computed, tp_count, invalid_rows


def rq1_flip_rates(
    projects: list[str],
    model_types: list[str],
    methods: list[str],
    total_cfs: int,
    max_features: int,
    selection: str = "first",
):
    rows = []
    for model_type in model_types:
        for method in methods:
            for project in projects:
                train, test = read_dataset()[project]
                feat_cols = [c for c in test.columns if c != "target"]
                model = get_model(project, model_type)

                # --- TP set via canonical helper (fixes mismatches) ---
                tp_idx = _tp_idx_set(model, train, test)
                tp_count = len(tp_idx)

                # Load selected CFs (one per TP if exists), as full vectors
                flips_full = _load_dice_selected_full(
                    project, model_type, method, total_cfs, max_features, train, test, selection=selection
                )
                if flips_full.empty or not tp_count:
                    rows.append([model_type, method, project, 0, 0, tp_count, 0.0])
                    continue

                # keep only TPs present in CFs
                in_file = set(flips_full.index.astype(int))
                work_idx = sorted(tp_idx.intersection(in_file))
                computed = len(work_idx)
                if computed == 0:
                    rows.append([model_type, method, project, 0, 0, tp_count, 0.0])
                    continue

                # Verify flips strictly: predict labels for CF rows with same preprocessing
                neg_label = _negative_label(model)
                X_train = train[feat_cols].values
                X_cf_raw = flips_full.loc[work_idx, feat_cols].values
                X_cf = _prepare_X_for_predict(model, model_type, X_train, X_cf_raw)
                cf_pred = np.asarray(model.predict(X_cf)).reshape(-1)

                flipped_TPs = int(np.sum(cf_pred == neg_label))
                rate = flipped_TPs / tp_count if tp_count else 0.0
                rows.append([model_type, method, project, flipped_TPs, computed, tp_count, rate])

    df = pd.DataFrame(rows, columns=["Model", "Method", "Project", "Flip", "Computed", "#TP", "Flip%"])
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    out = f"./evaluations/flip_rates_DiCE_cf{total_cfs}_max{max_features}.csv"
    df.to_csv(out, index=False)

    print("\nPer-project flip rates (strict verify):")
    print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))

    model_method_means = (
        df.groupby(["Model", "Method"])[["Flip", "Computed", "#TP", "Flip%"]]
        .mean(numeric_only=True).reset_index()
    )
    print("\nPer-model-method means:")
    print(tabulate(model_method_means, headers=model_method_means.columns, tablefmt="github", showindex=False))
    print(f"\nSaved to {out}")

# ----------------------------- Selected cache loader -----------------------------

def _load_selected_cache_for_project(
    project: str,
    model_type: str,
    method: str,
    total_cfs: int,
    max_features: int,
    distance: str,
    selection_strategy: str,
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Load the parameterized selected-cache and filter to this project.
    Returns (df_for_project_or_None, path_used)
    """
    sel_path = _selected_index_file_param(model_type, method, total_cfs, max_features, distance, selection_strategy)
    if not sel_path.exists():
        return None, sel_path
    try:
        sel_df = pd.read_csv(sel_path)
    except Exception:
        return None, sel_path
    if sel_df is None or sel_df.empty or "test_idx" not in sel_df.columns:
        return None, sel_path
    if "project" in sel_df.columns:
        sel_df = sel_df[sel_df["project"].astype(str) == str(project)].copy()
    if sel_df.empty:
        return None, sel_path
    return sel_df, sel_path


# ----------------------------- RQ3: Feasibility vs historical deltas -----------------------------

def flip_feasibility(
    project_list,
    model_type,
    method,
    total_cfs: int,
    max_features: int,
    *,
    distance="mahalanobis",
    selection_strategy="best",
    save_selected: bool = True,
) -> Tuple[list, int, int, list]:
    """
    Feasibility vs historical deltas.
    - selection_strategy='first': choose first candidate per test_idx
    - selection_strategy='best' : evaluate all and keep the one with the lowest *min* distance
    Returns: (results_all, total_seen, cannot, selected_records)
      where selected_records are dicts with full features + stats for caching.
    """
    ds = read_dataset()

    # Build historical delta pool once
    total_deltas = pd.DataFrame()
    for project in project_list:
        train, test = ds[project]
        common = train.index.intersection(test.index)
        deltas = test.loc[common, test.columns != "target"] - \
                 train.loc[common, train.columns != "target"]
        total_deltas = pd.concat([total_deltas, deltas], axis=0)

    cannot = 0
    results_all = []
    total_seen = 0
    selected_records = []

    for project in project_list:
        train, test = ds[project]
        feat_cols = [c for c in test.columns if c != "target"]

        flip_path = _dice_flip_path(project, model_type, method, total_cfs, max_features)
        flips_long = _load_flips_long(flip_path, feat_cols)
        if flips_long is None or flips_long.empty:
            continue

        fcols_present = [c for c in _feature_cols(flips_long) if c in feat_cols]

        for test_idx, g in flips_long.groupby("test_idx", sort=False):
            total_seen += 1
            original_row = test.loc[int(test_idx), feat_cols].astype(float)

            # choose rows to evaluate
            if selection_strategy == "first":
                g_eval = g.iloc[[0]]
            else:
                g_eval = g

            best = None
            best_cand_row = None
            best_cand_id = None

            for _, cand in g_eval.iterrows():
                # Handle full or partial candidate feature vectors
                has_full = set(feat_cols).issubset(cand.index)
                if has_full:
                    cand_vec = cand[feat_cols].astype(float).values
                    base_vec = original_row.values
                    names = feat_cols
                else:
                    cand_vec = cand[fcols_present].astype(float).values
                    base_vec = original_row[fcols_present].values
                    names = fcols_present

                # detect actually changed features (ALWAYS vs ORIGINAL)
                changed_mask = ~np.isclose(cand_vec, base_vec, rtol=1e-7, atol=1e-7)
                if not np.any(changed_mask):
                    continue

                # delta vector over changed features
                idxs = np.where(changed_mask)[0]
                x = pd.Series(cand_vec[changed_mask] - base_vec[changed_mask],
                              index=[names[i] for i in idxs], dtype=float)

                # strict historical pool: same features & all non-zero deltas
                sub = total_deltas[x.index].dropna()
                sub = sub.loc[(sub != 0).all(axis=1)]
                if sub.empty:
                    continue

                # distances
                if distance == "cosine":
                    dists = cosine_all(sub, x)
                elif distance == "mahalanobis":
                    if len(sub) <= len(x.index):
                        continue
                    dists = mahalanobis_all(sub, x)
                else:
                    raise ValueError("distance must be 'mahalanobis' or 'cosine'")

                if not dists:
                    continue

                cand_stats = {
                    "min": float(np.min(dists)),
                    "max": float(np.max(dists)),
                    "mean": float(np.mean(dists)),
                }

                if selection_strategy == "first":
                    best = cand_stats
                    best_cand_row = cand
                    best_cand_id = cand.get("candidate_id", np.nan)
                    break
                else:
                    if (best is None) or (cand_stats["min"] < best["min"]):
                        best = cand_stats
                        best_cand_row = cand
                        best_cand_id = cand.get("candidate_id", np.nan)

            if best is not None:
                results_all.append(best)

                # Build a *full* feature vector for caching by overlaying onto the original row
                if best_cand_row is not None:
                    if set(feat_cols).issubset(best_cand_row.index):
                        full_series = best_cand_row[feat_cols].astype(float).copy()
                    else:
                        full_series = original_row.copy()
                        present = [c for c in best_cand_row.index if c in feat_cols]
                        full_series[present] = best_cand_row[present].astype(float).values

                    sel_rec = {
                        "project": project,
                        "test_idx": int(test_idx),
                        "candidate_id": int(best_cand_id) if pd.notna(best_cand_id) else np.nan,
                        "min": best["min"],
                        "max": best["max"],
                        "mean": best["mean"],
                    }
                    for f in feat_cols:
                        sel_rec[f] = float(full_series[f])
                    selected_records.append(sel_rec)
            else:
                cannot += 1

    return results_all, total_seen, cannot, selected_records


def rq3_feasibility(
    model_types: list[str],
    projects: list[str],
    methods: list[str],
    total_cfs: int,
    max_features: int,
    distance: str = "mahalanobis",
    use_default_groups: bool = True,
    selection_strategy: str = "best",
    use_cached_selected: bool = False,
):
    Path(f"./evaluations/feasibility/{distance}").mkdir(parents=True, exist_ok=True)
    Path(SELECTED_DIR).mkdir(parents=True, exist_ok=True)

    ds = read_dataset()
    all_projects = list(sorted(ds.keys()))
    groups = DEFAULT_GROUPS if use_default_groups else [[p] for p in (projects or all_projects)]

    summary = []

    for m in model_types:
        for method in methods:
            # ----- fast path: use cached selected if requested and exists -----
            if use_cached_selected or selection_strategy == "selected":
                sel_path = _selected_index_file_param(m, method, total_cfs, max_features, distance, "best")
                if sel_path.exists():
                    try:
                        cached = pd.read_csv(sel_path)
                        # filter to requested projects if applicable
                        if projects is not None and projects != all_projects:
                            cached = cached[cached["project"].isin(projects)]
                        if not cached.empty and all(k in cached.columns for k in ["min", "max", "mean"]):
                            out = f"./evaluations/feasibility/{distance}/{MODEL_ABBR.get(m, m)}_DiCE_{method}_selected_cf{total_cfs}_max{max_features}.csv"
                            cached[["min", "max", "mean"]].to_csv(out, index=False)
                            summary.append([m, method, "DiCE",
                                            cached["min"].mean(),
                                            cached["max"].max(),
                                            cached["mean"].mean()])
                            print(f"[{m}/{method}] Loaded cached selections → {sel_path}")
                            continue
                        else:
                            print(f"[{m}/{method}] Cached file missing stats; recomputing.")
                    except Exception as e:
                        print(f"[{m}/{method}] Failed to read cached selections ({e}); recomputing.")

                if selection_strategy == "selected":
                    print(f"[{m}/{method}] No cached selections found at {sel_path}. Skipping.")
                    continue  # explicit 'selected' with no cache: skip

            # ----- compute normally, but also produce a selected cache -----
            all_rows, totals, cannots = [], 0, 0
            selected_accum = []
            for g in groups:
                rows, tot, cannot, selected_records = flip_feasibility(
                    g, m, method, total_cfs, max_features,
                    distance=distance, selection_strategy=selection_strategy, save_selected=True
                )
                totals += tot
                cannots += cannot
                all_rows.extend(rows)
                selected_accum.extend(selected_records)

            if all_rows:
                df = pd.DataFrame(all_rows)
                out = f"./evaluations/feasibility/{distance}/{MODEL_ABBR.get(m, m)}_DiCE_{method}_{selection_strategy}_cf{total_cfs}_max{max_features}.csv"
                df.to_csv(out, index=False)
                summary.append([
                    m, method, "DiCE",
                    df["min"].mean(),
                    df["max"].max(),
                    df["mean"].mean()
                ])
            print(f"[{m}/{method}/{selection_strategy}] totals={totals}, cannot={cannots}")

            # write the selected cache (full features + stats)
            if selected_accum:
                sel_df = pd.DataFrame(selected_accum)
                sel_path = _selected_index_file_param(
                    m, method, total_cfs, max_features, distance,
                    selection_strategy if selection_strategy != "selected" else "best"
                )
                sel_df.to_csv(sel_path, index=False)
                print(f"[{m}/{method}] Saved selected candidates → {sel_path}")

    if summary:
        s = pd.DataFrame(summary, columns=["Model", "Method", "Explainer", "Min", "Max", "Mean"])
        s.to_csv(f"./evaluations/feasibility_{distance}_DiCE_all_methods_{selection_strategy}_cf{total_cfs}_max{max_features}.csv", index=False)
        print("\nFeasibility summary:")
        print(tabulate(s, headers=s.columns, tablefmt="github", showindex=False))


# ----------------------------- Implications (no plans; direct diff) -----------------------------

def _build_historical_deltas():
    ds = read_dataset()
    total = pd.DataFrame()
    for proj, (train, test) in ds.items():
        common = train.index.intersection(test.index)
        if len(common) == 0:
            continue
        d = test.loc[common, test.columns != "target"] - \
            train.loc[common, train.columns != "target"]
        total = pd.concat([total, d], axis=0)
    return total


def implications(
    project: str,
    model_type: str,
    method: str,
    total_cfs: int,
    max_features: int,
    selection_strategy: str = "best",
    distance: str = "mahalanobis",
):
    """
    Total amount of change (scaled) = sum |z(flipped)-z(original)| over changed features.
    selection_strategy: 'first' or 'best' (compute) or 'selected' (use cached).
    """
    ds = read_dataset()
    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]
    scaler = StandardScaler().fit(train.drop("target", axis=1).values)
    print(f"[{project}/{model_type}/{method}] Implications using selection_strategy={selection_strategy}")
    # ---- use selected cache if asked ----
    if selection_strategy == "selected":
        sel_df, _ = _load_selected_cache_for_project(project, model_type, method, total_cfs, max_features, distance, "best")
        if sel_df is None or sel_df.empty:
            return []
        totals = []
        for ti, g in sel_df.groupby("test_idx", sort=False):
            original_row = test.loc[int(ti), feat_cols].astype(float)
            cand = g.iloc[0]
            flipped_row = cand[feat_cols].astype(float)
            changed = ~np.isclose(flipped_row.values, original_row.values, rtol=1e-7, atol=1e-7)
            if not np.any(changed):
                continue
            zf = scaler.transform([flipped_row.values])[0]
            zo = scaler.transform([original_row.values])[0]
            totals.append(float(np.abs(zf - zo)[changed].sum()))
        return totals

    # ---- otherwise compute from long flips ----
    flip_path = _dice_flip_path(project, model_type, method, total_cfs, max_features)
    flips_long = _load_flips_long(flip_path, feat_cols) if flip_path.exists() else None

    if flips_long is None or flips_long.empty:
        return []

    totals = []
    total_deltas = _build_historical_deltas() if selection_strategy == "best" else None

    for test_idx, g in flips_long.groupby("test_idx", sort=False):
        original_row = test.loc[int(test_idx), feat_cols].astype(float)

        if selection_strategy == "first":
            cand = g.iloc[0]
            flipped_row = cand[feat_cols].astype(float)
            changed_mask = ~np.isclose(flipped_row.values, original_row.values, rtol=1e-7, atol=1e-7)
            if not np.any(changed_mask):
                continue
            zf = scaler.transform([flipped_row.values])[0]
            zo = scaler.transform([original_row.values])[0]
            totals.append(float(np.abs(zf - zo)[changed_mask].sum()))
            continue

        # selection_strategy == "best"
        best_key, best_cand = None, None
        for _, cand in g.iterrows():
            # robust changed-feature detection (always vs original)
            flipped_row = cand[feat_cols].astype(float)
            changed_mask = ~np.isclose(flipped_row.values, original_row.values, rtol=1e-7, atol=1e-7)
            if not np.any(changed_mask):
                continue

            names = [feat_cols[i] for i in np.where(changed_mask)[0]]
            x = pd.Series((flipped_row.values - original_row.values)[changed_mask], index=names, dtype=float)

            sub = total_deltas[names].dropna()
            sub = sub.loc[(sub != 0).all(axis=1)]
            if sub.empty or (len(sub) <= len(names)):
                continue

            dists = mahalanobis_all(sub, x)
            if not dists:
                continue

            key = float(np.min(dists))  # choose candidate with smallest min distance
            if (best_key is None) or (key < best_key):
                best_key, best_cand = key, cand

        if best_cand is None:
            continue
        flipped_row = best_cand[feat_cols].astype(float)
        changed_mask = ~np.isclose(flipped_row.values, original_row.values, rtol=1e-7, atol=1e-7)
        if not np.any(changed_mask):
            continue
        zf = scaler.transform([flipped_row.values])[0]
        zo = scaler.transform([original_row.values])[0]
        totals.append(float(np.abs(zf - zo)[changed_mask].sum()))

    return totals


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="DiCE/kfeatures evaluation (cached best selection): RQ1, RQ3, Implications")
    ap.add_argument("--rq1", action="store_true", help="Flip rates (strict verify)")
    ap.add_argument("--rq3", action="store_true", help="Feasibility vs historical deltas (cache-aware)")
    ap.add_argument("--implications", action="store_true", help="Total scaled change (no plans)")

    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--methods", type=str, default="kfeature",
                    help="Comma-separated methods: random,kdtree,genetic,kfeature")
    ap.add_argument("--projects", type=str, default="all",
                    help="Project name(s) or 'all' (space/comma separated allowed)")

    ap.add_argument("--total_cfs", type=int, default=1,
                    help="total_CFs used in the generator (for file path)")
    ap.add_argument("--max_features", type=int, default=5,
                    help="max features changed used in the generator (for file path)")

    ap.add_argument("--distance", type=str, default="mahalanobis", choices=["mahalanobis", "cosine"],
                    help="Distance metric for RQ3")
    ap.add_argument("--use_default_groups", action="store_true",
                    help="Use predefined release groups for RQ3")
    ap.add_argument("--selection_strategy", type=str, choices=["best", "first", "selected"], default="best",
                    help="Selection for RQ3 and Implications: 'best' or 'first' (compute) or 'selected' (use cache)")
    ap.add_argument("--use_cached_selected", action="store_true",
                    help="If set and cached selected file exists, load it to skip recomputation")

    args = ap.parse_args()
    print(args.total_cfs)
    model_types = [m.strip() for m in args.models.replace(",", " ").split() if m.strip()]
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]

    valid_methods = ["random", "kdtree", "genetic", "kfeature"]
    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        print(f"ERROR: Invalid methods: {invalid_methods}")
        print(f"Valid methods are: {valid_methods}")
        return

    all_ds = read_dataset()
    if args.projects == "all":
        project_list = list(sorted(all_ds.keys()))
    else:
        project_list = [p.strip() for p in args.projects.replace(",", " ").split() if p.strip()]
    
    print(f"Evaluating {len(model_types)} models × {len(methods)} methods × {len(project_list)} projects")
    print(f"Models:  {model_types}")
    print(f"Methods: {methods}")
    print(f"Projects: {project_list[:3]}{'...' if len(project_list) > 3 else ''}")
    print(f"Using flip files: DiCE_all_{args.total_cfs}_max{args.max_features}feat.csv")
    print()

    if args.rq1:
        print("=== RQ1: Flip Rates (strict verification) ===")
        rq1_flip_rates(model_types, project_list, methods, args.total_cfs, args.max_features)

    if args.rq3:
        print(f"\n=== RQ3: Feasibility — strategy: {args.selection_strategy} ===")
        rq3_feasibility(
            model_types, project_list, methods,
            total_cfs=args.total_cfs,
            max_features=args.max_features,
            distance=args.distance,
            use_default_groups=args.use_default_groups,
            selection_strategy=args.selection_strategy,
            use_cached_selected=args.use_cached_selected,
        )

    if args.implications:
        print("\n=== Implications (no plans) ===")
        rows = []
        for m in model_types:
            for method in methods:
                all_scores = []
                for p in project_list:
                    vals = implications(
                        p, m, method,
                        total_cfs=args.total_cfs,
                        max_features=args.max_features,
                        selection_strategy=args.selection_strategy,
                        distance=args.distance,
                    )
                    all_scores.extend(vals)
                if all_scores:
                    out = f"./evaluations/abs_changes/{MODEL_ABBR.get(m, m)}_DiCE_{method}_{args.selection_strategy}_cf{args.total_cfs}_max{args.max_features}.csv"
                    pd.DataFrame(all_scores, columns=["score"]).to_csv(out, index=False)
                    rows.append([MODEL_ABBR.get(m, m), method, np.mean(all_scores)])
        if rows:
            tdf = pd.DataFrame(rows, columns=["Model", "Method", "Mean"])
            print(tabulate(tdf, headers=tdf.columns, tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
