#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DiCE-only evaluation that replicates the baseline script's behavior
(except for file paths/locations). RQ2 is skipped (DiCE has no plans_all.json).
- RQ1: Flip rates per model/method/project
- RQ3: Feasibility vs historical deltas (cosine | mahalanobis), supports best|first
- Implications: total scaled change without plans (diff flipped vs original)
"""

import math
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from tabulate import tabulate

from hyparams import PROPOSED_CHANGES, EXPERIMENTS  # PROPOSED_CHANGES unused (kept for parity)
from data_utils import read_dataset, get_model, get_true_positives

# ----------------------------- config / helpers -----------------------------
# Where selected Mahalanobis-best choices are stored
SELECTED_DIR = "./evaluations/feasibility/mahalanobis/selected"

def _selected_index_file(model_type: str, method: str) -> Path:
    """Selected rows / indices across projects for a model/method."""
    abbr = MODEL_ABBR.get(model_type, model_type)
    return Path(SELECTED_DIR) / f"{abbr}_DiCE_{method}_selected.csv"

def _load_selected_index_for_project(project: str, model_type: str, method: str):
    """
    Load selected rows for this project from the precomputed 'selected' file.
    Returns:
      - a DataFrame of selected rows if it already contains features; or
      - a set/dict used to filter the long DiCE file by (test_idx[, candidate_id]).
    Expected columns in selected file (best effort):
      - must have: 'test_idx'
      - nice to have: 'project', 'candidate_id' (and/or all feature columns)
      - always has: 'min' (distance) – used in RQ3 plots
    """
    sel_path = _selected_index_file(model_type, method)
    if not sel_path.exists():
        return None, None

    try:
        sel_df = pd.read_csv(sel_path)
    except Exception:
        return None, None

    if sel_df is None or sel_df.empty or "test_idx" not in sel_df.columns:
        return None, None

    # Filter to current project if a project column exists
    if "project" in sel_df.columns:
        sel_df = sel_df[sel_df["project"].astype(str) == str(project)].copy()

    if sel_df.empty:
        return None, None

    # If selection file already contains feature columns, we can return it directly
    # Heuristic: contains more than just identifiers & stats
    id_like = {"project", "test_idx", "candidate_id", "min", "max", "mean"}
    feat_like_cols = [c for c in sel_df.columns if c not in id_like]
    if any(c not in {"project", "test_idx", "candidate_id"} for c in feat_like_cols):
        # Looks like full rows are present
        return sel_df, None

    # Otherwise, build an index filter for (test_idx[, candidate_id])
    if "candidate_id" in sel_df.columns:
        # Map test_idx -> set(candidate_id)
        idx_map = {}
        for _, r in sel_df.iterrows():
            ti = int(r["test_idx"])
            cid = int(r["candidate_id"])
            idx_map.setdefault(ti, set()).add(cid)
        return None, idx_map
    else:
        # Just a set of test_idx to keep
        idx_set = set(sel_df["test_idx"].astype(int).tolist())
        return None, idx_set

MODEL_ABBR = {"SVM": "SVM", "RandomForest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM", "CatBoost": "CatB"}

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

def _dice_flip_path(project: str, model_type: str, method: str, use_hist_seed: bool) -> Path:
    """DiCE long-format outputs live under a method subdir."""
    filename = "DiCE_all_100_nofeat.csv"
    return Path(EXPERIMENTS) / project / model_type / method / filename

def _feature_cols(df: pd.DataFrame) -> list[str]:
    non_feats = {"test_idx", "candidate_id", "proba0", "proba1", "target"}
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

    # Ensure test_idx exists
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
    combinations = []
    feature_values = []
    for feature in data:
        feature_values.append(data[feature])
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
        inv_cov_matrix = (
            np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
        )
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

    max_vector_distance = mahalanobis(
        min_vector_standardized, max_vector_standardized, inv_cov_matrix
    )

    normalized_distance = (
        distance / max_vector_distance if max_vector_distance != 0 else 0
    )
    return normalized_distance

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        # baseline behavior prints and returns 0
        print(vec1, vec2)
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def cosine_all(df, x):
    distances = []
    for _, row in df.iterrows():
        distance = cosine_similarity(x, row)
        distances.append(distance)
    return distances

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
        inv_cov_matrix = (
            np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
        )
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

    max_vector_distance = mahalanobis(
        min_vector_standardized, max_vector_standardized, inv_cov_matrix
    )

    distances = []
    for _, y in df.iterrows():
        y_standardized = [
            (y[feature] - df[feature].mean()) / df[feature].std()
            for feature in df.columns
        ]
        distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)
        distances.append(
            distance / max_vector_distance if max_vector_distance != 0 else 0
        )
    return distances

# ----------------------------- RQ1: Flip rates (DiCE) -----------------------------

def _count_flips_for_project_model(project: str, model_type: str, method: str, use_hist_seed: bool = False) -> tuple[int, int, int]:
    """
    Returns (flipped_TPs, unique_testidx_in_file, num_TP).
    A TP is counted as flipped if *any* saved candidate for that test_idx predicts class 0.
    """
    ds = read_dataset()
    if project not in ds:
        return 0, 0, 0

    train, test = ds[project]
    feat_cols = list(_features_only(test).columns)

    model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)

    # True positives (defects predicted as defects)
    tp_df = get_true_positives(model, train, test)
    tp_idx_set = set(tp_df.index.astype(int).tolist())
    tp_count = len(tp_idx_set)

    flip_path = _dice_flip_path(project, model_type, method, use_hist_seed)
    flips = _load_flips_long(flip_path, feat_cols)
    if flips is None or flips.empty:
        return 0, 0, tp_count

    # consider only genuine TPs
    flips = flips[flips["test_idx"].astype(int).isin(tp_idx_set)]
    if flips.empty:
        return 0, 0, tp_count

    fcols = [c for c in _feature_cols(flips) if c in feat_cols]
    flipped_set = set()

    for t, group in flips.groupby("test_idx"):
        X = group[fcols].astype(float).values

        # reconstruct to full feature order if needed
        orig = test.loc[int(t), feat_cols].astype(float)
        full_rows = []
        for r in X:
            v = orig.copy()
            v[fcols] = r
            full_rows.append(v.values)
        X_use = np.asarray(full_rows)

        preds = model.predict(scaler.transform(X_use))
        if np.any(preds == 0):
            flipped_set.add(int(t))

    flipped = len(flipped_set)
    computed = flips["test_idx"].astype(int).nunique()
    return flipped, computed, tp_count

def rq1_flip_rates(model_types: list[str], projects: list[str], methods: list[str], use_hist_seed: bool = False):
    ds = read_dataset()
    project_list = list(sorted(ds.keys())) if projects is None else projects
    rows = []

    for m in model_types:
        for method in methods:
            for p in project_list:
                flipped, computed, tp_count = _count_flips_for_project_model(p, m, method, use_hist_seed)
                rate = flipped / tp_count if tp_count > 0 else 0.0
                rows.append([m, method, p, flipped, computed, tp_count, rate])

    df = pd.DataFrame(rows, columns=["Model", "Method", "Project", "Flip", "Computed", "#TP", "Flip%"])

    # Save & print
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    suffix = "_hist_seed" if use_hist_seed else ""
    out = f"./evaluations/flip_rates_DiCE_all_methods{suffix}.csv"
    df.to_csv(out, index=False)

    print("\nPer-project flip rates (DiCE):")
    print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))

    # Simple per-model-method summary (mean of columns, like your earlier DiCE eval)
    model_method_means = (
        df.groupby(["Model", "Method"])[["Flip", "Computed", "#TP", "Flip%"]]
        .mean(numeric_only=True).reset_index()
    )
    print("\nPer-model-method means (DiCE):")
    print(tabulate(model_method_means, headers=model_method_means.columns, tablefmt="github", showindex=False))
    print(f"\nSaved to {out}")

# ----------------------------- RQ3: Feasibility vs historical deltas (DiCE) -----------------------------

def flip_feasibility(
    project_list,
    model_type,
    method,
    *,
    distance="mahalanobis",
    selection_strategy="best",
    use_hist_seed=False
):
    """
    DiCE feasibility vs historical deltas.
    - selection_strategy='first': use the first candidate row per test_idx
    - selection_strategy='best' : evaluate all candidates and keep the one with the lowest *mean* distance
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

    for project in project_list:
        train, test = ds[project]
        feat_cols = [c for c in test.columns if c != "target"]

        flip_path = _dice_flip_path(project, model_type, method, use_hist_seed)
        flips_long = _load_flips_long(flip_path, feat_cols)
        if flips_long is None or flips_long.empty:
            continue

        for test_idx, g in flips_long.groupby("test_idx", sort=False):
            total_seen += 1
            original_row = test.loc[test_idx, feat_cols].astype(float)

            # choose rows to evaluate
            if selection_strategy == "first":
                g_eval = g.iloc[[0]]
            else:  # 'best' → evaluate all, pick lowest mean distance later
                g_eval = g

            best = None
            for _, cand in g_eval.iterrows():
                cand_row = cand[feat_cols].astype(float)

                # detect actually changed features
                changed_mask = ~np.isclose(cand_row.values, original_row.values, rtol=1e-7, atol=1e-7)
                if not np.any(changed_mask):
                    continue

                changed_features = {feat_cols[i]: (cand_row.values[i] - original_row.values[i])
                                    for i in np.where(changed_mask)[0]}
                names = list(changed_features.keys())
                x = pd.Series(changed_features, index=names, dtype=float)

                # strict historical pool: same features & all non-zero deltas
                sub = total_deltas[names].dropna()
                sub = sub.loc[(sub != 0).all(axis=1)]
                if sub.empty:
                    continue

                # distances
                if distance == "cosine":
                    dists = cosine_all(sub, x)
                elif distance == "mahalanobis":
                    # classical requirement: rows > features
                    if len(sub) <= len(names):
                        # print(f"Skipping test_idx {test_idx} in {project} ({len(sub)} rows ≤ {len(names)} features)")
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
                    break
                else:
                    if (best is None) or (cand_stats["min"] < best["min"]):
                        best = cand_stats

            if best is not None:
                results_all.append(best)
            else:
                cannot += 1

    return results_all, total_seen, cannot

def rq3_feasibility(
    model_types: list[str],
    projects: list[str],
    methods: list[str],
    distance: str = "mahalanobis",
    use_default_groups: bool = True,
    selection_strategy: str = "best",
    use_hist_seed: bool = False
):
    Path(f"./evaluations/feasibility/{distance}").mkdir(parents=True, exist_ok=True)

    ds = read_dataset()
    all_projects = list(sorted(ds.keys()))
    groups = DEFAULT_GROUPS if use_default_groups else [[p] for p in (projects or all_projects)]

    summary = []

    for m in model_types:
        for method in methods:
            # If user wants the preselected set, just read them and write to the standard location
            if selection_strategy == "selected":
                sel_path = _selected_index_file(m, method)
                if not sel_path.exists():
                    print(f"[{m}/{method}] No selected file at {sel_path}. Skipping.")
                    continue
                try:
                    df = pd.read_csv(sel_path)
                except Exception as e:
                    print(f"[{m}/{method}] Failed to read selected file: {e}")
                    continue
                if df is None or df.empty or "min" not in df.columns:
                    print(f"[{m}/{method}] Selected file missing 'min' column. Skipping.")
                    continue

                # Normalize/clamp just in case
                df["min"] = pd.to_numeric(df["min"], errors="coerce").clip(0, 1)
                out = f"./evaluations/feasibility/{distance}/{MODEL_ABBR.get(m, m)}_DiCE_{method}_selected.csv"
                df.to_csv(out, index=False)
                summary.append([m, method, "DiCE",
                                df["min"].mean(),
                                df["min"].max(),
                                df["min"].mean()])  # keep structure; 'max' is real, 'mean' repeated here
                print(f"[{m}/{method}] Loaded {len(df)} selected distances → {out}")
                continue

            # Otherwise, fall back to computing (best/first) like before
            all_rows, totals, cannots = [], 0, 0
            for g in groups:
                rows, tot, cannot = flip_feasibility(
                    g, m, method,
                    distance=distance,
                    selection_strategy=selection_strategy,
                    use_hist_seed=use_hist_seed
                )
                totals += tot
                cannots += cannot
                all_rows.extend(rows)

            if all_rows:
                df = pd.DataFrame(all_rows)
                out = f"./evaluations/feasibility/{distance}/{MODEL_ABBR.get(m, m)}_DiCE_{method}_{selection_strategy}{'_100'}.csv"
                df.to_csv(out, index=False)
                summary.append([
                    m, method, "DiCE",
                    df["min"].mean(),
                    df["max"].max(),
                    df["mean"].mean()
                ])
            print(f"[{m}/{method}/{selection_strategy}{' (hist_seed2)' if use_hist_seed else ''}] totals={totals}, cannot={cannots}")

    if summary:
        s = pd.DataFrame(summary, columns=["Model", "Method", "Explainer", "Min", "Max", "Mean"])
        s.to_csv(f"./evaluations/feasibility_{distance}_DiCE_all_methods_{selection_strategy}{'_100' if selection_strategy!='selected' else ''}.csv", index=False)
        print("\nFeasibility summary:")
        print(tabulate(s, headers=s.columns, tablefmt="github", showindex=False))


# ----------------------------- Implications (no plans; direct diff) -----------------------------

def _build_historical_deltas():
    """Pool historical deltas across all projects (test - train on overlapping rows)."""
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


def implications(project: str,
                 model_type: str,
                 method: str,
                 use_hist_seed: bool = False,
                 selection_strategy: str = "best"):
    """
    If selection_strategy == 'selected': use the pre-saved Mahalanobis-best rows.
    Else: 'first' or 'best' (the previous logic).
    Total amount of change (scaled) = sum |z(flipped)-z(original)| over changed features.
    """
    ds = read_dataset()
    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]
    scaler = StandardScaler().fit(train.drop("target", axis=1).values)

    flip_path = _dice_flip_path(project, model_type, method, use_hist_seed)
    flips_long = _load_flips_long(flip_path, feat_cols) if flip_path.exists() else None

    if selection_strategy == "selected":
        # Try to load selected rows for this project
        sel_rows, sel_filter = _load_selected_index_for_project(project, model_type, method)
        if sel_rows is not None:
            # Selected file already contains feature columns → use directly
            # Ensure features only
            need_cols = ["test_idx"] + [c for c in feat_cols if c in sel_rows.columns]
            sel_rows = sel_rows.loc[:, [c for c in need_cols if c in sel_rows.columns]].copy()
            groups = sel_rows.groupby("test_idx", sort=False)
        else:
            # Need to filter the long DiCE file by (test_idx[, candidate_id])
            if flips_long is None or flips_long.empty or sel_filter is None:
                return []
            if isinstance(sel_filter, dict):
                # test_idx -> set(candidate_id)
                mask_list = []
                for ti, group in flips_long.groupby("test_idx", sort=False):
                    if int(ti) in sel_filter:
                        keep_cids = sel_filter[int(ti)]
                        if "candidate_id" in group.columns:
                            mask_list.append(group[group["candidate_id"].astype(int).isin(keep_cids)])
                        else:
                            # No candidate_id saved → keep first row
                            mask_list.append(group.iloc[[0]])
                sel_rows = pd.concat(mask_list, axis=0) if mask_list else pd.DataFrame()
            else:
                # sel_filter is a set of test_idx
                sel_rows = flips_long[flips_long["test_idx"].astype(int).isin(sel_filter)].copy()

            groups = sel_rows.groupby("test_idx", sort=False)

        totals = []
        for ti, g in groups:
            original_row = test.loc[int(ti), feat_cols].astype(float)
            # use the (single) selected row per test_idx
            cand = g.iloc[0]
            flipped_row = cand[feat_cols].astype(float)
            changed = ~np.isclose(flipped_row.values, original_row.values, rtol=1e-7, atol=1e-7)
            if not np.any(changed):
                continue
            zf = scaler.transform([flipped_row.values])[0]
            zo = scaler.transform([original_row.values])[0]
            totals.append(float(np.abs(zf - zo)[changed].sum()))
        return totals

    # --- original 'first' / 'best' logic (unchanged except small cleanup) ---
    if flips_long is None or flips_long.empty:
        return []

    totals = []
    # optional: build historical deltas only if 'best' (we won't need for 'first')
    if selection_strategy == "best":
        total_deltas = _build_historical_deltas()

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

        # selection_strategy == "best": Mahalanobis-best (like RQ3)
        best_key, best_cand = None, None
        for _, cand in g.iterrows():
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

            key = float(np.min(dists))  # pick the candidate with smallest 'min' distance
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
    ap = ArgumentParser(description="DiCE-only evaluation (RQ1, RQ3, Implications)")
    ap.add_argument("--rq1", action="store_true", help="Flip rates")
    ap.add_argument("--rq3", action="store_true", help="Feasibility vs historical deltas")
    ap.add_argument("--implications", action="store_true", help="Total scaled change (no plans)")

    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated DiCE methods: random,kdtree,genetic")
    ap.add_argument("--projects", type=str, default="all",
                    help="Project name(s) or 'all' (space/comma separated allowed)")
    ap.add_argument("--distance", type=str, default="mahalanobis", choices=["mahalanobis", "cosine"],
                    help="Distance metric for RQ3")
    ap.add_argument("--use_default_groups", action="store_true",
                    help="Use predefined release groups for RQ3")
    ap.add_argument("--selection_strategy", type=str, choices=["best", "first"], default="best",
                    help="RQ3 selection: 'best' (lowest mean distance) or 'first' (first candidate)")
    ap.add_argument("--use_hist_seed", action="store_true",
                    help="Use historical seeding results (DiCE_all_hist_seed2.csv)")
    args = ap.parse_args()

    model_types = [m.strip() for m in args.models.replace(",", " ").split() if m.strip()]
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]

    # validate methods
    valid_methods = ["random", "kdtree", "genetic"]
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
    if args.use_hist_seed:
        print("Using historical seeding results")
    print()

    if args.rq1:
        print("=== Running RQ1: Flip Rates (DiCE) ===")
        rq1_flip_rates(model_types, project_list, methods, args.use_hist_seed)

    if args.rq3:
        print(f"\n=== Running RQ3: Feasibility (DiCE) — strategy: {args.selection_strategy} ===")
        rq3_feasibility(
            model_types, project_list, methods,
            distance=args.distance,
            use_default_groups=args.use_default_groups,
            selection_strategy=args.selection_strategy,
            use_hist_seed=args.use_hist_seed
        )

    if args.implications:
        print("\n=== Running Implications (DiCE, no plans) ===")
        rows = []
        for m in model_types:
            for method in methods:
                all_scores = []
                for p in project_list:
                    # vals = implications(p, m, method, args.use_hist_seed)
                    vals = implications(p, m, method, args.use_hist_seed, selection_strategy=args.selection_strategy)

                    all_scores.extend(vals)
                if all_scores:
                    out = f"./evaluations/abs_changes/{MODEL_ABBR.get(m, m)}_DiCE_{method}_{args.selection_strategy}{'_hist_seed2' if args.use_hist_seed else '_100'}.csv"
                    pd.DataFrame(all_scores, columns=["score"]).to_csv(out, index=False)
                    rows.append([MODEL_ABBR.get(m, m), method, np.mean(all_scores)])
        if rows:
            tdf = pd.DataFrame(rows, columns=["Model", "Method", "Mean"])
            print(tabulate(tdf, headers=tdf.columns, tablefmt="github", showindex=False))

if __name__ == "__main__":
    main()
