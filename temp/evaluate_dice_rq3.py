#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import json
import warnings
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from pandas.errors import EmptyDataError

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler

from hyparams import PROPOSED_CHANGES, EXPERIMENTS
from data_utils import read_dataset, get_model, get_output_dir, get_true_positives

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- Utilities ---------------------------

MODEL_MAP = {
    "SVM": "SVM",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

# Default groups (same shape as your original RQ3 evaluation)
DEFAULT_PROJECT_GROUPS = [
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

def _generate_all_combinations(data_dict: dict[str, list]) -> pd.DataFrame:
    """Cartesian product of value lists in data_dict -> DataFrame."""
    if not data_dict:
        return pd.DataFrame()
    keys = list(data_dict.keys())
    values = [data_dict[k] for k in keys]
    combos = list(product(*values))
    return pd.DataFrame(combos, columns=keys)

def _normalized_mahalanobis_distance(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> float:
    """Normalized Mahalanobis distance between x and y using df's covariance."""
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0.0

    standardized_df = (df - df.mean()) / df.std()

    x_std = [(x[f] - df[f].mean()) / df[f].std() for f in df.columns]
    y_std = [(y[f] - df[f].mean()) / df[f].std() for f in df.columns]

    cov = np.cov(standardized_df.T)
    inv_cov = np.linalg.pinv(cov) if cov.ndim > 0 else (np.array([[1/cov]]) if cov != 0 else np.array([[np.inf]]))

    dist = mahalanobis(x_std, y_std, inv_cov)

    min_vec = np.array([df[f].min() for f in df.columns])
    max_vec = np.array([df[f].max() for f in df.columns])
    min_std = [(min_vec[i] - df[f].mean()) / df[f].std() for i, f in enumerate(df.columns)]
    max_std = [(max_vec[i] - df[f].mean()) / df[f].std() for i, f in enumerate(df.columns)]
    max_span = mahalanobis(min_std, max_std, inv_cov)

    return float(dist / max_span) if max_span != 0 else 0.0

def _mahalanobis_all(df: pd.DataFrame, x: pd.Series) -> list[float]:
    """Normalized Mahalanobis distance from x to every row in df."""
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return []

    standardized_df = (df - df.mean()) / df.std()
    cov = np.cov(standardized_df.T)
    inv_cov = np.linalg.pinv(cov) if cov.ndim > 0 else (np.array([[1/cov]]) if cov != 0 else np.array([[np.inf]]))

    min_vec = np.array([df[f].min() for f in df.columns])
    max_vec = np.array([df[f].max() for f in df.columns])
    min_std = [(min_vec[i] - df[f].mean()) / df[f].std() for i, f in enumerate(df.columns)]
    max_std = [(max_vec[i] - df[f].mean()) / df[f].std() for i, f in enumerate(df.columns)]
    max_span = mahalanobis(min_std, max_std, inv_cov)
    if max_span == 0:
        return [0.0] * len(df)

    x_std = [(x[f] - df[f].mean()) / df[f].std() for f in df.columns]
    out = []
    for _, y in df.iterrows():
        y_std = [(y[f] - df[f].mean()) / df[f].std() for f in df.columns]
        d = mahalanobis(x_std, y_std, inv_cov)
        out.append(float(d / max_span))
    return out

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = float(np.dot(vec1, vec2))
    n1 = float(np.linalg.norm(vec1))
    n2 = float(np.linalg.norm(vec2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def _cosine_all(df: pd.DataFrame, x: pd.Series) -> list[float]:
    x_arr = x.values.astype(float)
    return [_cosine_similarity(x_arr, row.values.astype(float)) for _, row in df.iterrows()]

# ------------------- DiCE flip reconstruction -------------------

def _reconstruct_dice_flips(project: str, model_type: str,
                            train: pd.DataFrame, test: pd.DataFrame, model) -> pd.DataFrame:
    """
    Rebuild one candidate row per TP instance from per-instance DiCE CSVs at:
      OUTPUT/<project>/DiCE/<model_type>/<test_idx>.csv
    CSV schema: candidate_id, feature, original_value, cf_value

    Behavior:
      - Prefer a candidate that flips model to 0 (after StandardScaler on train).
      - If none flips, keep the nearest non-flip candidate (fewest changes, then smallest L2).
      - Skip empty/malformed files gracefully.
    Returns a DataFrame indexed by test_idx with full feature vectors.
    """
    out_dir = get_output_dir(project, "DiCE", model_type)
    if not out_dir.exists():
        return pd.DataFrame()

    X_train = train.drop(columns=["target"])
    scaler = StandardScaler().fit(X_train.values)

    # true positives as feature rows (test indices)
    tp_feats = get_true_positives(model, train, test)
    if tp_feats is None or len(tp_feats) == 0:
        return pd.DataFrame()

    rebuilt = {}

    for test_idx in tp_feats.index:
        csv_path = out_dir / f"{test_idx}.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue

        try:
            edits = pd.read_csv(csv_path)
        except EmptyDataError:
            continue
        if edits is None or edits.empty:
            continue

        # sanity: required columns
        required = {"feature", "cf_value"}
        if not required.issubset(set(edits.columns)):
            continue

        # Only use features present in train columns
        edits["feature"] = edits["feature"].astype(str)
        edits = edits[edits["feature"].isin(X_train.columns)]
        if edits.empty:
            continue

        if "candidate_id" not in edits.columns:
            edits["candidate_id"] = 0

        original = test.loc[test_idx, test.columns != "target"].copy()

        best_flip_key = None
        best_flip_x = None

        best_any_key = None
        best_any_x = None

        for cid, grp in edits.groupby("candidate_id"):
            x = original.copy()
            ok = True
            # apply edits
            for _, r in grp.iterrows():
                feat = str(r["feature"])
                # numeric only
                cfv_num = pd.to_numeric(r["cf_value"], errors="coerce")
                if pd.isna(cfv_num):
                    ok = False
                    break
                # cast to train dtype
                if pd.api.types.is_integer_dtype(X_train[feat].dtype):
                    cfv = int(np.round(float(cfv_num)))
                elif pd.api.types.is_float_dtype(X_train[feat].dtype):
                    cfv = float(cfv_num)
                else:
                    ok = False
                    break
                x[feat] = cfv

            if not ok:
                continue

            # metrics for tie-breaking
            num_changed = int((x != original).sum())
            l2 = float(np.linalg.norm((x.values - original.values), ord=2))
            key = (num_changed, l2)

            # Does it flip?
            x_scaled = scaler.transform([x.values])
            try:
                flipped = int(model.predict(x_scaled)[0]) == 0
            except Exception:
                flipped = bool(model.predict(x_scaled)[0] == 0)

            # track best flip, and best regardless
            if best_any_key is None or key < best_any_key:
                best_any_key, best_any_x = key, x
            if flipped and (best_flip_key is None or key < best_flip_key):
                best_flip_key, best_flip_x = key, x

        # prefer a flipping CF; otherwise keep the nearest non-flip
        chosen = best_flip_x if best_flip_x is not None else best_any_x
        if chosen is not None:
            rebuilt[int(test_idx)] = chosen

    return pd.DataFrame(rebuilt).T if rebuilt else pd.DataFrame()

def _load_flipped_or_reconstruct(project: str, model_type: str, train: pd.DataFrame, test: pd.DataFrame, model) -> pd.DataFrame:
    """
    Preferred: EXPERIMENTS/<project>/<model_type>/DiCE_all.csv
    Fallback: reconstruct from per-instance CSVs at OUTPUT/<project>/DiCE/<model_type>/
    """
    flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/DiCE_all.csv"
    if flip_path.exists():
        df = pd.read_csv(flip_path, index_col=0)
        return df.dropna(how="all")
    return _reconstruct_dice_flips(project, model_type, train, test, model)

# ------------------- RQ3 core (DiCE only) -------------------

def flip_feasibility_for_groups(project_groups: list[list[str]],
                                model_type: str,
                                distance: str = "mahalanobis") -> tuple[list[dict], int, int]:
    """
    For each project group (multiple releases of same project):
      - Build a pool of historical developer deltas (test - train) on common indices.
      - For each project in group, for each flipped instance:
          • identify changed features (using DiCE plans),
          • compute normalized Mahalanobis (or cosine) between the change vector
            (flipped - original) and the pool restricted to those features.
    Returns: (list of per-instance metrics dicts, total_flipped_count, cannot_count)
    """
    projects = read_dataset()

    # Build combined pool of historical deltas across all groups
    total_deltas = pd.DataFrame()
    for group in project_groups:
        for project in group:
            if project not in projects:
                continue
            train, test = projects[project]
            exist_idx = train.index.intersection(test.index)
            deltas = test.loc[exist_idx, test.columns != "target"] - \
                     train.loc[exist_idx, train.columns != "target"]
            total_deltas = pd.concat([total_deltas, deltas], axis=0)

    results = []
    totals = 0
    cannots = 0

    for group in project_groups:
        for project in group:
            if project not in projects:
                continue
            train, test = projects[project]
            model = get_model(project, model_type)

            # Load plans
            plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/DiCE/plans_all.json"
            if not plan_path.exists():
                continue
            with open(plan_path, "r") as f:
                plans = json.load(f)

            # Load or reconstruct flipped vectors
            flipped = _load_flipped_or_reconstruct(project, model_type, train, test, model)
            if flipped is None or len(flipped) == 0:
                continue
            flipped = flipped.dropna()

            totals += len(flipped)

            for test_idx in flipped.index:
                if str(test_idx) not in plans:
                    continue

                orig_row = test.loc[test_idx, test.columns != "target"]
                flip_row = flipped.loc[test_idx, :]

                # Changed features restricted to those in the plan for this instance
                changed = {}
                for feat in plans[str(test_idx)].keys():
                    try:
                        if not math.isclose(float(flip_row[feat]), float(orig_row[feat]), rel_tol=1e-7, abs_tol=1e-9):
                            changed[feat] = float(flip_row[feat]) - float(orig_row[feat])
                    except Exception:
                        if flip_row[feat] != orig_row[feat]:
                            # non-numeric fallback
                            changed[feat] = 0.0  # skip non-numerics by setting to 0.0
                if not changed:
                    cannots += 1
                    continue

                changed_features = list(changed.keys())
                pool = total_deltas[changed_features].dropna()
                pool = pool.loc[(pool != 0).any(axis=1)]  # keep rows with any non-zero delta

                if len(pool) == 0:
                    cannots += 1
                    continue

                changed_series = pd.Series(changed, index=changed_features)

                if distance.lower() == "cosine":
                    dists = _cosine_all(pool, changed_series)
                else:
                    # Mahalanobis needs at least a few rows for stable covariance
                    if len(pool) < 5:
                        cannots += 1
                        continue
                    dists = _mahalanobis_all(pool, changed_series)

                if len(dists) == 0:
                    cannots += 1
                    continue

                results.append({
                    "project": project,
                    "test_idx": test_idx,
                    "min": float(np.min(dists)),
                    "max": float(np.max(dists)),
                    "mean": float(np.mean(dists)),
                    "n_pool": int(len(pool)),
                    "n_changed_features": int(len(changed_features)),
                })

    return results, totals, cannots

# ------------------- CLI -------------------

def main():
    ap = ArgumentParser(description="RQ3 feasibility evaluation for DiCE (Mahalanobis/Cosine) without touching existing code.")
    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model list to evaluate")
    ap.add_argument("--distance", type=str, default="mahalanobis",
                    choices=["mahalanobis", "cosine"],
                    help="Distance metric")
    ap.add_argument("--outdir", type=str, default="./evaluations/feasibility",
                    help="Base output folder")
    ap.add_argument("--use_default_groups", action="store_true",
                    help="Use built-in project groups (like original RQ3)")
    args = ap.parse_args()

    project_groups = DEFAULT_PROJECT_GROUPS if args.use_default_groups else [list(read_dataset().keys())]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.outdir}/{args.distance}").mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary_rows = []

    for model_type in tqdm(models, desc="Models", leave=True):
        results, totals, cannots = flip_feasibility_for_groups(project_groups, model_type, args.distance)
        if not results:
            tqdm.write(f"[{model_type}] No results (totals={totals}, cannot={cannots}). Skipping save.")
            continue

        df = pd.DataFrame(results)
        model_code = MODEL_MAP.get(model_type, model_type)
        out_csv = f"{args.outdir}/{args.distance}/{model_code}_DiCE.csv"
        df.to_csv(out_csv, index=False)

        row = {
            "Model": model_type,
            "Explainer": "DiCE",
            "Min": float(df["min"].mean()),
            "Max": float(df["max"].mean()),
            "Mean": float(df["mean"].mean()),
            "Total": totals,
            "Cannot": cannots,
            "Cannot%": (cannots / totals * 100.0) if totals else 0.0,
        }
        summary_rows.append(row)
        tqdm.write(f"[{model_type}] saved -> {out_csv} | totals={totals} cannot={cannots} ({row['Cannot%']:.2f}%)")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f"{args.outdir}/feasibility_{args.distance}_DiCE_summary.csv", index=False)
        print("\n=== Summary (DiCE / RQ3) ===")
        print(summary_df)
    else:
        print("No results to summarize.")

if __name__ == "__main__":
    main()
