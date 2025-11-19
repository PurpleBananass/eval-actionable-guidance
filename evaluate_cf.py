#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from tabulate import tabulate

from hyparams import PROPOSED_CHANGES, EXPERIMENTS
from data_utils import read_dataset, get_model, get_true_positives
from flip_exp import get_flip_rates


# ----------------------------- small helpers -----------------------------

MODEL_ABBR = {
    "SVM": "SVM",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

EXPLAINER_NAME_MAP = {
    # display name
    "LIME": "LIME",
    "LIME-HPO": "LIME-HPO",
    "TimeLIME": "TimeLIME",
    "SQAPlanner_confidence": "SQAPlanner",
    # New CF generator output file: experiments/{project}/{model}/CF_all.csv
    "CF": "CF",
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

def _cf_flip_rate(project_list: List[str], model_type: str) -> dict:
    """
    Flip rate for CF across projects:
      - Denominator: # of true positives (per your get_true_positives).
      - Numerator  : # of those TPs that appear in CF_all.csv AND predict to class 0.
    """
    ds = read_dataset()
    flipped_tp = 0
    total_tp = 0

    for project in project_list:
        if project not in ds:
            continue

        train, test = ds[project]
        feat_cols = [c for c in test.columns if c != "target"]
        model = get_model(project, model_type)

        # same TP definition as your generation code
        tp_df = get_true_positives(model, train, test)
        if tp_df is None or tp_df.empty:
            continue
        tp_idx = set(tp_df.index.astype(int).tolist())
        total_tp += len(tp_idx)

        # load CF rows
        cf_path = _flip_path(project, model_type, "CF")
        flips = _load_flips_df(cf_path, feat_cols)
        if flips is None or flips.empty:
            continue

        # scaler exactly like your TP computation convention
        scaler = StandardScaler().fit(train.drop("target", axis=1).values)

        flipped_here = set()
        for ti, gi in flips.groupby("test_idx", sort=False):
            ti = int(ti)
            if ti not in tp_idx:
                continue

            # rebuild full candidate row from CF (fill any missing cols from original)
            orig = test.loc[ti, feat_cols].astype(float)
            cand = orig.copy()
            present_cols = [c for c in feat_cols if c in gi.columns]
            if present_cols:
                cand[present_cols] = gi.iloc[0][present_cols].astype(float).values

            # predict like get_true_positives (scale then predict)
            X = scaler.transform([cand.values])
            try:
                pred = int(model.predict(X)[0])
            except Exception:
                # safety fallback if a model returns probs only
                pred = int((getattr(model, "predict_proba")(X)[:, 1] >= 0.5)[0])

            if pred == 0:
                flipped_here.add(ti)

        flipped_tp += len(flipped_here)

    rate = (flipped_tp / total_tp) if total_tp > 0 else 0.0
    return {"Rate": rate, "Flipped": flipped_tp, "Total": total_tp}

def parse_models(arg: str) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return ["RandomForest", "SVM", "XGBoost", "CatBoost", "LightGBM"]
    return [m.strip() for m in arg.replace(",", " ").split() if m.strip()]


def parse_projects(arg: str, ds_keys: List[str]) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return list(sorted(ds_keys))
    return [p.strip() for p in arg.replace(",", " ").split() if p.strip()]


def _flip_path(project: str, model_type: str, explainer: str) -> Path:
    """
    Unified flip file paths.
    - CF → experiments/{project}/{model}/CF_all.csv
    - Others → experiments/{project}/{model}/{EXPLAINER}_all.csv
    """
    if explainer == "CF":
        return Path(EXPERIMENTS) / f"{project}/{model_type}/CF_all.csv"
    return Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"


def _load_flips_df(flip_path: Path, feat_cols: List[str]) -> Optional[pd.DataFrame]:
    """
    Load flips in *long* format:
      - expects a 'test_idx' column; if missing, try reading index_col=0 then reset.
      - keeps 'test_idx', optional 'candidate_id', and any present feature columns.
    """
    if not flip_path.exists() or flip_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(flip_path)
        if "test_idx" not in df.columns:
            df = pd.read_csv(flip_path, index_col=0).reset_index().rename(columns={"index": "test_idx"})
    except Exception:
        return None
    if df is None or df.empty or "test_idx" not in df.columns:
        return None

    keep = ["test_idx"] + ([c for c in ("candidate_id",) if c in df.columns]) + [c for c in feat_cols if c in df.columns]
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
    combinations = list(product(*[data[feature] for feature in data]))
    return pd.DataFrame(combinations, columns=data.keys())


# ----------------------------- distance utils -----------------------------

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
    # vec1, vec2 can be numpy arrays or pandas Series; use underlying values if Series
    v1 = vec1.values if hasattr(vec1, "values") else np.asarray(vec1)
    v2 = vec2.values if hasattr(vec2, "values") else np.asarray(vec2)
    dot_product = np.dot(v1, v2)
    norm_vec1 = np.linalg.norm(v1)
    norm_vec2 = np.linalg.norm(v2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        print(vec1, vec2)  # keep your baseline debug behavior
        return 0
    return dot_product / (norm_vec1 * norm_vec2)


def cosine_all(df: pd.DataFrame, x_series: pd.Series):
    # Align x to df.columns order to ensure correct dot products
    x_vec = x_series.reindex(df.columns).astype(float)
    return [cosine_similarity(x_vec, row.astype(float)) for _, row in df.iterrows()]


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


# ----------------------------- plan similarity / implications -----------------------------

def plan_similarity(project, model_type, explainer):
    """
    For explainers that have plans. CF has no plans → return [].
    """
    if explainer == "CF":
        return []  # no plan-based similarity for CF

    results = {}
    plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
    flip_path = _flip_path(project, model_type, explainer)

    if not flip_path.exists() or not plan_path.exists():
        return []

    with open(plan_path, "r") as f:
        plans = json.load(f)

    experiment = pd.read_csv(flip_path, index_col=0)
    drops = experiment.dropna().index.to_list()
    model = get_model(project, model_type)
    train, test = read_dataset()[project]
    scaler = StandardScaler().fit(train.drop("target", axis=1).values)

    for test_idx in drops:
        if str(test_idx) in plans:
            original = test.loc[test_idx, test.columns != "target"]
            original_scaled = scaler.transform([original])
            pred_o = model.predict(original_scaled)[0]
            row = experiment.loc[[test_idx], :]
            row_scaled = scaler.transform(row.values)
            pred = model.predict(row_scaled)[0]
            assert pred_o == 1 and pred == 0

            plan = {}
            for feature in plans[str(test_idx)]:
                if math.isclose(experiment.loc[test_idx, feature], original[feature], rel_tol=1e-7):
                    continue
                plan[feature] = plans[str(test_idx)][feature]

            if not plan:
                continue

            flipped = experiment.loc[test_idx, [feature for feature in plan]]
            min_changes = pd.Series([plan[f][0] for f in plan], index=flipped.index)
            combi = generate_all_combinations(plan)
            score = normalized_mahalanobis_distance(combi, flipped, min_changes)
            results[test_idx] = {"score": score}
    return results


def implications(project, explainer, model_type, compare_with_cf: bool = False):
    """
    Total scaled |Δ| over changed features (z-scored by train).
    - CF: infer changed features by raw diff (no plans).
    - Others: use plans to pick changed features (legacy).
    If compare_with_cf=True and explainer != "CF":
        returns a dict with paired lists on the SAME indices:
        {
          "explainer": [...],   # scores for `explainer`
          "cf": [...],          # scores for CF on the same test_idx
          "diff": [...],        # explainer - cf
          "paired_count": int
        }
    Otherwise returns the legacy list[float].
    """
    flip_path = _flip_path(project, model_type, explainer)
    if not flip_path.exists():
        return [] if not compare_with_cf else {"explainer": [], "cf": [], "diff": [], "paired_count": 0}

    ds = read_dataset()
    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]
    scaler = StandardScaler().fit(train.drop("target", axis=1).values)

    # --- helpers ---
    def score_from_rows(orig_row: pd.Series, flipped_row: pd.Series, changed_mask: np.ndarray) -> Optional[float]:
        if not np.any(changed_mask):
            return None
        zf = scaler.transform([flipped_row.values])[0]
        zo = scaler.transform([orig_row.values])[0]
        return float(np.abs(zf - zo)[changed_mask].sum())

    def cf_first_rows(cf_df: pd.DataFrame) -> pd.DataFrame:
        # pick first candidate per test_idx
        return cf_df.sort_values(["test_idx"] + ([ "candidate_id"] if "candidate_id" in cf_df.columns else [])).groupby("test_idx", as_index=False).head(1)

    # ---------------- CF-only path (legacy return: list[float]) ----------------
    if explainer == "CF" and not compare_with_cf:
        flips = _load_flips_df(flip_path, feat_cols)
        if flips is None or flips.empty:
            return []
        totals = []
        flips = cf_first_rows(flips)
        for _, row in flips.iterrows():
            t = int(row["test_idx"])
            orig = test.loc[t, feat_cols].astype(float)
            cand = orig.copy()
            present = [c for c in feat_cols if c in row.index]
            cand[present] = row[present].astype(float).values
            changed = ~np.isclose(cand.values, orig.values, rtol=1e-7, atol=1e-7)
            s = score_from_rows(orig, cand, changed)
            if s is not None:
                totals.append(s)
        return totals

    # ---------------- Non-CF legacy path (no comparison) ----------------
    if not compare_with_cf:
        plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
        if not plan_path.exists():
            return []
        with open(plan_path, "r") as f:
            plans = json.load(f)
        flipped_full = pd.read_csv(flip_path, index_col=0).dropna()
        totals = []
        for t in flipped_full.index.astype(int):
            key = str(t)
            if key not in plans:
                continue
            orig = test.loc[t, feat_cols].astype(float)
            flip = flipped_full.loc[t, feat_cols].astype(float)
            changed_feats = [f for f in plans[key] if not math.isclose(flip[f], orig[f], rel_tol=1e-7)]
            if not changed_feats:
                continue
            changed_mask = np.array([c in changed_feats for c in feat_cols], dtype=bool)
            s = score_from_rows(orig, flip, changed_mask)
            if s is not None:
                totals.append(s)
        return totals

    # ---------------- Paired comparison: explainer vs CF on same test_idx ----------------
    if explainer == "CF":
        # comparing CF to itself makes no sense
        return {"explainer": [], "cf": [], "diff": [], "paired_count": 0}

    # Load non-CF (full feature rows + plans)
    plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
    if not plan_path.exists():
        return {"explainer": [], "cf": [], "diff": [], "paired_count": 0}
    with open(plan_path, "r") as f:
        plans = json.load(f)
    flipped_full = pd.read_csv(flip_path, index_col=0).dropna()
    idx_e = set(int(i) for i in flipped_full.index.tolist())

    # Load CF (may have sparse columns; rebuild full rows from present columns)
    cf_path = _flip_path(project, model_type, "CF")
    cf_df = _load_flips_df(cf_path, feat_cols) if cf_path.exists() else None
    if cf_df is None or cf_df.empty:
        return {"explainer": [], "cf": [], "diff": [], "paired_count": 0}
    cf_df = cf_first_rows(cf_df)
    idx_cf = set(int(i) for i in cf_df["test_idx"].tolist())

    # Paired indices
    inter = sorted(idx_e.intersection(idx_cf))
    if not inter:
        return {"explainer": [], "cf": [], "diff": [], "paired_count": 0}

    scores_e, scores_cf = [], []
    for t in inter:
        key = str(t)
        if key not in plans:
            continue
        orig = test.loc[t, feat_cols].astype(float)

        # --- score for explainer (plan-based subset of changed features)
        flip_e = flipped_full.loc[t, feat_cols].astype(float)
        changed_feats_e = [f for f in plans[key] if not math.isclose(flip_e[f], orig[f], rel_tol=1e-7)]
        if not changed_feats_e:
            continue
        changed_mask_e = np.array([c in changed_feats_e for c in feat_cols], dtype=bool)
        se = score_from_rows(orig, flip_e, changed_mask_e)
        if se is None:
            continue

        # --- score for CF (diff-based)
        row_cf = cf_df.loc[cf_df["test_idx"] == t].iloc[0]
        cand_cf = orig.copy()
        present_cf = [c for c in feat_cols if c in row_cf.index]
        cand_cf[present_cf] = row_cf[present_cf].astype(float).values
        changed_cf = ~np.isclose(cand_cf.values, orig.values, rtol=1e-7, atol=1e-7)
        scf = score_from_rows(orig, cand_cf, changed_cf)
        if scf is None:
            continue

        scores_e.append(se)
        scores_cf.append(scf)

    diffs = [a - b for a, b in zip(scores_e, scores_cf)]
    return {"explainer": scores_e, "cf": scores_cf, "diff": diffs, "paired_count": len(scores_e)}


# ----------------------------- RQ3 feasibility -----------------------------

def flip_feasibility(project_list, explainer, model_type, distance="mahalanobis"):
    """
    Return (results, totals, cannots) with printed breakdown.
    CF branch: infer changed features by diff (no plans).
    """
    ds = read_dataset()
    # build historical deltas across the group
    total_deltas = pd.DataFrame()
    for project in project_list:
        train, test = ds[project]
        common = train.index.intersection(test.index)
        deltas = test.loc[common, test.columns != "target"] - \
                 train.loc[common, train.columns != "target"]
        total_deltas = pd.concat([total_deltas, deltas], axis=0)

    totals = 0
    cannots = 0
    skipped_no_flipfile = 0
    skipped_no_plan = 0
    skipped_zero_change = 0
    skipped_empty_nonzero = 0
    skipped_rank_too_low = 0
    written = 0

    results = []

    for project in project_list:
        train, test = ds[project]
        feat_cols = [c for c in test.columns if c != "target"]

        flip_path = _flip_path(project, model_type, explainer)
        flips = _load_flips_df(flip_path, feat_cols)
        if flips is None or flips.empty:
            skipped_no_flipfile += 1
            continue

        totals += len(flips)

        # ----- CF: no plans, changed = all non-zero feature diffs -----
        if explainer == "CF":
            for _, row in flips.iterrows():
                t = int(row["test_idx"])
                original_row = test.loc[t, feat_cols].astype(float)
                flipped_row = row[feat_cols].astype(float)

                diff = flipped_row.values - original_row.values
                changed_mask = ~np.isclose(diff, 0.0, rtol=1e-7, atol=1e-7)
                if not np.any(changed_mask):
                    skipped_zero_change += 1
                    continue

                names = [feat_cols[i] for i in np.where(changed_mask)[0]]
                changed_vec = pd.Series(diff[changed_mask], index=names, dtype=float)

                nonzero = total_deltas[names].dropna()
                nonzero = nonzero.loc[(nonzero != 0).all(axis=1)]

                if distance == "cosine":
                    if len(nonzero) == 0:
                        cannots += 1
                        skipped_empty_nonzero += 1
                        continue
                    dists = cosine_all(nonzero, changed_vec)
                else:
                    # print(len(nonzero), len(names))  # debug
                    if len(nonzero) <= len(names):
                        cannots += 1
                        skipped_rank_too_low += 1
                        continue
                    dists = mahalanobis_all(nonzero, changed_vec)

                if isinstance(dists, (list, np.ndarray)) and len(dists) > 0:
                    results.append({
                        "min": float(np.min(dists)),
                        "max": float(np.max(dists)),
                        "mean": float(np.mean(dists)),
                    })
                    written += 1
                else:
                    cannots += 1
                    skipped_empty_nonzero += 1
            continue  # done with this project for CF

        # ----- Non-CF: use plans (legacy) -----
        plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
        if not plan_path.exists():
            skipped_no_plan += len(flips)
            continue

        with open(plan_path, "r") as f:
            plans = json.load(f)

        for _, row in flips.iterrows():
            t = int(row["test_idx"])
            key = str(t)
            if key not in plans:
                skipped_no_plan += 1
                continue

            original_row = test.loc[t, feat_cols].astype(float)
            flipped_row = row[feat_cols].astype(float)

            changed_features = {
                f: float(flipped_row[f] - original_row[f])
                for f in plans[key]
                if not math.isclose(flipped_row[f], original_row[f], rel_tol=1e-7)
            }
            if not changed_features:
                skipped_zero_change += 1
                continue

            names = list(changed_features.keys())
            changed_vec = pd.Series(changed_features, index=names, dtype=float)

            nonzero = total_deltas[names].dropna()
            nonzero = nonzero.loc[(nonzero != 0).all(axis=1)]

            if distance == "cosine":
                if len(nonzero) == 0:
                    cannots += 1
                    skipped_empty_nonzero += 1
                    continue
                dists = cosine_all(nonzero, changed_vec)
            else:
                if len(nonzero) <= len(names):
                    cannots += 1
                    skipped_rank_too_low += 1
                    continue
                dists = mahalanobis_all(nonzero, changed_vec)

            if isinstance(dists, (list, np.ndarray)) and len(dists) > 0:
                results.append({
                    "min": float(np.min(dists)),
                    "max": float(np.max(dists)),
                    "mean": float(np.mean(dists)),
                })
                written += 1
            else:
                cannots += 1
                skipped_empty_nonzero += 1

    print(
        f"[{model_type} {explainer} {distance}] totals={totals}, written={written}, "
        f"cannot={cannots} | no_flipfile={skipped_no_flipfile}, no_plan={skipped_no_plan}, "
        f"zero_change={skipped_zero_change}, empty_nonzero={skipped_empty_nonzero}, "
        f"rank_too_low={skipped_rank_too_low}"
    )
    return results, totals, cannots


# ----------------------------- CLI -----------------------------

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--rq1", action="store_true", help="Flip rates (via get_flip_rates)")
    ap.add_argument("--rq2", action="store_true", help="Plan similarity (mahalanobis) — skipped for CF")
    ap.add_argument("--rq3", action="store_true", help="Feasibility vs historical deltas")
    ap.add_argument("--implications", action="store_true", help="Scaled total |Δ| over changed features")

    ap.add_argument("--explainer", type=str, default="all",
                    help='Explainers spaced/comma (e.g., "CF LIME"), or "all": LIME LIME-HPO TimeLIME SQAPlanner_confidence CF')
    ap.add_argument("--distance", type=str, default="mahalanobis", choices=["mahalanobis", "cosine"])

    # Toggles
    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,CatBoost,LightGBM",
                    help='Models spaced/comma (e.g., "SVM RF"), or "all"')
    ap.add_argument("--projects", type=str, default="all",
                    help='Projects spaced/comma, or "all"')
    ap.add_argument("--use_default_groups", action="store_true",
                    help="For RQ3, use predefined release groups; else per-project groups")

    args = ap.parse_args()

    # Prepare lists
    ds = read_dataset()
    all_projects = list(sorted(ds.keys()))
    model_types = parse_models(args.models)

    if args.explainer.strip().lower() == "all":
        explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner_confidence", "CF"]
    else:
        explainers = [e for e in args.explainer.replace(",", " ").split() if e.strip()]
    explainers = [e if e in EXPLAINER_NAME_MAP else e for e in explainers]

    project_list = parse_projects(args.projects, all_projects)

    print(f"Evaluating models={model_types}")
    print(f"Explainers={explainers}")
    print(f"Projects=({len(project_list)}) {project_list[:6]}{' ...' if len(project_list) > 6 else ''}\n")

    # ---------------- RQ1 ----------------
    # ---------------- RQ1 ----------------
    if args.rq1:
        table_rows = []
        for model_type in model_types:
            for explainer in explainers:
                disp_name = EXPLAINER_NAME_MAP.get(explainer, explainer)
                try:
                    if explainer == "CF":
                        result = _cf_flip_rate(project_list, model_type)
                    elif explainer == "SQAPlanner_confidence":
                        result = get_flip_rates("SQAPlanner", "confidence", model_type, verbose=False)
                    else:
                        result = get_flip_rates(explainer, None, model_type, verbose=False)
                    table_rows.append([MODEL_ABBR.get(model_type, model_type), disp_name, result["Rate"]])
                except Exception as e:
                    print(f"[rq1] Skip {model_type}/{disp_name}: {e}")

            # per-model mean (across whatever explainers succeeded, including CF)
            abbr = MODEL_ABBR.get(model_type, model_type)
            vals = [row[2] for row in table_rows if row[0] == abbr]
            if vals:
                table_rows.append([abbr, "All", float(np.mean(vals))])

        if table_rows:
            df = pd.DataFrame(table_rows, columns=["Model", "Explainer", "Flip Rate"])
            print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))
            Path("./evaluations").mkdir(parents=True, exist_ok=True)
            df.to_csv("./evaluations/flip_rates.csv", index=False)

    # ---------------- RQ2 ----------------
    if args.rq2:
        Path("./evaluations/similarities").mkdir(parents=True, exist_ok=True)
        for model_type in model_types:
            similarities = pd.DataFrame()
            for explainer in explainers:
                if explainer == "CF":
                    print(f"[rq2] Skipping plan-based similarity for CF.")
                    continue
                for project in project_list:
                    result = plan_similarity(project, model_type, explainer)
                    if not result:
                        continue
                    df = pd.DataFrame(result).T
                    df["project"] = project
                    df["explainer"] = EXPLAINER_NAME_MAP.get(explainer, explainer)
                    df["model"] = MODEL_ABBR.get(model_type, model_type)
                    similarities = pd.concat([similarities, df], axis=0, ignore_index=False)
            if not similarities.empty:
                similarities.to_csv(f"./evaluations/similarities/{MODEL_ABBR.get(model_type, model_type)}.csv")

    # ---------------- RQ3 ----------------
    if args.rq3:
        Path(f"./evaluations/feasibility/{args.distance}").mkdir(parents=True, exist_ok=True)

        groups = DEFAULT_GROUPS if args.use_default_groups else [[p] for p in project_list]

        summary = []
        totals = 0
        cannots = 0

        for model_type in model_types:
            for explainer in explainers:
                all_rows = []
                for g in groups:
                    result, total, cannot = flip_feasibility(g, explainer, model_type, args.distance)
                    totals += total
                    cannots += cannot
                    if result:
                        all_rows.extend(result)

                if not all_rows:
                    continue

                df = pd.DataFrame(all_rows)
                out = f"./evaluations/feasibility/{args.distance}/{MODEL_ABBR.get(model_type, model_type)}_{EXPLAINER_NAME_MAP.get(explainer, explainer)}.csv"
                df.to_csv(out, index=False)
                summary.append([
                    model_type,
                    explainer,
                    df["min"].mean(),
                    df["max"].mean(),
                    df["mean"].mean(),
                ])

        if summary:
            tdf = pd.DataFrame(summary, columns=["Model", "Explainer", "Min", "Max", "Mean"])
            print(tabulate(tdf, headers=tdf.columns, tablefmt="github", showindex=False))
            tdf.to_csv(f"./evaluations/feasibility_{args.distance}.csv", index=False)
            print(f"\nTotal flips seen: {totals} | Cannot: {cannots} ({(cannots / totals * 100.0) if totals else 0:.2f}%)")

    # ---------------- Implications ----------------
    if args.implications:
        Path("./evaluations/abs_changes").mkdir(parents=True, exist_ok=True)

        table_rows = []
        for model_type in model_types:
            for explainer in explainers:
                all_scores = []
                # buffers for paired comparison with CF
                paired_expl, paired_cf = [], []

                for project in project_list:
                    print(f"Processing {project} {model_type} {explainer}")

                    # (A) unpaired, as before
                    vals = implications(project, explainer, model_type)
                    all_scores.extend(vals)

                    # (B) paired vs CF (only for non-CF explainers)
                    if explainer != "CF":
                        pr = implications(project, explainer, model_type, compare_with_cf=True)
                        if isinstance(pr, dict) and pr.get("paired_count", 0) > 0:
                            paired_expl.extend(pr["explainer"])
                            paired_cf.extend(pr["cf"])

                # write per-explainer unpaired CSV (same as you do now)
                if all_scores:
                    out = f"./evaluations/abs_changes/{MODEL_ABBR.get(model_type, model_type)}_{EXPLAINER_NAME_MAP.get(explainer, explainer)}.csv"
                    pd.DataFrame({"score": all_scores}).to_csv(out, index=False)
                    table_rows.append([model_type, EXPLAINER_NAME_MAP.get(explainer, explainer), float(np.mean(all_scores))])

                # write paired-vs-CF CSV for non-CF explainers (new)
                if explainer != "CF" and len(paired_expl) > 0:
                    outp = f"./evaluations/abs_changes/{MODEL_ABBR.get(model_type, model_type)}_{EXPLAINER_NAME_MAP.get(explainer, explainer)}_pairedCF.csv"
                    diffs = (np.array(paired_expl) - np.array(paired_cf)).tolist()
                    pd.DataFrame({
                        "score_explainer": paired_expl,
                        "score_cf": paired_cf,
                        "diff_explainer_minus_cf": diffs
                    }).to_csv(outp, index=False)

            # per-model mean across explainers (exclude the 'Mean' row itself)
            model_means = [r[2] for r in table_rows if r[0] == model_type and r[1] != "Mean"]
            if model_means:
                table_rows.append([model_type, "Mean", float(np.mean(model_means))])

        if table_rows:
            df = pd.DataFrame(table_rows, columns=["Model", "Explainer", "Mean"])
            print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))
            df.to_csv(f"./evaluations/abs_changes_summary.csv", index=False)
