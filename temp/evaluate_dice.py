#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from tabulate import tabulate

from hyparams import PROPOSED_CHANGES, EXPERIMENTS
from data_utils import read_dataset, get_model, get_true_positives


# ----------------------------- Utilities -----------------------------

def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _features_only(df_or_row, label="target"):
    if isinstance(df_or_row, pd.Series):
        return df_or_row[df_or_row.index != label]
    return df_or_row.loc[:, df_or_row.columns != label]


def generate_all_combinations(data: dict[str, list]) -> pd.DataFrame:
    """data: {feature: [values...]}. Returns DataFrame of Cartesian product."""
    if not data:
        return pd.DataFrame()
    cols = list(data.keys())
    combos = list(product(*[data[c] for c in cols]))
    return pd.DataFrame(combos, columns=cols)


def normalized_mahalanobis_distance(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> float:
    """Distance(x,y) normalized by range distance over df space."""
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0.0

    standardized_df = (df - df.mean()) / df.std(ddof=0)

    x_standardized = [
        (x[feature] - df[feature].mean()) / df[feature].std(ddof=0)
        for feature in df.columns
    ]
    y_standardized = [
        (y[feature] - df[feature].mean()) / df[feature].std(ddof=0)
        for feature in df.columns
    ]

    cov_matrix = np.cov(standardized_df.T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix) if cov_matrix.ndim > 0 else (
        np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
    )

    distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)

    min_vector = np.array([df[c].min() for c in df.columns])
    max_vector = np.array([df[c].max() for c in df.columns])

    min_vector_standardized = [
        (min_vector[i] - df[col].mean()) / df[col].std(ddof=0)
        for i, col in enumerate(df.columns)
    ]
    max_vector_standardized = [
        (max_vector[i] - df[col].mean()) / df[col].std(ddof=0)
        for i, col in enumerate(df.columns)
    ]

    max_vector_distance = mahalanobis(min_vector_standardized, max_vector_standardized, inv_cov_matrix)
    if max_vector_distance == 0 or not np.isfinite(max_vector_distance):
        return 0.0

    return distance / max_vector_distance


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = float(np.dot(vec1, vec2))
    n1 = float(np.linalg.norm(vec1))
    n2 = float(np.linalg.norm(vec2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot_product / (n1 * n2)


def cosine_all(df: pd.DataFrame, x: pd.Series) -> list[float]:
    v = x.values.astype(float)
    vals = []
    for _, row in df.iterrows():
        vals.append(cosine_similarity(v, row.values.astype(float)))
    return vals


def mahalanobis_all(df: pd.DataFrame, x: pd.Series) -> list[float]:
    """Return normalized Mahalanobis distances from x to each row in df."""
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return []

    standardized_df = (df - df.mean()) / df.std(ddof=0)
    x_standardized = [
        (x[feature] - df[feature].mean()) / df[feature].std(ddof=0)
        for feature in df.columns
    ]

    cov_matrix = np.cov(standardized_df.T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix) if cov_matrix.ndim > 0 else (
        np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
    )

    min_vector = np.array([df[c].min() for c in df.columns])
    max_vector = np.array([df[c].max() for c in df.columns])

    min_vector_standardized = [
        (min_vector[i] - df[col].mean()) / df[col].std(ddof=0)
        for i, col in enumerate(df.columns)
    ]
    max_vector_standardized = [
        (max_vector[i] - df[col].mean()) / df[col].std(ddof=0)
        for i, col in enumerate(df.columns)
    ]
    max_vector_distance = mahalanobis(min_vector_standardized, max_vector_standardized, inv_cov_matrix)
    if max_vector_distance == 0 or not np.isfinite(max_vector_distance):
        return []

    distances = []
    for _, y in df.iterrows():
        y_standardized = [
            (y[feature] - df[feature].mean()) / df[feature].std(ddof=0)
            for feature in df.columns
        ]
        d = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)
        distances.append(d / max_vector_distance)
    return distances


# ----------------------------- RQ2: DiCE plan similarity -----------------------------

def plan_similarity_dice(project: str, model_type: str) -> dict[int, dict]:
    """
    For each TP (test_idx), consider all DiCE candidates and compute the
    normalized Mahalanobis score between the chosen candidate (changed features)
    and the plan's value-combination space (vs. 'min_changes' anchor).
    Keep the candidate with the **lowest score** per TP.
    """
    results = {}

    plan_path = Path(PROPOSED_CHANGES) / project / model_type / "DiCE" / "plans_all.json"
    flip_path = Path(EXPERIMENTS) / project / model_type / "DiCE_all.csv"
    if not plan_path.exists():
        return results

    flips = _safe_read_csv(flip_path)
    if flips is None or flips.empty:
        return results

    with open(plan_path, "r") as f:
        plans = json.load(f)

    train, test = read_dataset()[project]
    model = get_model(project, model_type)
    scaler = StandardScaler().fit(_features_only(train).values)

    # group by test_idx (multiple candidates)
    if "test_idx" not in flips.columns:
        # legacy wide format -> nothing to do
        return results

    feat_cols = list(_features_only(test).columns)

    for test_idx, g in flips.groupby("test_idx"):
        t = int(test_idx)
        if str(t) not in plans:
            continue

        original = test.loc[t, feat_cols]
        # quick sanity: original should be class-1 (defective)
        pred_o = model.predict(scaler.transform([original]))[0]
        if pred_o != 1:
            pass

        best_score = None
        for _, cand in g.iterrows():
            changed = {}
            for feat, vals in plans[str(t)].items():
                if feat in feat_cols:
                    if not math.isclose(float(cand[feat]), float(original[feat]), rel_tol=1e-9, abs_tol=1e-9):
                        changed[feat] = vals

            if not changed:
                continue

            flipped_vec = pd.Series({f: float(cand[f]) for f in changed.keys()})
            min_changes = pd.Series([changed[f][0] for f in changed.keys()], index=list(changed.keys()), dtype=float)
            combi = generate_all_combinations(changed)
            if combi.shape[1] == 0:
                continue

            score = normalized_mahalanobis_distance(combi, flipped_vec, min_changes)
            if best_score is None or score < best_score:
                best_score = score

        if best_score is not None:
            results[t] = {"score": best_score}

    return results


# ----------------------------- RQ3: Feasibility (DiCE) -----------------------------

def flip_feasibility_dice(project_list: list[str], model_type: str, distance: str = "mahalanobis"):
    """
    RQ3-style feasibility on DiCE long output.
    For each TP, among its candidates choose the one with the **lowest mean distance**
    to the empirical delta set (per release deltas across the group).
    """
    # Build the empirical delta pool across the group
    total_deltas = pd.DataFrame()
    ds = read_dataset()
    for project in project_list:
        train, test = ds[project]
        common = train.index.intersection(test.index)
        deltas = _features_only(test.loc[common]) - _features_only(train.loc[common])
        total_deltas = pd.concat([total_deltas, deltas], axis=0)

    cannot = 0
    results = []
    totals = 0  # total TPs encountered (with flips file present)

    for project in project_list:
        plan_path = Path(PROPOSED_CHANGES) / project / model_type / "DiCE" / "plans_all.json"
        flip_path = Path(EXPERIMENTS) / project / model_type / "DiCE_all.csv"
        if not plan_path.exists():
            continue

        flips = _safe_read_csv(flip_path)
        if flips is None or flips.empty or "test_idx" not in flips.columns:
            continue

        with open(plan_path, "r") as f:
            plans = json.load(f)

        train, test = ds[project]
        feat_cols = list(_features_only(test).columns)

        for test_idx, g in flips.groupby("test_idx"):
            t = int(test_idx)
            if str(t) not in plans:
                continue

            totals += 1
            original_row = test.loc[t, feat_cols]

            # For each candidate: compute change vector and its distances -> choose best (min mean)
            best_summary = None  # (mean, min, max)
            for _, cand in g.iterrows():
                changed_features = {
                    f: float(cand[f]) - float(original_row[f])
                    for f in plans[str(t)].keys()
                    if (f in feat_cols) and not math.isclose(float(cand[f]), float(original_row[f]), rel_tol=1e-9, abs_tol=1e-9)
                }
                if not changed_features:
                    continue

                changed_names = list(changed_features.keys())
                x = pd.Series(changed_features, index=changed_names, dtype=float)

                nonzero = total_deltas[changed_names].dropna()
                nonzero = nonzero.loc[(nonzero != 0).all(axis=1)]

                if distance == "cosine":
                    if len(nonzero) == 0:
                        cannot += 1
                        continue
                    dists = cosine_all(nonzero, x)
                else:  # mahalanobis
                    if len(nonzero) < 5:
                        cannot += 1
                        continue
                    dists = mahalanobis_all(nonzero, x)

                if len(dists) == 0:
                    continue

                summary = (float(np.mean(dists)), float(np.min(dists)), float(np.max(dists)))
                if (best_summary is None) or (summary[0] < best_summary[0]):
                    best_summary = summary

            if best_summary is not None:
                mean_, min_, max_ = best_summary
                results.append({"min": min_, "max": max_, "mean": mean_})

    return results, totals, cannot


# ----------------------------- RQ1: Flip rate (DiCE) -----------------------------

def _count_flips_for_project(project: str, model_type: str) -> tuple[int, int, int, int]:
    """
    Returns (flipped_count, computed_count, plan_count, tp_count) for a project/model.
    - flipped_count: #unique test_idx that have at least one candidate that truly flips to class 0
    - computed_count: #unique test_idx present in DiCE_all.csv
    - plan_count: #TPs with plans (len keys in plans_all.json)
    - tp_count: #TPs according to model (get_true_positives)
    """
    ds = read_dataset()
    train, test = ds[project]
    model = get_model(project, model_type)
    scaler = StandardScaler().fit(_features_only(train).values)

    plans_path = Path(PROPOSED_CHANGES) / project / model_type / "DiCE" / "plans_all.json"
    flip_path = Path(EXPERIMENTS) / project / model_type / "DiCE_all.csv"

    if not plans_path.exists():
        return 0, 0, 0, 0

    with open(plans_path, "r") as f:
        plans = json.load(f)
    plan_count = len(plans)

    # count TPs on test
    tp_df = get_true_positives(model, train, test)
    tp_count = len(tp_df)

    flips = _safe_read_csv(flip_path)
    if flips is None or flips.empty:
        return 0, 0, plan_count, tp_count

    feat_cols = list(_features_only(test).columns)
    # Identify feature columns present in flips file
    if "test_idx" in flips.columns:
        # long format
        computed = flips["test_idx"].astype(int).nunique()
        flipped_set = set()
        # best-effort detect candidate id col
        non_feature_cols = {"test_idx", "candidate_id"}
        flip_feat_cols = [c for c in flips.columns if c not in non_feature_cols]
        # intersect with real features to be safe
        flip_feat_cols = [c for c in flip_feat_cols if c in feat_cols]

        for test_idx, group in flips.groupby("test_idx"):
            t = int(test_idx)
            if len(flip_feat_cols) == 0:
                continue
            X = group[flip_feat_cols].astype(float).values
            # expand to full feature order, filling non-varied with original
            # simplest approach: evaluate on provided subset only if model accepts full set;
            # safer: reinsert into a full vector per row using original row as base
            original = test.loc[t, feat_cols].astype(float)
            full_rows = []
            for r in X:
                v = original.copy()
                v[flip_feat_cols] = r
                full_rows.append(v.values)
            full_X = np.array(full_rows)
            preds = model.predict(scaler.transform(full_X))
            if np.any(preds == 0):
                flipped_set.add(t)
        flipped = len(flipped_set)
        return flipped, computed, plan_count, tp_count
    else:
        # wide format: treat each row as a TP attempt; verify with model
        # try to locate an index column
        computed = len(flips)
        # If a column accidentally named 'Unnamed: 0' exists, drop it
        if "Unnamed: 0" in flips.columns:
            flips = flips.drop(columns=["Unnamed: 0"])
        # Intersect feature columns
        flip_feat_cols = [c for c in flips.columns if c in feat_cols]
        if not flip_feat_cols:
            return 0, computed, plan_count, tp_count

        # We don't know test_idx -> just count rows that really flip
        X = flips[flip_feat_cols].astype(float).values
        preds = model.predict(scaler.transform(X))
        flipped = int(np.sum(preds == 0))
        return flipped, computed, plan_count, tp_count


def flip_rates_dice(model_types: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing flip stats per model and per project.
    Flip% is computed as Flip / #Plan (i.e., among TPs that had plans).
    Also writes ./evaluations/flip_rates_DiCE.csv
    """
    projects = read_dataset()
    rows = []
    for model_type in model_types:
        for project in projects.keys():
            flipped, computed, plan_count, tp_count = _count_flips_for_project(project, model_type)
            rate_plan = (flipped / plan_count) if plan_count > 0 else 0.0
            rows.append([model_type, project, flipped, computed, plan_count, tp_count, rate_plan])

    df = pd.DataFrame(
        rows,
        columns=["Model", "Project", "Flip", "Computed", "#Plan", "#TP", "Flip% (of #Plan)"]
    )

    # Per-model means
    model_means = (
        df.groupby("Model")[["Flip", "Computed", "#Plan", "#TP", "Flip% (of #Plan)"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    # Save
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    df.to_csv("./evaluations/flip_rates_DiCE.csv", index=False)
    return df, model_means


# ----------------------------- Implications (DiCE) -----------------------------

def implications_dice(project: str, model_type: str) -> list[float]:
    """
    Total standardized absolute change per TP.
    For multiple candidates per TP, keep the **smallest** total change.
    """
    plan_path = Path(PROPOSED_CHANGES) / project / model_type / "DiCE" / "plans_all.json"
    flip_path = Path(EXPERIMENTS) / project / model_type / "DiCE_all.csv"
    if not plan_path.exists():
        return []

    flips = _safe_read_csv(flip_path)
    if flips is None or flips.empty or "test_idx" not in flips.columns:
        return []

    with open(plan_path, "r") as f:
        plans = json.load(f)

    train, test = read_dataset()[project]
    scaler = StandardScaler().fit(_features_only(train).values)
    feat_cols = list(_features_only(test).columns)

    totals = []
    for test_idx, g in flips.groupby("test_idx"):
        t = int(test_idx)
        if str(t) not in plans:
            continue

        original = test.loc[t, feat_cols]
        best_total = None

        for _, cand in g.iterrows():
            changed_feats = [f for f in plans[str(t)].keys()
                             if (f in feat_cols) and not math.isclose(float(cand[f]), float(original[f]), rel_tol=1e-9, abs_tol=1e-9)]
            if not changed_feats:
                continue

            scaled_flipped = scaler.transform([cand[feat_cols].values.astype(float)])[0]
            scaled_original = scaler.transform([original.values.astype(float)])[0]
            deltas = pd.Series(scaled_flipped - scaled_original, index=feat_cols).abs()
            total = float(deltas[changed_feats].sum())

            if (best_total is None) or (total < best_total):
                best_total = total

        if best_total is not None:
            totals.append(best_total)

    return totals


# ----------------------------- CLI / Orchestration -----------------------------

DEFAULT_MODEL_ORDER = ["CatBoost", "RandomForest", "SVM", "XGBoost", "LightGBM"]
MODEL_MAP = {"SVM": "SVM", "RandomForest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM", "CatBoost": "CatB"}

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


def main():
    ap = ArgumentParser()
    ap.add_argument("--rq1", action="store_true", help="Flip rate (DiCE)")
    ap.add_argument("--rq2", action="store_true", help="Plan similarity (DiCE)")
    ap.add_argument("--rq3", action="store_true", help="Feasibility vs historical deltas (DiCE)")
    ap.add_argument("--implications", action="store_true", help="Sum abs standardized changes (DiCE)")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODEL_ORDER),
                    help="Comma-separated model types")
    ap.add_argument("--distance", type=str, default="mahalanobis", choices=["mahalanobis", "cosine"])
    ap.add_argument("--use_default_groups", action="store_true",
                    help="Use built-in project release groups for RQ3")
    args = ap.parse_args()

    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    projects = read_dataset()

    if args.rq1:
        df, model_means = flip_rates_dice(model_types)
        # Print per-project and per-model summaries
        print("\nPer-project flip rates (DiCE):")
        print(tabulate(df, headers=df.columns, tablefmt="github", showindex=False))
        print("\nPer-model means (DiCE):")
        print(tabulate(model_means, headers=model_means.columns, tablefmt="github", showindex=False))
        print("\nSaved to ./evaluations/flip_rates_DiCE.csv")

    if args.rq2:
        Path("./evaluations/similarities").mkdir(parents=True, exist_ok=True)
        for model_type in model_types:
            similarities = pd.DataFrame()
            for project in projects:
                res = plan_similarity_dice(project, model_type)
                if not res:
                    continue
                df2 = pd.DataFrame(res).T
                df2["project"] = project
                df2["explainer"] = "DiCE"
                df2["model"] = MODEL_MAP.get(model_type, model_type)
                similarities = pd.concat([similarities, df2], axis=0, ignore_index=False)
            if not similarities.empty:
                out = f"./evaluations/similarities/{MODEL_MAP.get(model_type, model_type)}.csv"
                similarities.to_csv(out)

    if args.rq3:
        Path(f"./evaluations/feasibility/{args.distance}").mkdir(parents=True, exist_ok=True)
        summary = []
        for model_type in model_types:
            results_all = []
            totals = 0
            cannots = 0

            if args.use_default_groups:
                groups = DEFAULT_GROUPS
            else:
                groups = [[p] for p in projects.keys()]

            for group in groups:
                res, tot, cannot = flip_feasibility_dice(group, model_type, distance=args.distance)
                totals += tot
                cannots += cannot
                if res:
                    results_all.extend(res)

            if results_all:
                df3 = pd.DataFrame(results_all)
                out = f"./evaluations/feasibility/{args.distance}/{MODEL_MAP.get(model_type, model_type)}_DiCE.csv"
                df3.to_csv(out, index=False)
                summary.append([model_type, "DiCE", df3["min"].mean(), df3["max"].mean(), df3["mean"].mean()])
            print(f"[{model_type}] totals={totals}, cannot={cannots}")

        if summary:
            print(tabulate(summary, headers=["Model", "Explainer", "Min", "Max", "Mean"]))
            pd.DataFrame(summary, columns=["Model", "Explainer", "Min", "Max", "Mean"]).to_csv(
                f"./evaluations/feasibility_{args.distance}_DiCE.csv", index=False
            )
        else:
            print("No results to summarize.")

    if args.implications:
        Path("./evaluations/abs_changes").mkdir(parents=True, exist_ok=True)
        table = []
        for model_type in model_types:
            vals = []
            for project in projects:
                v = implications_dice(project, model_type)
                vals.extend(v)
            if vals:
                s = pd.Series(vals)
                out = f"./evaluations/abs_changes/{MODEL_MAP.get(model_type, model_type)}_DiCE.csv"
                s.to_csv(out, index=False, header=False)
                table.append([model_type, "DiCE", s.mean()])
        if table:
            print(tabulate(table, headers=["Model", "Explainer", "Mean"]))
            pd.DataFrame(table, columns=["Model", "Explainer", "Mean"]).to_csv(
                "./evaluations/abs_changes_DiCE.csv", index=False
            )


if __name__ == "__main__":
    main()
