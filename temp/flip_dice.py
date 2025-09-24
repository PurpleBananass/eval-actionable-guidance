#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import warnings
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from data_utils import read_dataset, get_model, get_true_positives
from hyparams import PROPOSED_CHANGES, EXPERIMENTS, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)


# ----------------------------- helpers -----------------------------

def _proba_class0(model, X_scaled: np.ndarray) -> np.ndarray:
    """Return P(class=0) robustly across different model types."""
    # 1) predict_proba if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        # proba shape (n_samples, n_classes) is expected; fallbacks covered
        if proba.ndim == 2:
            nclasses = proba.shape[1]
            if nclasses == 2:
                # Find column for class 0 if possible
                try:
                    if hasattr(model, "classes_"):
                        classes = np.asarray(model.classes_)
                        idx0 = int(np.where(classes == 0)[0][0])
                        return proba[:, idx0]
                    else:
                        # Assume column 0 is class 0 (typical sklearn order: [0,1])
                        return proba[:, 0]
                except Exception:
                    # Safe fallback: assume proba[:, 1] is class 1
                    return 1.0 - proba[:, 1]
            elif nclasses >= 2:
                # Multiclass unexpected here; take min prob as a conservative proxy for class 0
                # (should not happen in this pipeline)
                return proba.min(axis=1)
            elif nclasses == 1:
                # Some libs might return only positive class prob
                p1 = proba[:, 0]
                return 1.0 - p1
        elif proba.ndim == 1:
            # 1-D vector -> assume positive class
            p1 = proba
            return 1.0 - p1

    # 2) decision_function fallback -> logistic squashing
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_scaled)
        s = np.clip(s, -50, 50)
        # decision_function positive means class 1 typically; map to prob(1)
        p1 = 1.0 / (1.0 + np.exp(-s))
        if p1.ndim > 1:
            p1 = p1[:, 0]
        return 1.0 - p1

    # 3) Final fallback: predicted labels
    y = model.predict(X_scaled)
    return (y == 0).astype(float)


def _score_candidate(original_vec: np.ndarray, candidate_vec: np.ndarray):
    """Score tuple (num_features_changed, L2); lower is better."""
    num_changed = int(np.sum(candidate_vec != original_vec))
    l2 = float(np.linalg.norm(candidate_vec - original_vec, ord=2))
    return (num_changed, l2)


def _flip_instance_with_plans_topk(
    original_row: pd.Series,
    plans_for_instance: dict[str, list],
    model,
    scaler: StandardScaler,
    max_combinations: int = 10000,
    k: int = 10,
) -> pd.DataFrame:
    """
    Returns up to k flipping candidates (class 0) for one TP.
    Output columns: all feature columns + 'candidate_id'.
    Returns an EMPTY DataFrame if no flip found.
    """
    feat_cols = list(original_row.index)
    original_vec = original_row.values.astype(float)

    # Build per-feature value lists (include original first)
    feature_names, value_lists = [], []
    for f, values in plans_for_instance.items():
        if f in original_row.index:
            feature_names.append(f)
            # ensure float convertibility
            try:
                vals = [float(original_row[f])] + [float(v) for v in values]
            except Exception:
                # skip un-castable features silently
                continue
            value_lists.append(vals)

    if not feature_names:
        return pd.DataFrame(columns=feat_cols + ["candidate_id"])

    def _collect_flips(batch_arr: np.ndarray, flips: list):
        probs0 = _proba_class0(model, scaler.transform(batch_arr))
        idx = np.where(probs0 >= 0.5)[0]
        if idx.size == 0:
            return
        # Sort within this batch by (#changed, L2)
        changed = (batch_arr[idx] != original_vec)
        num_changed = changed.sum(axis=1).astype(int)
        l2 = np.linalg.norm(batch_arr[idx] - original_vec, axis=1)
        order = np.lexsort((l2, num_changed))
        for j in order:
            flips.append(((int(num_changed[j]), float(l2[j])), batch_arr[idx[j]]))

    # Explore candidates
    total = 1
    for vals in value_lists:
        total *= len(vals)

    flips: list[tuple[tuple[int, float], np.ndarray]] = []
    B = 1000  # batch size

    if total <= max_combinations:
        combos = list(product(*value_lists))
        for i in range(0, len(combos), B):
            chunk = combos[i : i + B]
            batch = []
            for combo in chunk:
                vec = original_vec.copy()
                for j, f in enumerate(feature_names):
                    idxf = feat_cols.index(f)
                    vec[idxf] = combo[j]
                batch.append(vec)
            if batch:
                _collect_flips(np.vstack(batch), flips)
            if len(flips) >= k:
                break
    else:
        rng = np.random.default_rng(SEED)
        trials = max_combinations
        for start in range(0, trials, B):
            size = min(B, trials - start)
            batch = []
            for _ in range(size):
                vec = original_vec.copy()
                for j, f in enumerate(feature_names):
                    idxf = feat_cols.index(f)
                    vals = value_lists[j]
                    vec[idxf] = vals[rng.integers(0, len(vals))]
                batch.append(vec)
            if batch:
                _collect_flips(np.vstack(batch), flips)
            if len(flips) >= k:
                break

    if not flips:
        return pd.DataFrame(columns=feat_cols + ["candidate_id"])

    # Global sort, dedupe exact vectors, keep top-k
    flips.sort(key=lambda t: t[0])
    uniq = []
    seen = set()
    for key, vec in flips:
        tup = tuple(np.round(vec, 12))
        if tup not in seen:
            seen.add(tup)
            uniq.append(vec)
        if len(uniq) == k:
            break

    out = pd.DataFrame(uniq, columns=feat_cols)
    out["candidate_id"] = np.arange(len(out), dtype=int)
    return out


# ----------------------------- per-project runner -----------------------------

def run_project(project: str, model_type: str, verbose: bool, max_combinations: int, k: int):
    projects = read_dataset()
    if project not in projects:
        tqdm.write(f"[{project}/{model_type}] not found in dataset. Skipping.")
        return

    train, test = projects[project]
    model = get_model(project, model_type)

    # only work on true positives (DiCE plans are created for TPs)
    tp_df = get_true_positives(model, train, test)
    tp_indices = list(tp_df.index.astype(int))

    # paths
    plans_path = Path(PROPOSED_CHANGES) / project / model_type / "DiCE" / "plans_all.json"
    exp_dir = Path(EXPERIMENTS) / project / model_type
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_csv = exp_dir / "DiCE_all.csv"

    if not plans_path.exists():
        tqdm.write(f"[{project}/{model_type}] plans file missing: {plans_path}")
        return

    # load plans
    with open(plans_path, "r") as f:
        plans = json.load(f)

    # prepare scaler (consistent with your pipeline)
    feat_cols = list(test.columns.drop("target"))
    scaler = StandardScaler().fit(train.drop(columns=["target"]).values)

    # seed output schema
    if exp_csv.exists():
        results_df = pd.read_csv(exp_csv)
        if "test_idx" not in results_df.columns or "candidate_id" not in results_df.columns:
            # migrate old wide format if necessary
            results_df = pd.DataFrame(columns=["test_idx", "candidate_id"] + feat_cols)
            results_df.to_csv(exp_csv, index=False)
        done_tp = set(results_df["test_idx"].unique().astype(int))
        pending = [i for i in tp_indices if str(i) in plans and i not in done_tp]
    else:
        results_df = pd.DataFrame(columns=["test_idx", "candidate_id"] + feat_cols)
        results_df.to_csv(exp_csv, index=False)
        pending = [i for i in tp_indices if str(i) in plans]

    if len(pending) == 0:
        tqdm.write(f"[{project}/{model_type}] nothing new to do.")
        return

    # Light work first: sort by search space size
    workload = []
    for idx in pending:
        space = 1
        for vlist in plans[str(idx)].values():
            space *= (len(vlist) + 1)
        workload.append((idx, space))
    workload.sort(key=lambda x: x[1])
    order = [i for i, _ in workload]

    max_workers = min(8, (os.cpu_count() or 1), len(order))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for idx in order:
            original = test.loc[idx, test.columns != "target"].astype(float)
            futures[pool.submit(
                _flip_instance_with_plans_topk,
                original,
                plans[str(idx)],
                model,
                scaler,
                max_combinations,
                k
            )] = idx

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"{project}/{model_type}", leave=False, disable=not verbose):
            idx = futures[fut]
            try:
                rows = fut.result()  # DataFrame with 0..k rows (feature cols + candidate_id)
                if rows is not None and not rows.empty:
                    rows.insert(0, "test_idx", idx)
                    # append and save incrementally
                    results_df = pd.concat([results_df, rows], axis=0, ignore_index=True)
                    if len(results_df) % 50 == 0:
                        results_df.to_csv(exp_csv, index=False)
            except Exception as e:
                tqdm.write(f"[{project}/{model_type}] error on {idx}: {e}")

    results_df.to_csv(exp_csv, index=False)
    # summary
    flips_total = len(results_df)
    tps = results_df["test_idx"].nunique()
    tqdm.write(f"[OK] {project}/{model_type} -> {flips_total} flip rows across {tps} TPs written to {exp_csv}")


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser()
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated list of model types to run")
    ap.add_argument("--max_combinations", type=int, default=10000,
                    help="Exhaustive if space <= this; else random sample up to this many")
    ap.add_argument("--k", type=int, default=10,
                    help="Max flips to keep per TP")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    projects = read_dataset()
    project_list = list(sorted(projects.keys())) if args.project == "all" else args.project.split()
    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]

    combos = [(p, m) for p in project_list for m in model_types]
    for p, m in tqdm(combos, desc="Projects/Models", leave=True, disable=not args.verbose):
        run_project(p, m, args.verbose, args.max_combinations, args.k)


if __name__ == "__main__":
    main()
