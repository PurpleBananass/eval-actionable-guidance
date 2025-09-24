#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build plans_all.json from DiCE CSV outputs for **all model combinations**.

Modes:
  - min : use only <idx>.csv (the “best” CF saved by your DiCE run)
          → writes to ./plans/<project>/<model_type>/DiCE/plans_all.json
  - all : union of <idx>.csv and <idx>_*.csv (all kept CFs per instance)
          → writes to ./plans/<project>/<model_type>/DiCE_all/plans_all.json

It iterates over:
  - all projects in read_dataset()
  - and either a user-specified list of --model_types OR
    auto-discovers available models per project from hyparams.MODELS.

The output format exactly matches your existing pipeline’s plans_all.json.
"""

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import read_dataset, get_model, get_true_positives, get_output_dir
from hyparams import MODELS  # for auto-discovery of available models on disk


# ---------------------- helpers (mirroring your plan_explanations.py) ----------------------

def perturb(low, high, current, values, dtype):
    """Pick candidate values in [low, high] from training distinct values;
       reduce to ≤10 reps by chunking; sort by |v - current|."""
    dtype_str = str(dtype)
    if dtype_str == "int64":
        perturbations = [val for val in values if low <= val <= high]
    elif dtype_str == "float64":
        perturbations = []
        candidates = [val for val in values if low <= val <= high]
        if len(candidates) == 0:
            return []
        last = candidates[0]
        perturbations.append(last)
        for candidate in candidates[1:]:
            if round(last, 2) != round(candidate, 2):
                perturbations.append(candidate)
                last = candidate
    else:
        # default numeric-ish
        perturbations = [val for val in values if low <= val <= high]

    if len(perturbations) > 10:
        groups = np.array_split(np.array(perturbations, dtype=float), 10)
        perturbations = [float(np.median(g)) for g in groups]

    try:
        if current in perturbations:
            perturbations.remove(current)
    except Exception:
        pass

    return sorted(perturbations, key=lambda x: abs(float(x) - float(current)))


def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    """
    From a rule like 'a < feature <= b' decide which side to explore based on sign of importance:
      - importance > 0 (CF decreased feature): explore [min_val, a]
      - importance < 0 (CF increased feature): explore [b, max_val]
    Also supports 'feature > a' and 'feature <= b'.
    """
    num = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    # a < f <= b
    m = re.search(fr"{num}\s*<\s*{re.escape(feature)}\s*<=\s*{num}", rule_str)
    if m:
        a, b = map(float, m.groups())
        return [min_val, feature, a] if importance > 0 else [b, feature, max_val]
    # f > a
    m = re.search(fr"{re.escape(feature)}\s*>\s*{num}", rule_str)
    if m:
        a = float(m.group(1))
        return [min_val, feature, a]
    # f <= b
    m = re.search(fr"{re.escape(feature)}\s*<=\s*{num}", rule_str)
    if m:
        b = float(m.group(1))
        return [b, feature, max_val]
    return None


def _convert_int64(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


# ---------------------- model discovery ----------------------

def discover_models_for_project(project_name: str) -> list[str]:
    """
    Inspect hyparams.MODELS/<project_name> and return available model type names.
    - *.joblib  -> stem is the model_type (e.g., RandomForest.joblib -> "RandomForest")
    - XGBoost.xgb -> model_type "XGBoost"
    """
    base = Path(MODELS) / project_name
    types = set()
    if not base.exists():
        return []
    for p in base.glob("*.joblib"):
        types.add(p.stem)
    if (base / "XGBoost.xgb").exists():
        types.add("XGBoost")
    return sorted(types)


# ---------------------- core plan builder ----------------------

def build_plans_for_project_and_model(project_name: str,
                                      model_type: str,
                                      mode: str,
                                      verbose: bool):
    """
    Read DiCE CSVs and write plans_all.json for one (project, model_type).
    mode='min' -> only <idx>.csv ; mode='all' -> <idx>.csv + <idx>_*.csv
    """
    projects = read_dataset()
    if project_name not in projects:
        return False, f"Unknown project: {project_name}"

    train, test = projects[project_name]

    # Load model; if model isn't present, skip gracefully
    try:
        model = get_model(project_name, model_type)
    except Exception as e:
        return False, f"get_model failed for {project_name}/{model_type}: {e}"

    # Where DiCE CSVs live (created by your run_dice.py)
    dice_out_dir = get_output_dir(project_name, "DiCE", model_type)
    # If no DiCE outputs directory yet, produce empty plans and move on
    if not dice_out_dir.exists():
        plans_dir_name = "DiCE" if mode == "min" else "DiCE_all"
        plans_dir = Path("./plans") / project_name / model_type / plans_dir_name
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / "plans_all.json").write_text("{}")
        return True, f"{project_name}/{model_type}: no DiCE outputs; wrote empty plans."

    # Where to write plans
    plans_dir_name = "DiCE" if mode == "min" else "DiCE_all"
    plans_dir = Path("./plans") / project_name / model_type / plans_dir_name
    plans_dir.mkdir(parents=True, exist_ok=True)
    plans_json_path = plans_dir / "plans_all.json"

    # Train stats
    train_min = train.min(numeric_only=True)
    train_max = train.max(numeric_only=True)
    feature_values = {
        feat: sorted(set(train[feat].dropna().values))
        for feat in train.columns if feat != "target"
    }
    dtypes = train.dtypes

    # True positives indices to iterate
    try:
        true_positives = get_true_positives(model, train, test)
    except Exception as e:
        # If scoring fails, emit empty and continue
        plans_json_path.write_text("{}")
        return False, f"{project_name}/{model_type}: get_true_positives failed: {e}"

    all_plans = {}
    idx_iter = true_positives.index
    if len(idx_iter) == 0:
        plans_json_path.write_text("{}")
        return True, f"{project_name}/{model_type}: no TPs; wrote empty plans."

    for test_idx in tqdm(idx_iter, desc=f"{project_name}/{model_type} ({mode})", disable=not verbose, leave=False):
        # the test instance (for current values)
        try:
            test_instance = test.loc[test_idx]
        except KeyError:
            # Index mismatch—skip entry but keep key with empty dict
            all_plans[int(test_idx)] = {}
            continue

        # Gather the relevant DiCE CSVs
        files = []
        base = dice_out_dir / f"{test_idx}.csv"
        if base.exists():
            files.append(base)
        if mode == "all":
            files.extend(sorted(dice_out_dir.glob(f"{test_idx}_*.csv")))

        if not files:
            all_plans[int(test_idx)] = {}
            continue

        feature_to_values = {}  # feature -> set of candidate numeric values

        for fpath in files:
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue

            for _, row in df.iterrows():
                feature = str(row.get("feature", ""))
                if not feature or feature not in test.columns:
                    continue
                try:
                    importance = float(row.get("importance", 0.0))
                except Exception:
                    continue
                rule_str = str(row.get("rule", ""))

                # derive [left, feature, right] based on rule and importance sign
                proposed = flip_feature_range(
                    feature=feature,
                    min_val=float(train_min.get(feature, np.nan)),
                    max_val=float(train_max.get(feature, np.nan)),
                    importance=importance,
                    rule_str=rule_str,
                )
                if not proposed:
                    continue

                left, _, right = proposed
                if not np.isfinite(left) or not np.isfinite(right) or left > right:
                    continue

                # build perturbations from train discrete values
                cur_val = test_instance[feature]
                dtype = dtypes[feature]
                vals = feature_values.get(feature, [])
                perturbs = perturb(left, right, cur_val, vals, dtype)
                if not perturbs:
                    continue

                feature_to_values.setdefault(feature, set()).update(map(float, perturbs))

        # sort per feature by |v - original|
        per_feat_sorted = {}
        for feat, vals in feature_to_values.items():
            try:
                cur = float(test.loc[test_idx, feat])
            except Exception:
                continue
            per_feat_sorted[feat] = sorted({float(v) for v in vals}, key=lambda v: abs(v - cur))

        all_plans[int(test_idx)] = per_feat_sorted

    with open(plans_json_path, "w") as f:
        json.dump(all_plans, f, indent=4, default=_convert_int64)

    return True, f"[OK] {project_name}/{model_type} ({mode}) -> {plans_json_path}"


# ---------------------- CLI ----------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="all",
                        help="Project name or 'all'")
    parser.add_argument("--model_types", type=str, default=None,
                        help="Comma-separated list of model types, e.g. 'RandomForest,SVM,XGBoost,LightGBM,CatBoost'. "
                             "If omitted, models are auto-discovered per project from hyparams.MODELS.")
    parser.add_argument("--mode", type=str, default="min", choices=["min", "all"],
                        help="'min' uses <idx>.csv; 'all' unions <idx>.csv and <idx>_*.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split()

    # Resolve model list per project
    explicit_models = None
    if args.model_types:
        explicit_models = [m.strip() for m in args.model_types.split(",") if m.strip()]

    # Progress over all (project, model) combos
    combos = []
    for project in project_list:
        if explicit_models is not None:
            models = explicit_models
        else:
            models = discover_models_for_project(project)
            if not models:
                models = []  # no models found; we'll still show a message
        for m in models:
            combos.append((project, m))

    if not combos:
        tqdm.write("No (project, model) combinations found. Check hyparams.MODELS or --model_types.")
        return

    pbar = tqdm(combos, desc="Project/Model combos", leave=True)
    for project, model_type in pbar:
        ok, msg = build_plans_for_project_and_model(project, model_type, args.mode, args.verbose)
        if args.verbose and msg:
            tqdm.write(msg)

    tqdm.write("Done.")


if __name__ == "__main__":
    main()
