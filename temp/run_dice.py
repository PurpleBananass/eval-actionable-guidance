#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DiCE (random) → pick a simple best CF (no Mahalanobis) → per-instance CSVs

What this does:
- Loads models via your data_utils.get_model (joblib for most, .xgb for XGBoost).
- Scales features with StandardScaler fitted on TRAIN (to match get_true_positives).
- For each TRUE POSITIVE in TEST, generates DiCE counterfactuals (method="random").
- Keeps only CFs that change ≤ max_varied_features features.
- Selects the single "best" CF by (fewest changed features, then smallest L2 distance).
- Writes ONE CSV per instance at get_output_dir(project, "DiCE", model_type)/<test_idx>.csv
  with columns: feature,value,importance,min,max,rule,importance_ratio
- Skips instances whose CSV already exists.
- Shows progress with tqdm.

Run example:
  python run_dice_random_csv.py --project all --model_type RandomForest --total_cfs 20 --max_varied_features 5
"""

import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import dice_ml
from dice_ml import Dice

# Your utils
from data_utils import read_dataset, get_model, get_output_dir, get_true_positives

# Project constants
from hyparams import SEED

np.random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------- Model wrapper that applies the train-fitted scaler ----------
class ScaledModel:
    """Wrap base model; apply StandardScaler (fit on TRAIN) before inference (matches get_true_positives)."""

    def __init__(self, base_model, train_X: pd.DataFrame):
        self.base = base_model
        self.scaler = StandardScaler().fit(train_X.values)

    def predict(self, X):
        Xs = self.scaler.transform(np.asarray(X))
        return self.base.predict(Xs)

    def predict_proba(self, X):
        if hasattr(self.base, "predict_proba"):
            Xs = self.scaler.transform(np.asarray(X))
            return self.base.predict_proba(Xs)
        raise AttributeError("Base model has no predict_proba")

    def decision_function(self, X):
        if hasattr(self.base, "decision_function"):
            Xs = self.scaler.transform(np.asarray(X))
            return self.base.decision_function(Xs)
        raise AttributeError("Base model has no decision_function")


# ---------- Helpers ----------
def _detect_int_features(df: pd.DataFrame) -> set[str]:
    X = df.drop(columns=["target"])
    return {c for c, dt in X.dtypes.items() if np.issubdtype(dt, np.integer)}

def _round_int_cols(s: pd.Series, int_cols: set[str]) -> pd.Series:
    """Round integer-typed features safely (coerce to numeric first)."""
    s2 = s.copy()
    for c in int_cols:
        if c in s2.index:
            v = pd.to_numeric(pd.Series([s2[c]]), errors="coerce").iloc[0]
            s2[c] = float(np.round(v)) if pd.notna(v) else np.nan
    return s2

def _num_changed(a: pd.Series, b: pd.Series) -> int:
    """
    Count changed features: numeric columns use isclose; non-numerics use equality.
    NaN==NaN counts as equal.
    """
    A = pd.DataFrame({"a": a, "b": b})
    a_num = pd.to_numeric(A["a"], errors="coerce")
    b_num = pd.to_numeric(A["b"], errors="coerce")
    both_num = a_num.notna() & b_num.notna()

    eq = pd.Series(False, index=A.index)

    # numeric compare (tolerant)
    eq.loc[both_num] = np.isclose(a_num[both_num].to_numpy(),
                                  b_num[both_num].to_numpy(),
                                  atol=1e-9)

    # non-numeric compare (or when one side is non-numeric)
    other = ~both_num
    if other.any():
        a_ = A.loc[other, "a"]
        b_ = A.loc[other, "b"]
        eq.loc[other] = (a_.isna() & b_.isna()) | (a_.astype(str) == b_.astype(str))

    return int((~eq).sum())

def _l2(a: pd.Series, b: pd.Series) -> float:
    """
    Euclidean distance over NUMERIC columns only; non-numerics ignored.
    """
    A = pd.DataFrame({"a": a, "b": b})
    a_num = pd.to_numeric(A["a"], errors="coerce")
    b_num = pd.to_numeric(A["b"], errors="coerce")
    diff = (a_num - b_num).fillna(0.0)
    return float(np.sqrt(np.sum(np.square(diff.to_numpy()))))



# ---------- Main per-project runner ----------
def run_project(project: str,
                model_type: str,
                total_cfs: int = 20,
                max_varied_features: int = 5,
                actionable: bool = False,
                verbose: bool = True,
                num_cfs: int = 10):   # <- NEW

    # Load data & model
    projects = read_dataset()
    if project not in projects:
        raise ValueError(f"Unknown project: {project}")
    train_df, test_df = projects[project]

    base_model = get_model(project, model_type)
    train_X = train_df.drop(columns=["target"])
    scaled_model = ScaledModel(base_model, train_X)

    # Output directory for per-instance files
    out_dir = get_output_dir(project, "DiCE", model_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    # True positives in TEST (features only, already scaled internally in get_true_positives)
    tp_df = get_true_positives(base_model, train_df, test_df)
    if len(tp_df) == 0:
        if verbose:
            print(f"[{project}/{model_type}] No true positives.")
        return

    # DiCE setup (use raw train_df; model wrapper handles scaling)
    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=list(train_X.columns),
        outcome_name="target",
    )
    dice_model = dice_ml.Model(model=scaled_model, backend="sklearn")
    explainer = Dice(dice_data, dice_model, method="random")

    # Actionable feature set
    inactionable = {
        "MAJOR_COMMIT", "MAJOR_LINE", "MINOR_COMMIT", "MINOR_LINE",
        "OWN_COMMIT", "OWN_LINE", "ADEV", "Added_lines", "Del_lines",
    }
    feat_order = list(train_X.columns)
    actionable_features = [f for f in feat_order if f not in inactionable]

    # Train bounds (for min/max columns)
    train_min = train_df.min(numeric_only=True)
    train_max = train_df.max(numeric_only=True)

    # Integer features (so we can round both original and candidates for comparison / rule)
    int_features = _detect_int_features(train_df)

    # Iterate TPs with progress
    pbar = tqdm(tp_df.index, desc=f"{project}/{model_type} instances", leave=False)
    for idx in pbar:
        out_file = out_dir / f"{idx}.csv"
        if out_file.exists():
            pbar.set_postfix_str("skip")
            continue

        x0_raw = test_df.loc[idx].drop(labels=["target"])
        x0 = _round_int_cols(x0_raw, int_features)
        query_df = pd.DataFrame([x0.values], columns=feat_order)

        # Generate DiCE CFs
        try:
            if actionable:
                cf_obj = explainer.generate_counterfactuals(
                    query_instances=query_df,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    random_seed=SEED,
                    features_to_vary=actionable_features,
                )
            else:
                cf_obj = explainer.generate_counterfactuals(
                    query_instances=query_df,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    random_seed=SEED,
                )
        except Exception as e:
            out_file.write_text("")
            if verbose:
                tqdm.write(f"[{project}/{model_type}] idx={idx} DiCE error: {e}")
            continue

        cf_df = None
        if cf_obj.cf_examples_list and cf_obj.cf_examples_list[0].final_cfs_df is not None:
            cf_df = cf_obj.cf_examples_list[0].final_cfs_df[feat_order].copy()

        if cf_df is None or len(cf_df) == 0:
            out_file.write_text("")
            continue

        # Round integer-typed features in candidates too (robust to strings/None)
        for c in int_features:
            if c in cf_df.columns:
                cf_df[c] = pd.to_numeric(cf_df[c], errors="coerce").round().astype(float)

        # Keep CFs that change ≤ max_varied_features
        def _safe_num_changed(a: pd.Series, b: pd.Series) -> int:
            A = pd.DataFrame({"a": a, "b": b})
            a_num = pd.to_numeric(A["a"], errors="coerce")
            b_num = pd.to_numeric(A["b"], errors="coerce")
            both_num = a_num.notna() & b_num.notna()
            eq = pd.Series(False, index=A.index)
            # numeric tolerant compare
            eq.loc[both_num] = np.isclose(a_num[both_num].to_numpy(),
                                          b_num[both_num].to_numpy(),
                                          atol=1e-9)
            # non-numeric / mixed compare
            other = ~both_num
            if other.any():
                a_ = A.loc[other, "a"]
                b_ = A.loc[other, "b"]
                eq.loc[other] = (a_.isna() & b_.isna()) | (a_.astype(str) == b_.astype(str))
            return int((~eq).sum())

        def _safe_l2(a: pd.Series, b: pd.Series) -> float:
            A = pd.DataFrame({"a": a, "b": b})
            a_num = pd.to_numeric(A["a"], errors="coerce")
            b_num = pd.to_numeric(A["b"], errors="coerce")
            diff = (a_num - b_num).fillna(0.0)
            return float(np.sqrt(np.sum(np.square(diff.to_numpy()))))

        # Rank CFs by (#changed, L2); take top num_cfs
        change_counts = cf_df.apply(lambda r: _safe_num_changed(x0, r), axis=1)
        cf_df = cf_df.loc[change_counts <= max_varied_features]
        if len(cf_df) == 0:
            out_file.write_text("")
            continue

        l2 = cf_df.apply(lambda r: _safe_l2(x0, r), axis=1)
        rank = pd.DataFrame({"changes": change_counts.loc[cf_df.index], "l2": l2})
        ranked = cf_df.loc[rank.sort_values(["changes", "l2"]).index]
        selected = ranked.head(num_cfs)

        # If ALL outputs already exist, skip
        needed = [out_dir / f"{idx}_{k}.csv" for k in range(len(selected))]
        if all(p.exists() for p in needed) and (out_dir / f"{idx}.csv").exists():
            pbar.set_postfix_str("skip")
            continue

        # Helper to write one CF into one CSV (same schema as your original)
        def _write_one_cf_csv(target_file: Path, original: pd.Series, cf_row: pd.Series):
            rows = []
            eps = 1e-12
            for f in feat_order:
                v0 = pd.to_numeric(pd.Series([original[f]]), errors="coerce").iloc[0]
                v1 = pd.to_numeric(pd.Series([cf_row[f]]), errors="coerce").iloc[0]
                v0f = float(v0) if pd.notna(v0) else np.nan
                v1f = float(v1) if pd.notna(v1) else np.nan
                if (pd.isna(v0f) and pd.isna(v1f)) or (pd.notna(v0f) and pd.notna(v1f) and np.isclose(v0f, v1f, atol=1e-9)):
                    continue

                a, b = (v1f, v0f) if v1f < v0f else (v0f, v1f)
                rule = f"{a} < {f} <= {b}"

                f_min = float(train_min.get(f, np.nan))
                f_max = float(train_max.get(f, np.nan))
                rng = max(f_max - f_min, eps)
                ratio = abs(v1f - v0f) / rng

                # importance: signed normalized delta (v1 < v0 => positive, else negative)
                imp = ratio if v1f < v0f else -ratio

                rows.append({
                    "feature": f,
                    "value": v0f,
                    "importance": imp,
                    "min": f_min,
                    "max": f_max,
                    "rule": rule,
                    "importance_ratio": ratio,
                })

            if rows:
                pd.DataFrame(rows, columns=[
                    "feature", "value", "importance", "min", "max", "rule", "importance_ratio"
                ]).to_csv(target_file, index=False)
            else:
                target_file.write_text("")

        # Write best to <idx>.csv (for compatibility), and all selected to <idx>_k.csv
        best_cf = selected.iloc[0]
        _write_one_cf_csv(out_file, x0, best_cf)

        for k, (_, cf_row) in enumerate(selected.iterrows()):
            _write_one_cf_csv(out_dir / f"{idx}_{k}.csv", x0, cf_row)


def main():
    ap = ArgumentParser()
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--total_cfs", type=int, default=20,
                    help="How many CFs to ask DiCE for per instance")
    ap.add_argument("--max_varied_features", type=int, default=5,
                    help="Max number of features allowed to change")
    ap.add_argument("--actionable", action="store_true",
                    help="Restrict to actionable features only")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--num_cfs", type=int, default=10,
                    help="How many CF CSVs to save per instance")
    args = ap.parse_args()

    # Quick, fixed list of models to sweep
    MODEL_TYPES = ["RandomForest", "SVM", "XGBoost", "LightGBM", "CatBoost"]

    projects = read_dataset()
    project_list = list(sorted(projects.keys())) if args.project == "all" else args.project.split()

    tqdm.write(f"| Project / Model | #files written |")
    tqdm.write(f"| ---------------- | -------------- |")

    combos = [(p, m) for p in project_list for m in MODEL_TYPES]

    for project, model_type in tqdm(combos, desc="Projects × Models", leave=True):
        run_project(
            project=project,
            model_type=model_type,
            total_cfs=args.total_cfs,
            max_varied_features=args.max_varied_features,
            actionable=args.actionable,
            verbose=args.verbose,
            num_cfs=args.num_cfs,
        )
        out_dir = get_output_dir(project, "DiCE", model_type)
        n_files = len(list(out_dir.glob("*.csv")))
        tqdm.write(f"| {project} / {model_type} | {n_files} |")



if __name__ == "__main__":
    main()
