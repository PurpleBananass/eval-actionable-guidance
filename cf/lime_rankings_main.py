#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIME-based feature-importance rankings, wired to your loaders.

- Uses: read_dataset(), get_model(), get_true_positives() from your codebase
- Works with any classifier; wraps models that lack predict_proba
- Per-instance rankings (+ raw value snapshot) and an aggregate summary
- Optionally run only on true positives per project/model

Outputs (per project/model):
  ./evaluations/lime/{project}/{model}/lime_per_instance.csv
  ./evaluations/lime/{project}/{model}/lime_summary.csv
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Your helpers
from data_utils import read_dataset, get_model, get_true_positives
try:
    from hyparams import SEED
except Exception:
    SEED = 42

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception as e:
    raise RuntimeError("Please install LIME: pip install lime") from e


# ----------------------------- model wrapper -----------------------------

class ProbModelWrapper:
    """
    Ensures a predict_proba(X) -> [N, 2] interface.

    - If the base model has predict_proba, we use it.
    - Else if it has decision_function, we apply a logistic link.
    - Else we fall back to predict() and make a degenerate probability.
    """

    def __init__(self, base_model):
        self.m = base_model

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.m, "predict_proba"):
            p = self.m.predict_proba(X)
            if p.ndim == 1:
                p1 = p
                return np.stack([1 - p1, p1], axis=1)
            if p.shape[1] == 2:
                return p
            # Multi-class -> collapse to a binary [0, 1] on the argmax
            p1 = np.max(p, axis=1)
            return np.stack([1 - p1, p1], axis=1)

        if hasattr(self.m, "decision_function"):
            s = self.m.decision_function(X)
            if np.ndim(s) > 1:
                s = s[:, 0]
            p1 = self._sigmoid(s)
            return np.stack([1 - p1, p1], axis=1)

        # Last resort: use predict as hard labels
        y = self.m.predict(X)
        p1 = (y == 1).astype(float)
        return np.stack([1 - p1, p1], axis=1)


# ----------------------------- LIME config & core -----------------------------

@dataclass
class LimeConfig:
    discretizer: str = "entropy"           # 'entropy' | 'quartile' | None
    feature_selection: str = "lasso_path"  # 'auto'|'lasso_path'|...
    num_samples: int = 5000                # samples for LIME's local surrogate
    top_k: int = 10                        # keep top-K features per instance
    seed: int = SEED


def _build_explainer(X_train_scaled: np.ndarray,
                     y_train: np.ndarray,
                     feature_names: List[str],
                     cfg: LimeConfig) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=y_train.astype(int),
        feature_names=feature_names,
        discretizer=cfg.discretizer,
        feature_selection=cfg.feature_selection,
        random_state=cfg.seed,
        mode="classification",
        class_names=[0, 1],
    )


def _explain_one(explainer: LimeTabularExplainer,
                 scaler: StandardScaler,
                 model: ProbModelWrapper,
                 x_row: np.ndarray,
                 feature_names: List[str],
                 *,
                 label_to_explain: int,
                 cfg: LimeConfig) -> pd.DataFrame:
    """
    Explain a single instance. Returns columns:
      ['feature','weight','abs_weight','rank', 'value']
    """
    x_scaled = scaler.transform(x_row.reshape(1, -1))[0]

    exp = explainer.explain_instance(
        data_row=x_scaled,
        predict_fn=lambda X_scaled: model.predict_proba(scaler.inverse_transform(X_scaled)),
        labels=[label_to_explain],
        num_samples=cfg.num_samples,
        num_features=len(feature_names),
    )

    fmap = dict(exp.as_map()[label_to_explain])  # {feature_index: weight}
    rows = []
    for j, fname in enumerate(feature_names):
        w = float(fmap.get(j, 0.0))
        if w != 0.0:
            rows.append((fname, w))

    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    rows = rows[:cfg.top_k]

    data = []
    for rnk, (fname, w) in enumerate(rows, start=1):
        j = feature_names.index(fname)
        data.append({
            "feature": fname,
            "weight": float(w),
            "abs_weight": float(abs(w)),
            "rank": int(rnk),
            "value": float(x_row[j]),
        })
    return pd.DataFrame(data, columns=["feature", "weight", "abs_weight", "rank", "value"])


def lime_rankings_for_split(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            model,
                            *,
                            only_true_positives: bool,
                            top_k: int,
                            num_samples: int,
                            label_mode: str = "opposite",   # 'opposite' | 'pred' | 'fixed0' | 'fixed1'
                            max_eval: Optional[int] = None,
                            seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute LIME per-instance rankings + aggregate summary on the requested test subset.

    label_mode:
      - 'pred'     : explain contribution toward the predicted class
      - 'opposite' : explain contribution toward the opposite of predicted class
      - 'fixed0'   : always explain class 0
      - 'fixed1'   : always explain class 1
    """
    feat_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feat_cols].astype(float)
    y_train = train_df["target"].astype(int)

    # Choose evaluation instances
    if only_true_positives:
        tp_df = get_true_positives(model, train_df, test_df)
        X_eval = test_df.loc[tp_df.index, feat_cols].astype(float)
        y_eval = test_df.loc[tp_df.index, "target"].astype(int)
        eval_index = X_eval.index
    else:
        X_eval = test_df[feat_cols].astype(float)
        y_eval = test_df["target"].astype(int)
        eval_index = X_eval.index

    if max_eval is not None and X_eval.shape[0] > max_eval:
        X_eval = X_eval.iloc[:max_eval]
        y_eval = y_eval.loc[X_eval.index]
        eval_index = X_eval.index

    # Scale on train; LIME will pass scaled X to predict_fn (we inverse_transform)
    scaler = StandardScaler().fit(X_train.values)
    X_train_scaled = scaler.transform(X_train.values)

    # Wrap model predict_proba
    model_p = ProbModelWrapper(model)

    # Build LIME explainer once per (project, model)
    cfg = LimeConfig(
        discretizer="entropy",
        feature_selection="lasso_path",
        num_samples=int(num_samples),
        top_k=int(top_k),
        seed=seed,
    )
    explainer = _build_explainer(X_train_scaled, y_train.values, feat_cols, cfg)

    # Precompute predicted probs on X_eval (original scale)
    P_eval = model_p.predict_proba(X_eval.values)
    y_pred = (P_eval[:, 1] >= 0.5).astype(int)

    # Map label_mode -> label per instance
    def label_for(i: int) -> int:
        if label_mode == "fixed0":
            return 0
        if label_mode == "fixed1":
            return 1
        if label_mode == "pred":
            return int(y_pred[i])
        # 'opposite'
        return 1 - int(y_pred[i])

    # Explain each row
    per_rows = []
    for i, (rid, x) in enumerate(zip(eval_index, X_eval.values)):
        df_one = _explain_one(
            explainer, scaler, model_p, x, feat_cols,
            label_to_explain=label_for(i), cfg=cfg
        )
        if df_one.empty:
            continue
        df_one.insert(0, "test_idx", int(rid))
        df_one.insert(1, "instance_order", int(i))
        df_one.insert(2, "y_true", int(y_eval.loc[rid]))
        df_one.insert(3, "y_pred", int(y_pred[i]))
        df_one.insert(4, "proba1", float(P_eval[i, 1]))
        per_rows.append(df_one)

    per_instance = pd.concat(per_rows, axis=0, ignore_index=True) if per_rows else \
                   pd.DataFrame(columns=["test_idx","instance_order","y_true","y_pred","proba1",
                                         "feature","weight","abs_weight","rank","value"])

    # Aggregate summary
    if per_instance.empty:
        summary = pd.DataFrame(columns=["feature","mean_abs_weight","mean_weight","fraction_selected","count"])
        return per_instance, summary

    grouped = per_instance.groupby("feature", as_index=False).agg(
        mean_abs_weight=("abs_weight", "mean"),
        mean_weight=("weight", "mean"),
        count=("feature", "count"),
    )
    n_instances = int(len(np.unique(per_instance["test_idx"])))
    grouped["fraction_selected"] = grouped["count"] / max(1, n_instances)
    summary = grouped.sort_values(["mean_abs_weight","fraction_selected"], ascending=[False, False], kind="stable")

    return per_instance, summary


# ----------------------------- CLI entry -----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="LIME feature-importance rankings (wired to your loaders)")
    ap.add_argument("--projects", type=str, default="all",
                    help="Project name(s) or 'all' (comma/space separated)")
    ap.add_argument("--models", type=str, default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
                    help="Comma-separated model types")
    ap.add_argument("--only_tp", action="store_true", help="If set, explain only true positives")
    ap.add_argument("--label_mode", type=str, default="opposite",
                    choices=["opposite","pred","fixed0","fixed1"],
                    help="Which class to explain per instance")
    ap.add_argument("--top_k", type=int, default=10, help="Top-K features per instance to keep")
    ap.add_argument("--num_samples", type=int, default=5000, help="LIME num_samples")
    ap.add_argument("--max_eval", type=int, default=None, help="Max #instances per project/model to explain")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed for LIME")
    args = ap.parse_args()

    ds = read_dataset()
    if args.projects == "all":
        project_list = list(sorted(ds.keys()))
    else:
        project_list = [p.strip() for p in args.projects.replace(",", " ").split() if p.strip()]

    model_types = [m.strip() for m in args.models.replace(",", " ").split() if m.strip()]

    print(f"Running LIME rankings for {len(project_list)} projects × {len(model_types)} models")
    print(f"only_tp={args.only_tp}, label_mode={args.label_mode}, top_k={args.top_k}, num_samples={args.num_samples}")
    print()

    for project in project_list:
        train, test = ds[project]
        feat_cols = [c for c in train.columns if c != "target"]

        for model_type in model_types:
            model = get_model(project, model_type)

            per, summary = lime_rankings_for_split(
                train_df=train, test_df=test, model=model,
                only_true_positives=args.only_tp,
                top_k=args.top_k,
                num_samples=args.num_samples,
                label_mode=args.label_mode,
                max_eval=args.max_eval,
                seed=args.seed,
            )

            out_dir = os.path.join("evaluations", "lime", project, model_type)
            _ensure_dir(out_dir)
            per_path = os.path.join(out_dir, "lime_per_instance.csv")
            sum_path = os.path.join(out_dir, "lime_summary.csv")

            per.to_csv(per_path, index=False)
            summary.to_csv(sum_path, index=False)

            print(f"[OK] {project}/{model_type}: wrote {len(per)} rows (instances={per['test_idx'].nunique() if not per.empty else 0})")
            print(f"     → {per_path}")
            print(f"     → {sum_path}")

if __name__ == "__main__":
    main()
