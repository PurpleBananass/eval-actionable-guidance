#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIME-based feature-importance rankings (per-instance + aggregated).

- Fits a StandardScaler on X_train and explains points in X_eval using LIME.
- Supports any classifier with predict_proba; wraps models missing it (e.g., linear SVM).
- Returns per-instance rankings and an aggregate summary (mean |weight|, mean signed weight, hit-rate).

Usage (example):
    per_inst, summary = lime_rankings(
        X_train=df_train[feat_cols],
        y_train=df_train['target'],
        X_eval=df_test[feat_cols].iloc[:100],   # first 100 instances to explain
        model=fitted_model,
        top_k=10,
        label=1,                                # explain contribution toward class 1
        num_samples=5000,
        seed=42
    )
    per_inst.to_csv("lime_per_instance.csv", index=False)
    summary.to_csv("lime_summary.csv", index=False)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
            if s.ndim > 1:
                s = s[:, 0]
            p1 = self._sigmoid(s)
            return np.stack([1 - p1, p1], axis=1)

        # Last resort: use predict as hard labels
        y = self.m.predict(X)
        p1 = (y == 1).astype(float)
        return np.stack([1 - p1, p1], axis=1)


# ----------------------------- core LIME API -----------------------------

@dataclass
class LimeConfig:
    discretizer: str = "entropy"           # 'entropy' or 'quartile' or None
    feature_selection: str = "lasso_path"  # 'auto', 'lasso_path', 'forward_selection', ...
    num_samples: int = 5000                # #samples drawn by LIME local surrogate
    label: int = 1                         # which class to explain (0 or 1)
    top_k: int = 10                        # how many features to keep per instance
    seed: int = 42


def _build_explainer(X_train_scaled: np.ndarray,
                     y_train: np.ndarray,
                     feature_names: List[str],
                     cfg: LimeConfig) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=y_train,
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
                 cfg: LimeConfig,
                 feature_names: List[str]) -> pd.DataFrame:
    """
    Explain a single instance. Returns a DataFrame with columns:
    ['instance_index','feature','weight','abs_weight','rank','value']
    """
    x_scaled = scaler.transform(x_row.reshape(1, -1))[0]

    exp = explainer.explain_instance(
        data_row=x_scaled,
        predict_fn=lambda X_scaled: model.predict_proba(scaler.inverse_transform(X_scaled)),
        labels=[cfg.label],
        num_samples=cfg.num_samples,
        num_features=len(feature_names),
    )

    # LIME returns a list of (feature_name_or_interval, weight) strings for the label.
    # We’ll map weights back to raw feature names by using the index-based map for the label.
    fmap = dict(exp.as_map()[cfg.label])  # {feature_index: weight}
    rows = []
    for j, fname in enumerate(feature_names):
        w = float(fmap.get(j, 0.0))
        if w != 0.0:
            rows.append((fname, w))

    # Sort by |weight| desc and keep top_k
    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    rows = rows[:cfg.top_k]

    # Pack results
    data = []
    for rnk, (fname, w) in enumerate(rows, start=1):
        data.append({
            "feature": fname,
            "weight": float(w),
            "abs_weight": float(abs(w)),
            "rank": rnk,
        })
    df = pd.DataFrame(data, columns=["feature", "weight", "abs_weight", "rank"])
    # Add the raw values of the explained row for convenience
    if len(df) > 0:
        vals = {f"val[{fname}]": x_row[list(feature_names).index(fname)] for fname in df["feature"]}
        for k, v in vals.items():
            df[k] = v
    return df


def lime_rankings(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_eval: pd.DataFrame,
                  model,
                  *,
                  cfg: Optional[LimeConfig] = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-instance LIME rankings for all rows in X_eval,
    and an aggregated summary across instances.

    Returns:
        per_instance: long DataFrame with columns
            ['instance','feature','weight','abs_weight','rank', ...vals...]
        summary: DataFrame with columns
            ['feature','mean_abs_weight','mean_weight','fraction_selected','count']
    """
    cfg = cfg or LimeConfig()
    feature_names = list(X_train.columns)

    # Scale: fit on X_train, apply to both (LIME likes standardized inputs)
    scaler = StandardScaler().fit(X_train.values)
    X_train_scaled = scaler.transform(X_train.values)

    # Wrap model to ensure predict_proba
    model_p = ProbModelWrapper(model)

    # Build explainer once
    explainer = _build_explainer(X_train_scaled, y_train.values.astype(int), feature_names, cfg)

    # Explain each row in X_eval
    per_rows = []
    for i, (_, row) in enumerate(X_eval.iterrows()):
        df_one = _explain_one(explainer, scaler, model_p, row.values.astype(float), cfg, feature_names)
        if df_one.empty:
            continue
        df_one.insert(0, "instance", int(i))  # relative index within X_eval
        per_rows.append(df_one)

    per_instance = pd.concat(per_rows, axis=0, ignore_index=True) if per_rows else \
                   pd.DataFrame(columns=["instance","feature","weight","abs_weight","rank"])

    # Aggregate summary
    if per_instance.empty:
        summary = pd.DataFrame(columns=["feature","mean_abs_weight","mean_weight","fraction_selected","count"])
        return per_instance, summary

    grouped = per_instance.groupby("feature", as_index=False).agg(
        mean_abs_weight=("abs_weight", "mean"),
        mean_weight=("weight", "mean"),
        count=("feature", "count")
    )
    n_instances = int(X_eval.shape[0])
    grouped["fraction_selected"] = grouped["count"] / max(1, n_instances)
    summary = grouped.sort_values(["mean_abs_weight", "fraction_selected"], ascending=[False, False], kind="stable")

    return per_instance, summary


# ----------------------------- demo main (optional) -----------------------------
if __name__ == "__main__":
    # Minimal self-test on a toy dataset; replace this with your own loader/model.
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(Xtr, ytr)

    cfg = LimeConfig(
        discretizer="entropy",
        feature_selection="lasso_path",
        num_samples=5000,
        label=1,
        top_k=10,
        seed=42
    )

    per, summary = lime_rankings(Xtr, ytr, Xte.iloc[:50], clf, cfg=cfg)
    print("\n=== Per-instance (first 5 rows) ===")
    print(per.head())
    print("\n=== Aggregate summary (top 10) ===")
    print(summary.head(10))
