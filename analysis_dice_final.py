#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizations for DiCE evaluation outputs (top-K LIME constrained).

Inputs (from your DiCE evaluator):
  - ./evaluations/flip_rates_DiCE_topkLIME.csv
  - ./evaluations/feasibility/{distance}/{MODEL}_DiCE_{method}_{selection}.csv

Outputs:
  - ./evaluations/rq1_dice.png
  - ./evaluations/rq3_dice_{distance}_{selection}.png
  - (optional per-method) ./evaluations/rq3_dice_{distance}_{selection}_permethod.png
"""

from __future__ import annotations
import re
import glob
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- config -----------------------------

MODEL_ABBR = {
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "SVM": "SVM",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}
MODEL_ABBR_INV = {v: k for k, v in MODEL_ABBR.items()}

plt.rcParams["font.family"] = "Times New Roman"
sns.set_palette("crest")

EVAL_DIR = Path("./evaluations")


# ----------------------------- helpers -----------------------------

def _winsorize_by_group(df: pd.DataFrame, group_cols: list[str], ycol: str, q: float) -> pd.Series:
    """Return a clipped series to the [q, 1-q] quantiles per group (winsorization).
    If q == 0, returns the original series.
    """
    if q <= 0:
        return df[ycol]
    lo = df.groupby(group_cols)[ycol].transform(lambda s: s.quantile(q))
    hi = df.groupby(group_cols)[ycol].transform(lambda s: s.quantile(1 - q))
    return df[ycol].clip(lower=lo, upper=hi)


def _downsample(df: pd.DataFrame, group_cols: list[str], n: int | None) -> pd.DataFrame:
    """Optionally cap the number of rows per group to n by random sampling without replacement."""
    if not n or n <= 0:
        return df
    return (
        df.groupby(group_cols, group_keys=False)
          .apply(lambda g: g.sample(n=min(n, len(g)), random_state=7))
          .reset_index(drop=True)
    )


def _parse_method_from_path(p: str) -> str | None:
    # pattern: .../{ABBR}_DiCE_{method}_{selection}.csv
    m = re.search(r"_DiCE_([a-zA-Z0-9]+)_(best|first)\.csv$", p)
    return m.group(1) if m else None


# ----------------------------- RQ1: flip rates -----------------------------

def visualize_rq1_flip_rates_dice(csv_path: Path = EVAL_DIR / "flip_rates_DiCE_topkLIME.csv") -> None:
    if not csv_path.exists():
        print(f"[RQ1] Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Expect columns: ["Model", "Method", "Project", "Flip", "Computed", "#TP", "Flip%"]
    # Summarize per Model × Method
    g = df.groupby(["Model", "Method"], as_index=False)["Flip%"].mean(numeric_only=True)
    g["Flip%"] = g["Flip%"].fillna(0.0)

    plt.figure(figsize=(7, 3.8))
    ax = sns.barplot(
        data=g,
        x="Model", y="Flip%", hue="Method",
        hue_order=sorted(df["Method"].unique()),
        edgecolor="0.2"
    )
    # annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=2, fontsize=9)

    ax.set_ylim(0, min(0.0, max(0.05, g["Flip%"].max() * 1.15)))
    ax.set_xlabel("")
    ax.set_ylabel("Flip rate (mean over projects)")
    ax.legend(title="", frameon=False, ncols=min(3, len(g["Method"].unique())))
    plt.tight_layout()
    out = EVAL_DIR / "rq1_dice.png"
    plt.savefig(out, dpi=300)
    print(f"[RQ1] Saved {out}")


# ----------------------------- RQ3: feasibility -----------------------------

def _load_rq3_frames(distance: str, selection: str) -> pd.DataFrame:
    """
    Load all per-model per-method CSVs into one DataFrame.
    Returns columns: Model, Method, min, max, mean
    """
    base = EVAL_DIR / "feasibility" / distance
    if not base.exists():
        print(f"[RQ3] Dir not found: {base}")
        return pd.DataFrame()

    parts = []
    for abbr in MODEL_ABBR.values():
        for path in glob.glob(str(base / f"{abbr}_DiCE_*_{selection}.csv")):
            try:
                df = pd.read_csv(path)
                method = _parse_method_from_path(path)
                if method is None:
                    continue
                df = df.loc[:, [c for c in df.columns if c in {"min", "max", "mean"}]].copy()
                df["Model"] = MODEL_ABBR_INV.get(abbr, abbr)
                df["Method"] = method
                parts.append(df)
            except Exception as e:
                print(f"[RQ3] Could not read {path}: {e}")

    if not parts:
        print(f"[RQ3] No feasibility files found under {base}")
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    # ensure numeric
    for c in ("min", "max", "mean"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["min"])


def visualize_rq3_dice():
    plt.rcParams["font.family"] = "Times New Roman"
    # unchanged except we append "DiCE" to plot
    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"]

    # unchanged models
    models_to_plot = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
        "LightGBM": "LGBM",
        "CatBoost": "CatB",
    }

    total_df = pd.DataFrame()
    base_dir = Path("./evaluations/feasibility/mahalanobis")

    for model_full, abbr in models_to_plot.items():
        # original four explainers (unchanged)
        for explainer in ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]:
            try:
                df = pd.read_csv(base_dir / f"{abbr}_{explainer}.csv")
                df["Model"] = model_full
                df["Explainer"] = explainer
                if df is None or df.empty:
                    continue
                total_df = pd.concat([total_df, df], ignore_index=True)
            except FileNotFoundError:
                print(f"Warning: feasibility file not found for {abbr}_{explainer}")
                continue

        # NEW: DiCE — collect any per-method files and pool them under one label
        dice_files = list(base_dir.glob(f"{abbr}_DiCE*.csv"))
        if not dice_files:
            # silently OK; DiCE might not exist for this model
            pass
        else:
            parts = []
            for fpath in dice_files:
                try:
                    d = pd.read_csv(fpath)
                    parts.append(d)
                except Exception as e:
                    print(f"Warning: could not read {fpath}: {e}")
            if parts:
                dcat = pd.concat(parts, ignore_index=True)
                dcat["Model"] = model_full
                dcat["Explainer"] = "DiCE"
                total_df = pd.concat([total_df, dcat], ignore_index=True)

    fig = plt.figure(figsize=(6, 5.5))
    test_df = total_df.loc[:, ["Model", "Explainer", "min"]]

    sns.stripplot(
        data=test_df,
        x="Explainer",
        y="min",
        hue="Model",
        palette="crest",
        dodge=True,
        jitter=0.2,
        size=4,
        alpha=0.25,
        legend=False,
    )
    ax = sns.pointplot(
        data=test_df,
        x="Explainer",
        y="min",
        hue="Model",
        palette=["red"] * len(models_to_plot),
        dodge=0.8 - 0.8 / len(models_to_plot),
        errorbar=None,
        markers="x",
        markersize=4,
        linestyles="none",
        legend=False,
        zorder=10,
    )

    mean_df = test_df.groupby(["Model", "Explainer"]).mean(numeric_only=True)
    # same 5-model offsets you already use
    offsets = [-0.4, -0.2, 0, 0.2, 0.4]
    explainer_order = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"]
    for (model_name, expl), row in mean_df.iterrows():
        if expl not in explainer_order:
            continue
        model_idx = list(models_to_plot.keys()).index(model_name)
        offset = offsets[model_idx]
        # label formatting unchanged
        val_str = f'.{row["min"]:.2f}'
        val_str = "." + val_str[3:]
        ax.text(
            explainer_order.index(expl) + offset,
            row["min"] + 0.01,
            val_str,
            va="bottom",
            ha="center",
            fontsize=12,
            fontfamily="monospace",
            color="black",
        )

    plt.ylabel("")
    plt.xlabel("")
    # keep your original 0–1.5 axis range (unchanged)
    plt.ylim(0, 1.5)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    colors = sns.color_palette("crest", len(models_to_plot))
    legend_elements = [
        Path(
            facecolor=colors[i],
            edgecolor="black",
            label=models_to_plot[list(models_to_plot.keys())[i]],
        )
        for i in range(len(models_to_plot))
    ]
    fig.legend(
        handles=legend_elements,
        title="",
        loc="upper center",
        fontsize=10,
        frameon=False,
        ncols=5,
        bbox_to_anchor=(0.525, 0.94),
    )

    plt.savefig("./evaluations/rq3.png", dpi=300, bbox_inches="tight")


# ----------------------------- CLI -----------------------------

def main():
    ap = ArgumentParser(description="Visualize DiCE evaluation results (top-K LIME constrained)")
    ap.add_argument("--rq1", action="store_true", help="Make flip-rate chart for DiCE")
    ap.add_argument("--rq3", action="store_true", help="Make feasibility chart(s) for DiCE")
    ap.add_argument("--distance", type=str, default="mahalanobis", choices=["mahalanobis", "cosine"],
                    help="Which feasibility distance to load")
    ap.add_argument("--selection", type=str, default="best", choices=["best", "first"],
                    help="Which selection strategy file to load")
    ap.add_argument("--metric", type=str, default="min", choices=["min", "max", "mean"],
                    help="Which metric to plot on the y-axis")
    ap.add_argument("--winsor_q", type=float, default=0.01,
                    help="Winsorization q for display clipping per group (0 disables)")
    ap.add_argument("--per_method", action="store_true",
                    help="Also produce a per-method RQ3 plot (hue=Method)")
    ap.add_argument("--sample_per_group", type=int, default=None,
                    help="Downsample to at most N points per group for faster plotting")
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.rq1:
        visualize_rq1_flip_rates_dice()

    if args.rq3:
        visualize_rq3_dice(
            # distance=args.distance,
            # selection=args.selection,
            # metric=args.metric,
            # winsor_q=args.winsor_q,
            # per_method=args.per_method,
            # sample_per_group=args.sample_per_group,
        )


if __name__ == "__main__":
    main()
