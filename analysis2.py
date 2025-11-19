#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from argparse import ArgumentParser
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from scipy.stats import ranksums, wilcoxon
from sklearn.preprocessing import StandardScaler
from cliffs_delta import cliffs_delta
from tabulate import tabulate

from data_utils import get_model, get_true_positives, read_dataset

# ----------------------------- config & fallbacks -----------------------------

try:
    from hyparams import EXPERIMENTS, PROPOSED_CHANGES
except Exception:
    # sensible defaults if hyparams isn't available
    EXPERIMENTS = "flipped_instances"
    PROPOSED_CHANGES = "proposed_changes"

# Try to import helper from your CF evaluator; otherwise provide a compatible fallback.
try:
    from evaluate_cf import _flip_path as _cf_flip_path
except Exception:
    # Fallback: {EXPERIMENTS}/{project}/{ModelFull}/{ExplainerToken}_all.csv
    def _cf_flip_path(project: str, model_abbr: str, explainer_key: str) -> Path:
        ABBR2FULL = {
            "RF": "RandomForest",
            "XGB": "XGBoost",
            "SVM": "SVM",
            "LGBM": "LightGBM",
            "CatB": "CatBoost",
        }
        model_full = ABBR2FULL.get(model_abbr, model_abbr)
        token = explainer_key  # e.g., "CF"
        return Path(EXPERIMENTS) / f"{project}/{model_full}/{token}_all.csv"

ABBR2FULL = {
    "RF": "RandomForest",
    "XGB": "XGBoost",
    "SVM": "SVM",
    "LGBM": "LightGBM",
    "CatB": "CatBoost",
}
FULL2ABBR = {v: k for k, v in ABBR2FULL.items()}

SEL_DIR = Path("./evaluations/feasibility/mahalanobis/selected")

EXPL_ABBR_FILE = "CF"   # token used in filenames in /selected/
EXPL_LABEL = "CF"       # how it appears in plots

# ----------------------------- small helpers -----------------------------
def _save_csv(rows, columns, path):
    df = pd.DataFrame(rows, columns=columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}")

def _load_all_selected_for_model_abbr(model_abbr: str,
                                      expl_abbr: str = EXPL_ABBR_FILE) -> pd.DataFrame | None:
    """
    Load all cached 'selected' rows for a given model abbr (RF/XGB/...) and explainer token (CF).
    """
    if not SEL_DIR.exists():
        return None
    paths = glob(str(SEL_DIR / f"{model_abbr}_{expl_abbr}_*.csv"))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_flips_df(path: Path, feat_cols: list[str]) -> pd.DataFrame | None:
    """
    Load a flips CSV and try to standardize 'test_idx' as index if available.
    Ensures feature columns exist (drops missing ones if necessary).
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    # standardize 'test_idx'
    if "test_idx" in df.columns:
        df["test_idx"] = df["test_idx"].astype(int)
        df.set_index("test_idx", drop=True, inplace=True)
    else:
        # if first column looks like an index, set it
        first = df.columns[0].lower()
        if first in {"idx", "id", "index"}:
            df.set_index(df.columns[0], drop=True, inplace=True)
            try:
                df.index = df.index.astype(int)
            except Exception:
                pass

    # Keep only present feature columns if some are missing
    present = [c for c in feat_cols if c in df.columns]
    if len(present) == 0:
        return None
    # retain other columns too (e.g., meta), but ensure features are available when accessed
    return df


def _cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147: return "negligible"
    if d < 0.33:  return "small"
    if d < 0.474: return "medium"
    return "large"


# ----------------------------- CF-based computations -----------------------------

def cf_selected_flip_rates_df() -> pd.DataFrame:
    """
    Flip Rate for CF (selected) per *full model name*.
    Uses selected rows to reconstruct candidate values and re-predict.
    """
    ds = read_dataset()
    out_rows = []

    for abbr, model_full in ABBR2FULL.items():
        sel = _load_all_selected_for_model_abbr(abbr, EXPL_ABBR_FILE)
        if sel is None or sel.empty or "project" not in sel.columns or "test_idx" not in sel.columns:
            continue

        total_tp = 0
        flipped_tp = 0

        for project, g in sel.groupby("project", sort=False):
            project = str(project)
            if project not in ds:
                continue

            train, test = ds[project]
            feat_cols = [c for c in test.columns if c != "target"]
            present_cols = [c for c in feat_cols if c in g.columns]

            model = get_model(project, model_full)
            scaler = StandardScaler().fit(train[feat_cols].values)

            tp_df = get_true_positives(model, train, test)
            tp_idx = set(tp_df.index.astype(int).tolist())
            if not tp_idx:
                continue
            total_tp += len(tp_idx)

            flipped_here = set()
            for ti, gi in g.groupby("test_idx", sort=False):
                ti = int(ti)
                if ti not in tp_idx:
                    continue
                orig = test.loc[ti, feat_cols].astype(float)
                r = gi.iloc[0]
                cand = orig.copy()
                if present_cols:
                    cand[present_cols] = r[present_cols].astype(float).values
                X = scaler.transform([cand.values])
                if hasattr(model, "predict_proba"):
                    pred = int((model.predict_proba(X)[:, 1] >= 0.5)[0])
                else:
                    pred = int(model.predict(X)[0])
                if pred == 0:
                    flipped_here.add(ti)

            flipped_tp += len(flipped_here)

        if total_tp > 0:
            out_rows.append({
                "Explainer": EXPL_LABEL,
                "Model": model_full,
                "Flip Rate": flipped_tp / total_tp
            })

    return pd.DataFrame(out_rows)


def _read_abs_scores(path: str) -> pd.Series:
    """
    Read a CSV with a 'score' column, or a 1-col CSV to be named 'score'.
    Returns a float Series (may be empty).
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "score" not in df.columns:
        # try fallback single column
        if df.shape[1] == 1:
            df.columns = ["score"]
        else:
            return pd.Series(dtype=float)
    return pd.to_numeric(df["score"], errors="coerce").dropna()


def compare_changes_unpaired(model="XGB", ex_other="LIME", baseline="CF"):
    """
    Unpaired test: Mann-Whitney (ranksums) + Cliff's delta, ex_other vs baseline (CF).
    """
    s1 = _read_abs_scores(f"./evaluations/abs_changes/{model}_{ex_other}.csv")
    s2 = _read_abs_scores(f"./evaluations/abs_changes/{model}_{baseline}.csv")
    if len(s1) == 0 or len(s2) == 0:
        return [model, ex_other, baseline, "Mann-Whitney (unpaired)", 0, np.nan, np.nan, ""]

    # Mann-Whitney (via ranksums) p-value
    _, p = ranksums(s1, s2)

    # cliffs_delta returns (d, magnitude_str)
    d, mag = cliffs_delta(s1, s2)
    d = float(d) if d is not None else np.nan
    return [model, ex_other, baseline, "Mann-Whitney (unpaired)",
            int(min(len(s1), len(s2))), float(p), d, mag]
def run_implications_stats(baseline: str = "CF",
                           save_csv: str = "./evaluations/implications_vs_CF_stats.csv"):
    """
    For each model and each explainer in [LIME, LIME-HPO, TimeLIME, SQAPlanner],
    compare the distributions of 'total amount of changes required' vs CF.
    Uses paired Wilcoxon when paired files exist; falls back to unpaired.
    """
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    models = ["XGB", "RF", "SVM", "LGBM", "CatB"]
    others = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]

    rows = []
    for model in models:
        for other in others:
            res = compare_changes_paired_auto(model=model, ex_other=other, baseline=baseline)
            # normalize to 9 fields: [Model, Other, Baseline, Test, N, p, δ, Magnitude, MedianDiff]
            if len(res) == 9:
                rows.append(res)
            elif len(res) == 8:
                rows.append(res + [np.nan])
            else:
                # legacy 5-field return: [model, other, baseline, p, size]
                m, o, b, p, d = res[:5]
                rows.append([m, o, b, "Mann-Whitney (unpaired)", np.nan, p, d, _cliffs_magnitude(d), np.nan])

    df = pd.DataFrame(rows, columns=[
        "Model", "Other", "Baseline", "Test", "N",
        "p_value", "cliffs_delta", "Magnitude", "Median(other−CF)"
    ])
    df.to_csv(save_csv, index=False)
    print(f"Saved {save_csv}")

    # (optional) also pretty-print, like before
    from tabulate import tabulate
    pretty = []
    for _, r in df.iterrows():
        pretty.append([
            r["Model"], r["Other"], r["Baseline"], r["Test"],
            "NA" if pd.isna(r["N"]) else int(r["N"]),
            "NA" if pd.isna(r["p_value"]) else f'{float(r["p_value"]):.3e}',
            "NA" if pd.isna(r["cliffs_delta"]) else f'{float(r["cliffs_delta"]):.3f}',
            r["Magnitude"],
            "—" if pd.isna(r["Median(other−CF)"]) else f'{float(r["Median(other−CF)"]):.3f}',
        ])
    print(tabulate(
        pretty,
        headers=["Model","Other","Baseline","Test","N","p-value","Cliff’s δ","Magnitude","Median(other−CF)"],
        tablefmt="grid"
    ))


def compare_changes_paired_auto(model="XGB", ex_other="LIME", baseline="CF"):
    """
    Prefer paired Wilcoxon if {model}_{ex_other}_pairedCF.csv exists.
    Columns expected: score_explainer, score_cf, (optional) diff_explainer_minus_cf
    Falls back to unpaired if not available/usable.
    """
    pth = f"./evaluations/abs_changes/{model}_{ex_other}_pairedCF.csv"
    try:
        df = pd.read_csv(pth)
    except FileNotFoundError:
        return compare_changes_unpaired(model, ex_other, baseline)
    except Exception:
        return compare_changes_unpaired(model, ex_other, baseline)

    needed = {"score_explainer", "score_cf"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return compare_changes_unpaired(model, ex_other, baseline)

    a = pd.to_numeric(df["score_explainer"], errors="coerce").dropna()
    b = pd.to_numeric(df["score_cf"], errors="coerce").dropna()
    n = int(min(len(a), len(b)))
    if n == 0:
        return compare_changes_unpaired(model, ex_other, baseline)

    a = a.iloc[:n]
    b = b.iloc[:n]

    try:
        # Paired Wilcoxon
        _, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except Exception:
        return compare_changes_unpaired(model, ex_other, baseline)

    # Use Cliff's δ for effect magnitude reporting (still fine for paired samples)
    d, mag = cliffs_delta(a, b)
    d = float(d) if d is not None else np.nan

    med_diff = np.nan
    if "diff_explainer_minus_cf" in df.columns:
        diffs = pd.to_numeric(df["diff_explainer_minus_cf"], errors="coerce").dropna()
        if len(diffs) > 0:
            med_diff = float(np.median(diffs))

    return [model, ex_other, baseline, "Wilcoxon (paired)", n, float(p), d, mag, med_diff]

# ----------------------------- RQ1: Flip rates (adds CF if available) -----------------------------

def visualize_rq1():
    base_df = pd.read_csv("./evaluations/flip_rates.csv")
    try:
        cf_sel = cf_selected_flip_rates_df()
        df = pd.concat([base_df, cf_sel], ignore_index=True) if not cf_sel.empty else base_df
    except Exception as e:
        print(f"[rq1] Could not add {EXPL_LABEL}: {e}")
        df = base_df

    # drop "All" explainer
    df = df[df["Explainer"] != "All"].copy()

    plt.rcParams["font.family"] = "Times New Roman"
    sns.set_theme(style="whitegrid", context="paper")

    # explainer order
    expl_order = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", EXPL_LABEL]
    present_expl = [e for e in expl_order if e in set(df["Explainer"])]
    if not present_expl:
        present_expl = sorted(df["Explainer"].unique().tolist())

    # model order: by mean flip rate
    model_order = (
        df.groupby("Model", as_index=False)["Flip Rate"]
          .mean()
          .sort_values("Flip Rate", ascending=False)["Model"]
          .tolist()
    )
    if not model_order:
        print("[rq1] No data to plot.")
        return

    # ---------- use same bluish 'crest' palette as RQ2 ----------
    # Just like RQ2: pastel blue gradient
    palette = sns.color_palette("crest", len(present_expl))
    color_map = dict(zip(present_expl, palette))

    # make CF (if present) slightly darker to stand out
    if EXPL_LABEL in color_map:
        c = np.array(color_map[EXPL_LABEL])
        color_map[EXPL_LABEL] = tuple(np.clip(c * 0.85, 0.0, 1.0))
    # -------------------------------------------------------------

    n_models = len(model_order)

    fig, axes = plt.subplots(
        1,
        n_models,
        sharey=True,
        figsize=(1.7 * n_models + 1.5, 4.0),
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_full in zip(axes, model_order):
        sub = df[df["Model"] == model_full].copy()
        if sub.empty:
            ax.axis("off")
            continue

        # explainer order restricted to those present for this model
        expls_here = [e for e in present_expl if e in set(sub["Explainer"])]
        sub = (
            sub.set_index("Explainer")
               .reindex(expls_here)
               .dropna(subset=["Flip Rate"])
        )
        values = sub["Flip Rate"].values
        labels = sub.index.tolist()

        x = np.arange(len(labels))
        bar_width = 0.6

        colors = [color_map.get(e, "0.8") for e in labels]
        bars = ax.bar(
            x,
            values,
            width=bar_width,
            color=colors,
            edgecolor="0.4",
            linewidth=0.6,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax, left=False, bottom=False, right=True, top=True)

        # value labels
        for xi, v in zip(x, values):
            if np.isnan(v):
                continue
            ax.text(
                xi,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontfamily="monospace",
            )

        model_abbr = FULL2ABBR.get(model_full, model_full)
        ax.set_title(model_abbr, fontsize=11)

    axes[0].set_ylabel("Flip Rate", fontsize=11)
    for ax in axes[1:]:
        ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig("./evaluations/rq1.png", dpi=300, bbox_inches="tight")



# ----------------------------- RQ2 (unchanged, classic 4 explainers) -----------------------------

def visualize_rq2():
    explainers = {
        "LIME": "LIME",
        "LIME-HPO": "LIME-HPO",
        "TimeLIME": "TimeLIME",
        "SQAPlanner": "SQAPlanner_confidence",
    }
    models = {
        "RF": "RandomForest",
        "XGB": "XGBoost",
        "SVM": "SVM",
        "LGBM": "LightGBM",
        "CatB": "CatBoost",
    }

    plt.rcParams["font.family"] = "Times New Roman"

    # ---------------- load similarities base df ----------------
    total_df = pd.DataFrame()
    for model in models:
        try:
            df = pd.read_csv(f"./evaluations/similarities/{model}.csv", index_col=0)
            total_df = pd.concat([total_df, df], ignore_index=False)
        except FileNotFoundError:
            print(f"Warning: similarities file for {model} not found")
            continue

    if total_df.empty:
        print("No data to plot for RQ2.")
        return

    total_df.index.set_names("idx", inplace=True)
    total_df = total_df.set_index([total_df.index, total_df["project"]])
    total_df = total_df.drop(columns=["project"])

    # ---------------- add flipped/unflipped rows ----------------
    dset = read_dataset()
    for project in dset:
        train, test = dset[project]
        for model_type, model_full in models.items():
            try:
                true_positives = get_true_positives(
                    get_model(project, model_full), train, test
                )
            except Exception as e:
                print(f"Warning: could not get TPs for {project} {model_type}: {e}")
                continue

            for expl_label, expl_token in explainers.items():
                flip_path = (
                    Path(EXPERIMENTS)
                    / f"{project}/{model_full}/{expl_token}_all.csv"
                )
                if not flip_path.exists():
                    continue

                try:
                    df = pd.read_csv(flip_path, index_col=0)
                except Exception:
                    continue

                df["model"] = model_type
                df["explainer"] = expl_label
                df["project"] = project

                flipped = df.dropna()

                # add unflipped with score=None
                unflipped_index = true_positives.index.difference(flipped.index)
                unflipped = pd.DataFrame(index=unflipped_index)
                unflipped["model"] = model_type
                unflipped["explainer"] = expl_label
                unflipped["project"] = project
                unflipped["score"] = None
                unflipped.set_index(
                    [unflipped.index, unflipped["project"]], inplace=True
                )
                unflipped = unflipped.drop(columns=["project"])

                total_df = pd.concat(
                    [total_df, unflipped[["model", "explainer", "score"]]],
                    ignore_index=False,
                )

    if total_df.empty:
        print("No data to plot for RQ2 after adding flipped/unflipped.")
        return

    # -------- max count per explainer (for y-limit) --------
    colors = sns.color_palette("crest", len(models))
    max_count = {}
    for expl in explainers.keys():
        max_count[expl] = 0
        for model in models.keys():
            df = total_df[
                (total_df["explainer"] == expl) & (total_df["model"] == model)
            ]
            max_count[expl] = max(max_count[expl], len(df))

    expl_list = list(explainers.keys())               # row order
    model_list = list(models.keys())                  # column order
    n_rows, n_cols = len(expl_list), len(model_list)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 5.5),
        sharex=True,
        sharey=False,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    for r, expl in enumerate(expl_list):
        for c, model in enumerate(model_list):
            ax = axes[r, c]
            df = total_df[
                (total_df["explainer"] == expl) & (total_df["model"] == model)
            ]

            if len(df) > 0:
                sns.histplot(
                    data=df,
                    x="score",
                    ax=ax,
                    color=colors[c],
                    stat="count",
                    common_norm=False,
                    common_bins=True,
                    cumulative=True,
                    bins=10,
                )

            ax.set_ylim(0, max_count[expl] + 250)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("")  # no per-panel label

            # spines
            if c == 0:
                sns.despine(ax=ax, left=False, right=True, top=False, bottom=True)
            elif c == n_cols - 1:
                sns.despine(ax=ax, left=True, right=False, top=False, bottom=True)
            else:
                sns.despine(ax=ax, left=True, right=True, top=False, bottom=True)

            # column titles (top row)
            if r == 0:
                ax.set_title(model, fontsize=12)

            # row labels (first column)
            if c == 0:
                ax.set_ylabel(
                    expl,
                    fontsize=12,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=25,
                )

            # percentage annotations
            if len(df) > 0:
                for container in ax.containers:
                    for bar_idx, bar in enumerate(container):
                        if bar_idx == 0 or bar_idx == len(container) - 1:
                            ax.text(
                                bar.get_x() + bar.get_width() * (0.35 if bar_idx == 0 else 0.5),
                                bar.get_height() + 20,
                                f".{bar.get_height()/len(df)*100:.0f}",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                fontfamily="monospace",
                            )

            # x-axis ticks: only bottom row (SQAPlanner)
            if r < n_rows - 1:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labeltop=False,
                )
            else:
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticks, fontsize=10)
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=True,
                    top=False,
                    labelbottom=True,
                    labeltop=False,
                    pad=2,
                )

    # global x-label
    fig.text(0.5, 0.04, "Similarity Score", ha="center", fontsize=12)

    plt.tight_layout(rect=[0.04, 0.08, 0.99, 0.98])
    plt.savefig("./evaluations/rq2_combined.png", dpi=300)



# ----------------------------- Implications (CF baseline; no DiCE) -----------------------------

def visualize_implications():
    """
    Boxplot of distributions of total amount of changes required.
    Reads existing CSVs in ./evaluations/abs_changes/{MODEL}_{EXPLAINER}.csv
    Includes CF and excludes DiCE.
    """
    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "CF"]
    models = ["RF", "XGB", "SVM", "LGBM", "CatB"]
    total_df = pd.DataFrame()
    plt.rcParams["font.family"] = "Times New Roman"

    def _read_scores_df(path: str) -> pd.DataFrame | None:
        s = _read_abs_scores(path)
        if len(s) == 0:
            return None
        return pd.DataFrame({"score": s})

    for model in models:
        for explainer in explainers:
            if explainer == "CF":
                # CF might be single or sharded
                parts = []
                main = _read_scores_df(f"./evaluations/abs_changes/{model}_CF.csv")
                if main is not None: parts.append(main)
                else:
                    for p in glob(f"./evaluations/abs_changes/{model}_CF_*.csv"):
                        d = _read_scores_df(p)
                        if d is not None and not d.empty:
                            parts.append(d)
                if not parts:
                    print(f"Warning: no CF abs_changes files found for {model}")
                    continue
                df = pd.concat(parts, ignore_index=True)
            else:
                df = _read_scores_df(f"./evaluations/abs_changes/{model}_{explainer}.csv")
                if df is None:
                    print(f"Warning: abs_changes file not found for {model}_{explainer}")
                    continue

            df["Model"] = model
            df["Explainer"] = explainer
            total_df = pd.concat([total_df, df], ignore_index=True)

    if total_df.empty:
        print("No data to plot for implications.")
        return
    # --- NEW: export long-form data used by the boxplot ---
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    out_long = "./evaluations/implications_data_long.csv"
    total_df[["Model", "Explainer", "score"]].to_csv(out_long, index=False)
    print(f"Saved {out_long}")

    # (optional) quick summary table by Model×Explainer
    summary = (
        total_df.groupby(["Model", "Explainer"])["score"]
                .agg(N="count", mean="mean", median="median", std="std")
                .reset_index()
    )
    out_summary = "./evaluations/implications_data_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Saved {out_summary}")

    # # --- plotting continues below ---
    # # Plot (CF included; no DiCE)
    # present = [e for e in explainers if e in set(total_df["Explainer"])]
    # plt.figure(figsize=(6.2, 3.2))

    # Plot (CF included; no DiCE)
    present = [e for e in explainers if e in set(total_df["Explainer"])]
    plt.figure(figsize=(6.2, 3.2))
    ax = sns.boxplot(
        data=total_df,
        x="Explainer",
        y="score",
        hue="Model",
        order=present,
        hue_order=models,
        palette="crest",
        showfliers=False,
    )
    ax.set_ylabel("Total Amount of Changes Required", rotation=90, labelpad=3, fontsize=12)
    ax.set_xlabel("")
    plt.yticks(fontsize=12, ticks=[])
    ax.set_yticklabels(labels=[])
    ax.set_xticklabels(fontsize=12, labels=present)
    ax.get_legend().set_title("")
    ax.legend(loc="upper right", title="", fontsize=10, frameon=False)

    plt.ylim(-0.5, 30)  # keep your prior look; adjust if needed
    plt.tight_layout()
    plt.savefig("./evaluations/implications.png", dpi=300)


# ----------------------------- RQ3 (includes CF alongside others) -----------------------------
def visualize_rq3_bar():
    """
    RQ3 (bar version): same data as visualize_rq3, but show
    mean normalized distances as grouped bar plots.
    """
    plt.rcParams["font.family"] = "Times New Roman"

    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "CF"]
    models_to_plot = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
        "LightGBM": "LGBM",
        "CatBoost": "CatB",
    }

    distance_dir = "./evaluations/feasibility/mahalanobis"
    total_df = pd.DataFrame()

    # ---- load exactly the same data as visualize_rq3 ----
    for model_full, abbr in models_to_plot.items():
        for explainer in explainers:
            paths = []

            # main file
            main_path = f"{distance_dir}/{abbr}_{explainer}.csv"
            if Path(main_path).exists():
                paths.append(main_path)

            # shards
            shard_paths = glob(f"{distance_dir}/{abbr}_{explainer}_*.csv")
            paths.extend(shard_paths)

            frames = []
            for p in paths:
                try:
                    df = pd.read_csv(p)
                    if df is not None and not df.empty:
                        frames.append(df)
                except Exception:
                    pass

            if not frames:
                print(f"Warning: feasibility file(s) not found for {abbr}_{explainer}")
                continue

            df_all = pd.concat(frames, ignore_index=True)
            if "min" not in df_all.columns:
                print(f"[RQ3-bar] Missing 'min' column for {abbr}_{explainer}; skipping.")
                continue

            df_all["Model"] = model_full
            df_all["Explainer"] = explainer
            total_df = pd.concat([total_df, df_all], ignore_index=True)

    if total_df.empty:
        print("No feasibility data found for RQ3 bar plot.")
        return

    # ---- normalize and aggregate (same normalization as RQ3) ----
    plot_df = total_df.loc[:, ["Model", "Explainer", "min"]].copy()
    plot_df["min"] = pd.to_numeric(plot_df["min"], errors="coerce")
    plot_df.dropna(subset=["min"], inplace=True)
    plot_df["min_norm"] = plot_df["min"].clip(0, 1)

    mean_df = (
        plot_df.groupby(["Explainer", "Model"], as_index=False)["min_norm"]
               .mean()
    )

    # enforce ordering
    mean_df["Explainer"] = pd.Categorical(
        mean_df["Explainer"], categories=explainers, ordered=True
    )
    mean_df["Model"] = pd.Categorical(
        mean_df["Model"], categories=list(models_to_plot.keys()), ordered=True
    )

    # ---- bar plot ----
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    colors = sns.color_palette("crest", len(models_to_plot))

    sns.barplot(
        data=mean_df,
        x="Explainer",
        y="min_norm",
        hue="Model",
        hue_order=list(models_to_plot.keys()),
        order=explainers,
        palette=colors,
        edgecolor="black",
        ax=ax,
    )

    # aesthetics to match your style
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticks([0, 0.1,0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0,0.1, 0.2,0.3, 0.4, 0.5], fontsize=12)

    ax.legend(title="", loc="upper right", frameon=False, fontsize=10)

    sns.despine(ax=ax, left=True, right=False, top=True, bottom=False)

    # label mean values above each bar as .xx (same style as scatter version)
    for p in ax.patches:
        height = p.get_height()
        if not np.isfinite(height):
            continue
        label = f".{height:.2f}".replace("0.", "")
        ax.annotate(
            label,
            (p.get_x() + p.get_width() / 2, height + 0.01),
            ha="center",
            va="bottom",
            fontsize=9,
            fontfamily="monospace",
        )

    plt.tight_layout()
    plt.savefig("./evaluations/rq3_bar.png", dpi=300, bbox_inches="tight")
    print("Saved ./evaluations/rq3_bar.png")

def visualize_rq3():
    plt.rcParams["font.family"] = "Times New Roman"

    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "CF"]
    models_to_plot = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
        "LightGBM": "LGBM",
        "CatBoost": "CatB",
    }

    distance_dir = "./evaluations/feasibility/mahalanobis"
    total_df = pd.DataFrame()

    for model_full, abbr in models_to_plot.items():
        for explainer in explainers:
            # Primary file
            path = f"{distance_dir}/{abbr}_{explainer}.csv"
            frames = []

            try:
                df = pd.read_csv(path)
                if df is not None and not df.empty:
                    frames.append(df)
            except FileNotFoundError:
                # Fallback to sharded
                shard_paths = glob(f"{distance_dir}/{abbr}_{explainer}_*.csv")
                for p in shard_paths:
                    try:
                        d = pd.read_csv(p)
                        if d is not None and not d.empty:
                            frames.append(d)
                    except Exception:
                        pass

            if not frames:
                print(f"Warning: feasibility file(s) not found for {abbr}_{explainer}")
                continue

            df_all = pd.concat(frames, ignore_index=True)
            if "min" not in df_all.columns:
                print(f"[RQ3] Missing 'min' column for {abbr}_{explainer}; skipping.")
                continue

            df_all["Model"] = model_full
            df_all["Explainer"] = explainer
            total_df = pd.concat([total_df, df_all], ignore_index=True)

    if total_df.empty:
        print("No feasibility data found.")
        return

    # Normalize/clamp to [0,1]
    plot_df = total_df.loc[:, ["Model", "Explainer", "min"]].copy()
    plot_df["min"] = pd.to_numeric(plot_df["min"], errors="coerce")
    plot_df.dropna(subset=["min"], inplace=True)
    plot_df["min_norm"] = plot_df["min"].clip(0, 1)

    # Plot
    fig = plt.figure(figsize=(6.8, 5.6))
    sns.stripplot(
        data=plot_df,
        x="Explainer",
        y="min_norm",
        hue="Model",
        palette="crest",
        dodge=True,
        jitter=0.2,
        size=4,
        alpha=0.25,
        legend=False,
    )
    ax = sns.pointplot(
        data=plot_df,
        x="Explainer",
        y="min_norm",
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

    # Mean labels
    mean_df = plot_df.groupby(["Model", "Explainer"], as_index=False)["min_norm"].mean()
    offsets = (-0.4, -0.2, 0, 0.2, 0.4)  # for 5 models
    expl_order = explainers
    for _, row in mean_df.iterrows():
        model_name = row["Model"]
        expl = row["Explainer"]
        if expl not in expl_order:
            continue
        mi = list(models_to_plot.keys()).index(model_name)
        x = expl_order.index(expl) + offsets[mi]
        y = float(row["min_norm"])
        label = f".{y:.2f}".replace("0.", "")
        ax.text(
            x, min(max(y, 0.0), 1.0) + 0.01,
            label,
            va="bottom", ha="center",
            fontsize=11, fontfamily="monospace", color="black",
        )

    plt.ylabel("")
    plt.xlabel("")
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    colors = sns.color_palette("crest", len(models_to_plot))
    legend_elements = [
        Patch(facecolor=colors[i], edgecolor="black",
              label=list(models_to_plot.keys())[i])
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

    plt.tight_layout()
    plt.savefig("./evaluations/rq3.png", dpi=300, bbox_inches="tight")
# --- add near the other imports ---
import numpy as np
from scipy.stats import mannwhitneyu  # unpaired test alternative to ranksums

# ---------- RQ3 statistical tests (Other vs CF) ----------
def _read_rq3_df(model_abbr: str, explainer: str) -> pd.DataFrame:
    """
    Returns a DataFrame with normalized 'min' values for a given (model_abbr, explainer).
    Looks in ./evaluations/feasibility/mahalanobis/{abbr}_{expl}.csv (and shards).
    Columns returned: ['min_norm'] plus optional ['test_idx' or 'idx'] if present.
    """
    base = "./evaluations/feasibility/mahalanobis"
    frames = []

    # primary file
    path = f"{base}/{model_abbr}_{explainer}.csv"
    try:
        df = pd.read_csv(path)
        if df is not None and not df.empty:
            frames.append(df)
    except FileNotFoundError:
        pass

    # shards fallback
    for p in glob(f"{base}/{model_abbr}_{explainer}_*.csv"):
        try:
            d = pd.read_csv(p)
            if d is not None and not d.empty:
                frames.append(d)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["min_norm"])

    all_df = pd.concat(frames, ignore_index=True)
    if "min" not in all_df.columns:
        return pd.DataFrame(columns=["min_norm"])

    # normalize to [0,1]
    x = pd.to_numeric(all_df["min"], errors="coerce").dropna().clip(0, 1)
    out = pd.DataFrame({"min_norm": x.values})

    # carry over an index column if available (useful for potential pairing)
    for k in ("test_idx", "idx"):
        if k in all_df.columns and len(all_df[k]) == len(all_df["min"]):
            out[k] = all_df[k].values
            break

    return out


def _cliffs_magnitude(delta: float) -> str:
    """Label magnitude from δ (same thresholds you cited)."""
    ad = abs(delta)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"


def _format_p(p: float) -> str:
    return "NA" if (p is None or not np.isfinite(p)) else f"{p:.3e}"

def run_implications_stats(baseline="CF",
                           save_csv="./evaluations/implications_vs_CF_stats_raw.csv",
                           save_pretty_csv="./evaluations/implications_vs_CF_table.csv"):
    others = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]
    models = ["XGB", "RF", "SVM", "LGBM", "CatB"]

    raw_rows = []
    pretty_rows = []
    for model in models:
        for other in others:
            res = compare_changes_paired_auto(model=model, ex_other=other, baseline=baseline)
            # Normalize to: [model, other, base, test, n, p, delta, mag, med]
            if len(res) == 9:
                m, o, b, test, n, p, d, mag, med = res
            elif len(res) == 8:
                m, o, b, test, n, p, d, mag = res
                med = np.nan
            else:  # very old fallback
                (m, o, b, p, d) = res[:5]
                test, n, mag, med = "Mann-Whitney (unpaired)", 0, "NA", np.nan

            raw_rows.append({
                "Model": m, "Other": o, "Baseline": b, "Test": test, "N": int(n),
                "p_value": (float(p) if p == p else np.nan),  # keep NaN if not finite
                "cliffs_delta": (float(d) if d == d else np.nan),
                "magnitude": mag,
                "median_other_minus_CF": (float(med) if med == med else np.nan),
            })

            pretty_rows.append([
                m, o, b, test, int(n),
                ("NA" if (p is None or not np.isfinite(p)) else f"{float(p):.3e}"),
                ("NA" if (d is None or not np.isfinite(d)) else f"{float(d):.3f}"),
                mag,
                ("—" if (med is None or not np.isfinite(med)) else f"{float(med):.3f}")
            ])

    # save raw + pretty
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(save_csv, index=False)
    headers = ["Model","Other","Baseline","Test","N","p-value","Cliff’s δ","Magnitude","Median(other−CF)"]
    _save_pretty_csv(pretty_rows, headers, save_pretty_csv)

    # print exactly what you already see
    print(tabulate(pretty_rows, headers=headers, tablefmt="grid"))
    print(f"Saved {save_csv}")
    print(f"Saved {save_pretty_csv}")
def _save_pretty_csv(pretty_rows, headers, path):
    df = pd.DataFrame(pretty_rows, columns=headers)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_rq3_stat_tests(baseline: str = "CF",
                       save_csv: str = "./evaluations/rq3_stats.csv",
                       save_pretty_csv: str = "./evaluations/rq3_stats_table.csv"):
    models = ["RF", "XGB", "SVM", "LGBM", "CatB"]
    others = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]

    rows = []
    pretty = []

    for m in models:
        cf_df = _read_rq3_df(m, baseline)
        cf = cf_df["min_norm"] if not cf_df.empty else pd.Series(dtype=float)

        for other in others:
            oth_df = _read_rq3_df(m, other)
            oth = oth_df["min_norm"] if not oth_df.empty else pd.Series(dtype=float)

            if oth.empty or cf.empty:
                rows.append({
                    "Model": m, "Other": other, "Baseline": baseline,
                    "Test": "rank-sum", "N_other": int(len(oth)), "N_baseline": int(len(cf)),
                    "p_value": np.nan, "cliffs_delta": np.nan, "magnitude": "NA",
                    "median_other_minus_baseline": np.nan
                })
                pretty.append([m, other, baseline, "rank-sum",
                               f"{len(oth)} / {len(cf)}", "NA", "NA", "NA", "NA"])
                continue

            try:
                _, p = mannwhitneyu(oth, cf, alternative="two-sided")
            except Exception:
                _, p = ranksums(oth, cf)

            delta, mag = cliffs_delta(oth, cf)
            mag_label = mag if isinstance(mag, str) else _cliffs_magnitude(delta)
            med_diff = float(np.median(oth) - np.median(cf))

            rows.append({
                "Model": m, "Other": other, "Baseline": baseline,
                "Test": "rank-sum", "N_other": int(len(oth)), "N_baseline": int(len(cf)),
                "p_value": float(p), "cliffs_delta": float(delta),
                "magnitude": mag_label,
                "median_other_minus_baseline": med_diff
            })

            pretty.append([
                m, other, baseline, "rank-sum",
                f"{len(oth)} / {len(cf)}",
                _format_p(p),
                f"{delta:.3f}",
                mag_label,
                f"{med_diff:.3f}",
            ])

    # save BOTH: raw + the exact CLI-looking table
    out_df = pd.DataFrame(rows)
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_csv, index=False)

    headers = ["Model","Other","Baseline","Test","N(other/baseline)",
               "p-value","Cliff’s δ","Magnitude","Median(other−CF)"]
    _save_pretty_csv(pretty, headers, save_pretty_csv)

    # print exactly what you already see
    print(tabulate(pretty, headers=headers, tablefmt="grid"))
    print(f"Saved {save_csv}")
    print(f"Saved {save_pretty_csv}")


# ----------------------------- Misc utilities from earlier code -----------------------------

def group_diff(d1, d2):
    d1 = pd.to_numeric(pd.Series(d1), errors="coerce").dropna()
    d2 = pd.to_numeric(pd.Series(d2), errors="coerce").dropna()
    if len(d1) == 0 or len(d2) == 0:
        return np.nan, np.nan
    _, p = ranksums(d1, d2)
    _, size = cliffs_delta(d1, d2)
    return p, size


def list_status(
    model_type="XGBoost",
    explainers=("TimeLIME", "LIME-HPO", "LIME", "SQAPlanner_confidence", "DiCE"),
):
    dset = read_dataset()
    table = []
    headers = ["Project"] + [exp[:8] for exp in explainers] + ["common", "left"]
    total = 0
    total_left = 0
    for project in sorted(dset.keys()):
        row = {}
        table_row = [project]
        for explainer in explainers:
            flipped_path = Path(f"flipped_instances/{project}/{model_type}/{explainer}_all.csv")
            if not flipped_path.exists():
                print(f"{flipped_path} not exists")
                row[explainer] = set()
            else:
                flipped = pd.read_csv(flipped_path, index_col=0)
                computed_names = set(flipped.index)
                row[explainer] = computed_names

        plan_path = Path(f"proposed_changes/{project}/{model_type}/{explainers[0]}/plans_all.json")
        if plan_path.exists():
            with open(plan_path, "r") as f:
                plans = json.load(f)
                total_names = set(plans.keys())
        else:
            total_names = set()

        # common names between explainers
        common_names = row.get(explainers[0], set())
        for explainer in explainers[1:]:
            common_names = common_names.intersection(row.get(explainer, set()))
        row["common"] = common_names
        row["total"] = total_names
        for explainer in explainers:
            table_row.append(len(row.get(explainer, set())))
        table_row.append(f"{len(common_names)}/{len(total_names)}")
        table_row.append(len(total_names) - len(common_names))
        table.append(table_row)
        total += len(common_names)
        total_left += len(total_names) - len(common_names)
    table.append(["Total"] + [""] * len(explainers) + [total, total_left])
    print(f"Model: {model_type}")
    print(tabulate(table, headers=headers))


# ----------------------------- CLI -----------------------------

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--rq1", action="store_true")
    argparser.add_argument("--rq2", action="store_true")
    argparser.add_argument("--rq3", action="store_true")
    argparser.add_argument("--implications", action="store_true")
    argparser.add_argument("--rq3_stats", action="store_true")
    args = argparser.parse_args()

    if args.rq1:
        visualize_rq1()
    if args.rq2:
        visualize_rq2()
    if args.rq3:
        visualize_rq3()
        visualize_rq3_bar()
    if args.implications:
        # 1) Figure (CF included; no DiCE)
        visualize_implications()
        run_implications_stats(baseline="CF", save_csv="./evaluations/implications_vs_CF_stats.csv")
        

    # add with the other args
    

    # and near the bottom:
    if args.rq3_stats:
        # 1) RQ3 (feasibility minima) stats vs CF: already saves inside
        run_rq3_stat_tests(baseline="CF")

        # 2) Abs-change totals vs CF (paired if paired file exists, else unpaired)
        rows = []  # raw rows for CSV
        others = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]
        for model in ["XGB", "RF", "SVM", "LGBM", "CatB"]:
            for other in others:
                res = compare_changes_paired_auto(model=model, ex_other=other, baseline="CF")
                # normalize to 9 fields: [model, other, baseline, test, N, p, delta, mag, med_diff]
                if len(res) == 9:
                    rows.append(res)
                elif len(res) == 8:
                    rows.append(res + [np.nan])  # pad median diff
                else:
                    # legacy 5-field return: [model, other, baseline, p, size] → convert
                    model_, other_, base_, p_, d_ = res[:5]
                    rows.append([model_, other_, base_, "Mann-Whitney (unpaired)", np.nan, p_, d_, _cliffs_magnitude(d_), np.nan])

        # ---- save CSV for abs-change stats
        cols = ["Model","Other","Baseline","Test","N","p_value","cliffs_delta","Magnitude","Median(other−CF)"]
        _save_csv(rows, cols, "./evaluations/abs_changes_vs_CF_stats.csv")

        # ---- pretty print (unchanged)
        pretty = []
        for model, other, base, test, n, p, d, mag, med in rows:
            pretty.append([
                model, other, base, test,
                "NA" if (n is None or (isinstance(n, float) and np.isnan(n))) else int(n),
                "NA" if (p is None or not np.isfinite(p)) else f"{p:.3e}",
                "NA" if (d is None or not np.isfinite(d)) else f"{d:.3f}",
                mag,
                "—" if (med is None or not np.isfinite(med)) else f"{float(med):.3f}",
            ])
        print(tabulate(
            pretty,
            headers=["Model","Other","Baseline","Test","N","p-value","Cliff’s δ","Magnitude","Median(other−CF)"],
            tablefmt="grid"
        ))

