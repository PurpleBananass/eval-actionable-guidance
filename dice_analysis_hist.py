#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from tabulate import tabulate


# ---- small helpers -----------------------------------------------------------

MODEL_ABBR = {"RandomForest":"RF","XGBoost":"XGB","SVM":"SVM","LightGBM":"LGBM","CatBoost":"CatB"}
ABBR_TO_MODEL = {v:k for k,v in MODEL_ABBR.items()}

def _load_first(*paths):
    """Return the first readable CSV as a DataFrame, else None."""
    for p in paths:
        if p and os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def _ensure_fonts():
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set_palette("crest")


# ---- RQ1: Flip rates (DiCE with methods) ------------------------------------

def visualize_rq1_dice(methods: list[str] = None):
    """
    Reads ./evaluations/flip_rates_DiCE_all_methods.csv and draws a bar chart
    of per-model flip rates, optionally filtering by methods.
    """
    _ensure_fonts()

    df = _load_first("./evaluations/flip_rates_DiCE_all_methods.csv")
    if df is None or df.empty:
        print("[RQ1] No DiCE flip-rate CSV found. Skipping.")
        return

    # Filter by methods if specified
    if methods:
        df = df[df["Method"].isin(methods)]
        if df.empty:
            print(f"[RQ1] No data for methods: {methods}")
            return

    # Expected columns: Model, Method, Project, Flip, Computed, #TP, Flip%
    if len(df["Method"].unique()) == 1:
        # Single method - show per-model means
        per_model = (
            df.groupby("Model")[["Flip", "Computed", "#TP", "Flip%"]]
              .mean(numeric_only=True)
              .reset_index()
        ).sort_values("Flip%", ascending=False)
        
        method_name = df["Method"].iloc[0]
        print(f"\n[RQ1] DiCE {method_name} per-model means:")
        print(tabulate(per_model, headers=per_model.columns, tablefmt="github", showindex=False))

        plt.figure(figsize=(4.2, 3.8))
        ax = sns.barplot(
            data=per_model,
            y="Model", x="Flip%",
            edgecolor="0.2"
        )
        for p in ax.patches:
            ax.text(
                p.get_width() - 0.01, p.get_y() + p.get_height()/2.0,
                f"{p.get_width():.2f}",
                ha="right", va="center", fontsize=10, fontfamily="monospace"
            )
            p.set_alpha(0.9)
            p.set_linewidth(0.6)
            p.set_edgecolor("black")

        ax.set_xlim(0, 1.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        plt.xticks([0, 0.25, 0.5, 0.75, 1.0], fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        out = f"./evaluations/rq1_dice_{method_name}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[RQ1] Saved {out}")
        
    else:
        # Multiple methods - show grouped comparison
        per_model_method = (
            df.groupby(["Model", "Method"])[["Flip%"]]
              .mean(numeric_only=True)
              .reset_index()
        )
        
        print(f"\n[RQ1] DiCE per-model-method means:")
        print(tabulate(per_model_method, headers=per_model_method.columns, tablefmt="github", showindex=False))

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            data=per_model_method,
            x="Model", y="Flip%", hue="Method",
            edgecolor="0.2"
        )
        
        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width()/2, p.get_height() + 0.01,
                f"{p.get_height():.2f}",
                ha="center", va="bottom", fontsize=9, fontfamily="monospace"
            )
            p.set_alpha(0.9)
            p.set_linewidth(0.6)
            p.set_edgecolor("black")

        ax.set_ylim(0, 1.0)
        ax.set_xlabel("")
        ax.set_ylabel("Flip Rate")
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(title="Method", fontsize=10)
        plt.tight_layout()
        
        methods_str = "_".join(sorted(df["Method"].unique()))
        out = f"./evaluations/rq1_dice_comparison_{methods_str}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[RQ1] Saved {out}")


# ---- RQ2: Plan similarity (DiCE with methods) -------------------------------

def visualize_rq2_dice(methods: list[str] = None):
    """
    Loads ./evaluations/similarities/{ABBR}_DiCE_{method}.csv and plots histograms
    for each model and method combination.
    """
    _ensure_fonts()

    all_df = pd.DataFrame()
    methods_to_check = methods or ["random", "kdtree", "genetic"]
    
    for abbr, model in ABBR_TO_MODEL.items():
        for method in methods_to_check:
            df = _load_first(f"./evaluations/similarities/{abbr}_DiCE_{method}.csv")
            if df is None or df.empty:
                continue
            if "score" not in df.columns:
                print(f"[RQ2] File {abbr}_DiCE_{method}.csv lacks 'score' column (skipping).")
                continue
            df["Model"] = model
            df["Method"] = method
            all_df = pd.concat([all_df, df], ignore_index=True)

    if all_df.empty:
        print("[RQ2] No similarities data found for DiCE. Skipping.")
        return

    unique_methods = sorted(all_df["Method"].unique())
    models_order = list(MODEL_ABBR.keys())
    
    if len(unique_methods) == 1:
        # Single method - show model comparison
        method = unique_methods[0]
        colors = sns.color_palette("crest", len(models_order))

        fig, axes = plt.subplots(1, len(models_order), figsize=(12, 2.6), sharey=True, sharex=True)
        for j, model in enumerate(models_order):
            ax = axes[j]
            sdf = all_df[(all_df["Model"] == model) & (all_df["Method"] == method)]
            if not sdf.empty:
                sns.histplot(
                    data=sdf,
                    x="score",
                    ax=ax,
                    color=colors[j],
                    stat="count",
                    common_norm=False,
                    bins=12
                )
            ax.set_title(MODEL_ABBR[model], fontsize=11)
            ax.set_xlabel("Similarity score")
            if j == 0:
                ax.set_ylabel("Count")
            else:
                ax.set_ylabel("")

        plt.tight_layout()
        out = f"./evaluations/rq2_dice_{method}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[RQ2] Saved {out}")
    else:
        # Multiple methods - show method comparison for each model
        n_methods = len(unique_methods)
        n_models = len(models_order)
        
        fig, axes = plt.subplots(n_models, n_methods, figsize=(3*n_methods, 2.5*n_models), 
                               sharey=True, sharex=True)
        axes = axes.reshape(n_models, n_methods) if n_models > 1 else axes.reshape(1, -1)
        
        colors = sns.color_palette("crest", n_methods)
        
        for i, model in enumerate(models_order):
            for j, method in enumerate(unique_methods):
                ax = axes[i, j]
                sdf = all_df[(all_df["Model"] == model) & (all_df["Method"] == method)]
                if not sdf.empty:
                    sns.histplot(
                        data=sdf,
                        x="score",
                        ax=ax,
                        color=colors[j],
                        stat="count",
                        common_norm=False,
                        bins=12
                    )
                if i == 0:
                    ax.set_title(method, fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"{MODEL_ABBR[model]}\nCount", fontsize=10)
                else:
                    ax.set_ylabel("")
                if i == n_models - 1:
                    ax.set_xlabel("Similarity score")
                else:
                    ax.set_xlabel("")

        plt.tight_layout()
        methods_str = "_".join(unique_methods)
        out = f"./evaluations/rq2_dice_comparison_{methods_str}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[RQ2] Saved {out}")


# ---- RQ3: Feasibility (DiCE with methods) -----------------------------------

def visualize_rq3_dice_comparison(methods: list[str] = None, strategies: list[str] = None):
    """Plot feasibility comparison across methods and/or strategies."""
    _ensure_fonts()
    
    methods_to_check = methods or ["random", "kdtree", "genetic"]
    strategies_to_check = strategies or ["best", "first"]
    model_order = ["RandomForest", "XGBoost", "SVM", "LightGBM", "CatBoost"]
    
    # Filter to only available combinations
    available_combos = []
    for method in methods_to_check:
        for strategy in strategies_to_check:
            # Check if any files exist for this combo
            found = False
            for model in model_order:
                abbr = MODEL_ABBR[model]
                file_path = f"./evaluations/feasibility/mahalanobis/{abbr}_DiCE_{method}_{strategy}_hist_seed.csv"
                if os.path.exists(file_path):
                    found = True
                    break
            if found:
                available_combos.append((method, strategy))
    
    if not available_combos:
        print("[RQ3] No feasibility data found.")
        return
    
    n_combos = len(available_combos)
    cols = min(3, n_combos)
    rows = (n_combos + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), sharey=True)
    if n_combos == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten() if cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for combo_idx, (method, strategy) in enumerate(available_combos):
        ax = axes[combo_idx] if combo_idx < len(axes) else None
        if ax is None:
            continue
            
        combo_data = []
        
        for model in model_order:
            abbr = MODEL_ABBR[model]
            file_path = f"./evaluations/feasibility/mahalanobis/{abbr}_DiCE_{method}_{strategy}_hist_seed.csv"
            df = _load_first(file_path)
            
            if df is None or df.empty:
                continue

            if "min" in df.columns:
                y = pd.to_numeric(df["min"], errors="coerce").dropna()
                # Filter outliers
                y = y[y <= 1.0]
            else:
                continue

            if not y.empty:
                combo_data.append(pd.DataFrame({"Model": model, "yval": y.values}))

        if not combo_data:
            continue
            
        data = pd.concat(combo_data, ignore_index=True)
        data["Model"] = pd.Categorical(data["Model"], categories=model_order, ordered=True)
        
        # Plot strips
        sns.stripplot(
            data=data, x="Model", y="yval", order=model_order,
            ax=ax, jitter=0.2, size=3, alpha=0.25, color="#6BAF92"
        )
        
        # Add means
        means = data.groupby("Model", sort=False)["yval"].mean().reindex(model_order)
        for i, m in enumerate(means.tolist()):
            if pd.notna(m):
                ax.scatter(i, m, marker="x", s=60, zorder=10, color="red")
                ax.text(i, m + 0.03, f"{m:.2f}", ha="center", va="bottom",
                       fontsize=10, fontfamily="monospace", color="black")
        
        ax.set_title(f"{method} ({strategy})", fontsize=12)
        ax.set_xlabel("")
        if combo_idx % cols == 0:
            ax.set_ylabel("Min distance (lower is better)")
        ax.set_ylim(0, 2)
        ax.set_xticklabels(model_order, fontsize=10, rotation=45)
        sns.despine(ax=ax)
    
    # Hide unused subplots
    for i in range(n_combos, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    combo_str = "_".join([f"{m}_{s}" for m, s in available_combos])
    out = f"./evaluations/rq3_dice_comparison_{combo_str[:50]}.png"  # Truncate long names
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[RQ3] Saved comparison: {out}")


def visualize_rq3_dice(method: str = "random", strategy: str = "best"):
    """
    Plot DiCE feasibility for a specific method and strategy combination.
    
    Args:
        method: DiCE method (random, kdtree, genetic)
        strategy: Selection strategy (best, first)
    """
    _ensure_fonts()
    
    model_order = ["RandomForest", "XGBoost", "SVM", "LightGBM", "CatBoost"]
    rows = []

    for model in model_order:
        abbr = MODEL_ABBR[model]
        file_path = f"./evaluations/feasibility/mahalanobis/{abbr}_DiCE_{method}_{strategy}_hist_seed.csv"
        df = _load_first(file_path)

        if df is None or df.empty:
            print(f"[RQ3] Missing file: {file_path}")
            continue

        if "min" in df.columns:  # Updated to match original format
            y = pd.to_numeric(df["min"], errors="coerce").dropna()
            print(len(y))
            # Filter outliers
            original_count = len(y)
            # y = y[y <= 10.0]  
            filtered_count = len(y)
            if original_count != filtered_count:
                print(f"[RQ3] {abbr}/{method}/{strategy}: Filtered {original_count - filtered_count} outliers")
        else:
            print(f"[RQ3] No 'min' column in {abbr}, available: {list(df.columns)}")
            continue

        if not y.empty:
            rows.append(pd.DataFrame({"Model": model, "yval": y.values}))

    if not rows:
        print(f"[RQ3] No data to plot for {method}/{strategy}.")
        return

    data = pd.concat(rows, ignore_index=True)
    data["Model"] = pd.Categorical(data["Model"], categories=model_order, ordered=True)

    # Plot
    plt.figure(figsize=(3.2, 4.8))
    ax = plt.gca()

    sns.stripplot(
        data=data,
        x="Model",
        y="yval",
        order=model_order,
        ax=ax,
        jitter=0.2,
        size=4,
        alpha=0.25,
        color="#6BAF92",
    )

    # Red X at the mean
    means = data.groupby("Model", sort=False)["yval"].mean().reindex(model_order)
    for i, m in enumerate(means.tolist()):
        if pd.notna(m):
            ax.scatter(i, m, marker="x", s=80, zorder=10, color="red")
            ax.text(i, m + 0.03, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=12, fontfamily="monospace", color="black")

    ax.set_xlabel("")
    ax.set_ylabel("Min distance to historical deltas (lower is better)")
    ax.set_ylim(0, 1.5)
    ax.set_title(f"DiCE {method} ({strategy} selection)", fontsize=14)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    sns.despine(ax=ax)
    plt.tight_layout()
    
    out = f"./evaluations/rq3_dice_{method}_{strategy}_hist.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[RQ3] Saved {out}")


# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize DiCE results across methods and strategies")
    ap.add_argument("--rq1", action="store_true", help="Draw DiCE RQ1 figure")
    ap.add_argument("--rq2", action="store_true", help="Draw DiCE RQ2 figure")
    ap.add_argument("--rq3", action="store_true", help="Draw DiCE RQ3 figure")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated methods: random,kdtree,genetic")
    ap.add_argument("--strategies", type=str, default="best",
                    help="Comma-separated strategies: best,first (RQ3 only)")
    ap.add_argument("--compare", action="store_true", 
                    help="Generate comparison plots across methods/strategies")
    
    args = ap.parse_args()
    
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    
    print(f"Visualizing methods: {methods}")
    if args.rq3:
        print(f"Strategies: {strategies}")
    
    if args.rq1:
        visualize_rq1_dice(methods=methods)
        
    if args.rq2:
        visualize_rq2_dice(methods=methods)
        
    if args.rq3:
        if args.compare:
            visualize_rq3_dice_comparison(methods=methods, strategies=strategies)
        else:
            # Single plot - use first method/strategy
            method = methods[0] if methods else "random"
            strategy = strategies[0] if strategies else "best"
            visualize_rq3_dice(method=method, strategy=strategy)


if __name__ == "__main__":
    main()