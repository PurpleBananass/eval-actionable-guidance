import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from cliffs_delta import cliffs_delta
from matplotlib.patches import Patch
from scipy.stats import ranksums
from tabulate import tabulate
import seaborn
from data_utils import get_model, get_true_positives, read_dataset
from hyparams import EXPERIMENTS


def visualize_rq1():
    df = pd.read_csv("./evaluations/flip_rates.csv")

    df = df.sort_values(by="Flip Rate", ascending=False)
    sns.set_palette("crest")
    plt.rcParams["font.family"] = "Times New Roman"

    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "All"]

    # Increase figure height to accommodate more models
    plt.figure(figsize=(4, 5))
    ax = sns.barplot(
        y="Model",
        x="Flip Rate",
        hue="Explainer",
        data=df,
        edgecolor="0.2",
        hue_order=explainers,
    )

    for i, container in enumerate(ax.containers):
        for bar in container:
            bar.set_height(bar.get_height() * 0.8)
            bar.set_edgecolor("black")
            bar.set_linewidth(0.5)

            left_text = plt.text(
                -0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{explainers[i]}",
                va="center",
                ha="right",
                fontsize=11,
            )

            value_text = plt.text(
                bar.get_width() - 0.01,
                bar.get_y() + bar.get_height() / 2 + 0.01,
                f"{round(bar.get_width(), 2)}",
                va="center",
                ha="right",
                fontsize=9,
                fontfamily="monospace",
            )

            if i != 4:
                bar.set_alpha(0.4)
            else:
                left_text.set_fontweight("bold")
                value_text.set_color("white")
                value_text.set_fontweight("bold")

    ax.margins(x=0, y=0.01)
    plt.xlim(0, 1)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.ylabel("")
    plt.xlabel("")

    plt.legend([], [], frameon=False)
    plt.xticks(fontsize=11, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig("./evaluations/rq1.png", dpi=300)


def visualize_rq2():
    explainers = {
        "LIME": "LIME",
        "LIME-HPO": "LIME-HPO",
        "TimeLIME": "TimeLIME",
        "SQAPlanner": "SQAPlanner_confidence",
    }
    models = {"RF": "RandomForest", "XGB": "XGBoost", "SVM": "SVM", "LGBM": "LightGBM", "CatB": "CatBoost"}
    plt.rcParams["font.family"] = "Times New Roman"

    # Load similarities data
    total_df = pd.DataFrame()
    for model in models:
        try:
            df = pd.read_csv(f"./evaluations/similarities/{model}.csv", index_col=0)
            total_df = pd.concat([total_df, df], ignore_index=False)
        except FileNotFoundError:
            print(f"Warning: similarities file for {model} not found")
            continue
            
    if not total_df.empty:
        total_df.index.set_names("idx", inplace=True)
        total_df = total_df.set_index([total_df.index, total_df["project"]])
        total_df = total_df.drop(columns=["project"])

    # Load flipped/unflipped data
    projects = read_dataset()
    for project in projects:
        train, test = projects[project]
        for model_type in models:
            try:
                true_positives = get_true_positives(
                    get_model(project, models[model_type]), train, test
                )
                for explainer in explainers:
                    flip_path = (
                        Path(EXPERIMENTS)
                        / f"{project}/{models[model_type]}/{explainers[explainer]}_all.csv"
                    )
                    if not flip_path.exists():
                        continue
                    df = pd.read_csv(flip_path, index_col=0)
                    df["model"] = model_type
                    df["explainer"] = explainer
                    df["project"] = project

                    flipped = df.dropna()

                    unflipped_index = true_positives.index.difference(flipped.index)
                    unflipped = pd.DataFrame(index=unflipped_index)
                    unflipped["model"] = model_type
                    unflipped["explainer"] = explainer
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
            except Exception as e:
                print(f"Warning: Error processing {project} {model_type}: {e}")
                continue

    # Calculate max counts
    colors = sns.color_palette("crest", len(models))
    max_count = {}
    for explainer in explainers:
        max_count[explainer] = 0
        for model in models:
            df = total_df[
                (total_df["explainer"] == explainer) & (total_df["model"] == model)
            ]
            max_count[explainer] = max(max_count[explainer], len(df))

    # Create ONE figure with 4 rows × 5 columns
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(4, 5, figure=fig, hspace=0.01, wspace=0)
    
    for e, explainer in enumerate(explainers):
        for j, model in enumerate(["RF", "XGB", "SVM", "LGBM", "CatB"]):
            ax = fig.add_subplot(gs[e, j])
            
            df = total_df[
                (total_df["explainer"] == explainer) & (total_df["model"] == model)
            ]
            
            if len(df) > 0:
                sns.histplot(
                    data=df,
                    x="score",
                    ax=ax,
                    color=colors[j],
                    stat="count",
                    common_norm=False,
                    common_bins=True,
                    cumulative=True,
                    bins=10,
                )
            
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylim(0, max_count[explainer] + 250)
            
            # Only show x-axis on bottom row
            if e < 3:
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12)
            
            ax.set_xlabel("")
            ax.set_ylabel("")
            
            # Despine logic
            if j == 0:
                sns.despine(ax=ax, left=False, right=True, top=False, trim=False)
            elif j == 4:
                sns.despine(ax=ax, left=True, right=False, top=False, trim=False)
            else:
                sns.despine(ax=ax, left=True, right=True, top=False, trim=False)
            
            # Add percentage labels
            if len(df) > 0:
                for container in ax.containers:
                    for bar_idx, bar in enumerate(container):
                        if bar_idx == len(container) - 1:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 20,
                                f".{bar.get_height()/len(df)*100:.0f}",
                                ha="center",
                                va="bottom",
                                fontsize=10,
                                fontfamily="monospace",
                            )
                        if bar_idx == 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 10 * 3.5,
                                bar.get_height() + 20,
                                f".{bar.get_height()/len(df)*100:.0f}",
                                ha="center",
                                va="bottom",
                                fontsize=10,
                                fontfamily="monospace",
                            )
    
    plt.tight_layout()
    plt.savefig("./evaluations/rq2_combined.png", dpi=300)



def compare_changes(model="XGBoost", ex1="TimeLIME", ex2="LIME-HPO"):
    try:
        df1 = pd.read_csv(
            f"./evaluations/abs_changes/{model}_{ex1}.csv", names=["score"], header=0
        )
        
        if ex2 == "DiCE":
            df2 = pd.read_csv(
            f"./evaluations/abs_changes/{model}_{ex2}_random_best_100.csv", names=["score"], header=0
        )
        else:
            df2 = pd.read_csv(
            f"./evaluations/abs_changes/{model}_{ex2}.csv", names=["score"], header=0
        )

        p, size = group_diff(df1["score"], df2["score"])
        print(f"{model} {ex1} {ex2} p-value: {p:.4f}, size: {size}")
        return [model, ex1, ex2, p, size]
    except FileNotFoundError as e:
        print(f"Warning: File not found for {model} comparison")
        print(f"./evaluations/abs_changes/{model}_{ex2}_random_best_100.csv")
        return [model, ex1, ex2, 0, 0]


def visualize_implications():
    import glob

    explainers_base = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]  # non-DiCE
    models = ["RF", "XGB", "SVM", "LGBM", "CatB"]
    total_df = pd.DataFrame()
    plt.rcParams["font.family"] = "Times New Roman"

    # 1) Load non-DiCE abs_changes (same as before)
    for model in models:
        for explainer in explainers_base:
            try:
                df = pd.read_csv(
                    f"./evaluations/abs_changes/{model}_{explainer}.csv",
                    names=["score"],
                    header=0,
                )
                df["Model"] = model
                df["Explainer"] = explainer
                total_df = pd.concat([total_df, df], ignore_index=True)
            except FileNotFoundError:
                print(f"Warning: abs_changes file not found for {model}_{explainer}")
                continue

    # 2) Load ALL DiCE abs_changes files for each model and aggregate as "DiCE"
    for model in models:
        dice_paths = glob.glob(f"./evaluations/abs_changes/{model}_DiCE_*.csv")
        if not dice_paths:
            print(f"Warning: no DiCE abs_changes files found for {model}")
            continue
        for path in dice_paths:
            try:
                df = pd.read_csv(path)  # these files were saved with a 'score' header
                if "score" not in df.columns:
                    # fallback to your older read style if needed
                    df = pd.read_csv(path, names=["score"], header=0)
                df["Model"] = model
                df["Explainer"] = "DiCE"
                total_df = pd.concat([total_df, df], ignore_index=True)
            except Exception as e:
                print(f"Warning: failed to read {path}: {e}")

    if total_df.empty:
        print("No data to plot for implications.")
        return

    # Plot
    plt.figure(figsize=(5, 3))
    ax = sns.boxplot(
        data=total_df,
        x="Explainer",
        y="score",
        hue="Model",
        palette="crest",
        showfliers=False,
        hue_order=models,
    )
    ax.set_ylabel("Total Amount of Changes Required", rotation=90, labelpad=3, fontsize=12)
    ax.set_xlabel("")
    plt.yticks(fontsize=12, ticks=[])
    ax.set_yticklabels(labels=[])
    plt.xticks(fontsize=12)
    ax.set_xticklabels(fontsize=12, labels=["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"])
    ax.get_legend().set_title("")
    ax.legend(loc="upper right", title="", fontsize=10, frameon=False)

    plt.ylim(-0.5, 30)  # keep your original y-limit
    plt.tight_layout()
    plt.savefig("./evaluations/implications.png", dpi=300)


def visualize_rq3_histograms_per_combo_selected():
    """
    One PNG per (Explainer × Model). For DiCE, uses ONLY the instances that were
    actually selected for the minimum distance in RQ3 (from 'selected' CSVs).
    Bins: 0, 0.2, ..., 1.0; heights are probabilities; labels show raw counts.
    """
    import os, glob
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    base = "./evaluations/feasibility/mahalanobis"
    sel_base = f"{base}/selected"
    out_dir = f"{base}/histograms_selected"
    os.makedirs(out_dir, exist_ok=True)

    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"]
    models = ["RF", "XGB", "SVM", "LGBM", "CatB"]

    bin_edges = np.linspace(0.0, 1.0, 6)  # 0,0.2,...,1.0

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"

    for model in models:
        for explainer in explainers:
            # ----- load distances -----
            if explainer == "DiCE":
                # Use ONLY the selected files saved by rq3_feasibility()
                paths = glob.glob(f"{sel_base}/{model}_DiCE_*_selected.csv")
                frames = []
                for p in paths:
                    try:
                        if os.path.getsize(p) > 0:
                            frames.append(pd.read_csv(p))
                    except Exception:
                        pass
                if not frames:
                    print(f"[skip] No selected DiCE files for {model}")
                    continue
                df = pd.concat(frames, ignore_index=True)
            else:
                # Non-DiCE (unchanged): read the standard RQ3 per-explainer file
                path = f"{base}/{model}_{explainer}.csv"
                if not os.path.exists(path):
                    print(f"[skip] {path} not found")
                    continue
                df = pd.read_csv(path)

            if "min" not in df.columns:
                print(f"[skip] {model}/{explainer} missing 'min' column")
                continue

            x = pd.to_numeric(df["min"], errors="coerce").dropna().clip(0, 1)
            if x.empty:
                print(f"[skip] {model}/{explainer} has no valid values")
                continue

            counts, _ = np.histogram(x.values, bins=bin_edges)
            total = counts.sum()
            if total == 0:
                print(f"[skip] {model}/{explainer} all-zero histogram")
                continue
            probs = counts / total  # normalized heights

            # ----- plot -----
            fig, ax = plt.subplots(figsize=(4.6, 3.4))
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            width = (bin_edges[1] - bin_edges[0]) * 0.9

            bars = ax.bar(centers, probs, width=width, edgecolor="0.2")

            for i, b in enumerate(bars):
                if counts[i] == 0:
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{counts[i]}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_xticks(bin_edges)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(0.05, probs.max() * 1.15))
            ax.set_xlabel("Normalized minimum distance (0–1)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            title_expl = "DiCE (selected)" if explainer == "DiCE" else explainer
            ax.set_title(f"{model} • {title_expl}  (n={total})", fontsize=12, pad=8)
            sns.despine(ax=ax)
            fig.tight_layout()

            out_path = f"{out_dir}/{model}_{explainer}_hist.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"[saved] {out_path}")


def visualize_rq3_hist_per_model():
    """
    One PNG with 5 subplots (RF, XGB, SVM, LGBM, CatB).
    Each subplot shows bars for explainers; bar height = mean(normalized 'min').
    Reads from ./evaluations/feasibility/mahalanobis/.
    Saves: ./evaluations/rq3_meanmins_all.png
    """
    import os, glob
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    base = "./evaluations/feasibility/mahalanobis"
    os.makedirs("./evaluations", exist_ok=True)

    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"]
    models = ["RF", "XGB", "SVM", "LGBM", "CatB"]

    # collect means (and n) per model×explainer
    rows = []
    for abbr in models:
        for exp in explainers:
            if exp == "DiCE":
                paths = glob.glob(f"{base}/{abbr}_DiCE_*.csv")
                frames = []
                for p in paths:
                    try:
                        frames.append(pd.read_csv(p))
                    except Exception:
                        pass
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                else:
                    # optional fallback if you already pre-aggregated to a single file
                    try:
                        df = pd.read_csv(f"{base}/{abbr}_DiCE.csv")
                    except Exception:
                        continue
            else:
                try:
                    df = pd.read_csv(f"{base}/{abbr}_{exp}.csv")
                except Exception:
                    continue

            if "min" not in df.columns:
                continue
            x = pd.to_numeric(df["min"], errors="coerce").dropna()
            if x.empty:
                continue

            # ensure normalized 0–1 (defensive)
            x = x.clip(0, 1)

            rows.append({"Model": abbr, "Explainer": exp,
                         "mean_min": float(x.mean()), "n": int(x.size)})

    if not rows:
        print("[warn] No feasibility files found.")
        return

    plot_df = pd.DataFrame(rows)
    plot_df["Explainer"] = pd.Categorical(plot_df["Explainer"],
                                          categories=explainers, ordered=True)
    plot_df = plot_df.sort_values(["Model", "Explainer"])

    # plotting
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"
    palette = sns.color_palette("crest", n_colors=len(explainers))
    color_map = {e: palette[i] for i, e in enumerate(explainers)}

    fig, axes = plt.subplots(
        nrows=1, ncols=len(models), figsize=(15, 3.8), sharey=True, constrained_layout=True
    )

    for ax, abbr in zip(axes, models):
        data = plot_df[plot_df["Model"] == abbr]
        if data.empty:
            ax.set_title(abbr)
            ax.set_axis_off()
            continue

        # draw bars in explainer order
        colors = [color_map[e] for e in data["Explainer"]]
        bars = ax.bar(data["Explainer"], data["mean_min"], color=colors, edgecolor="0.2")

        # put an 'x' marker at the bar height and annotate μ & n
        for i, (bar, (_, r)) in enumerate(zip(bars, data.iterrows())):
            ax.plot(i, r["mean_min"], marker="x", markersize=6, color="black", zorder=5)
            ax.text(bar.get_x() + bar.get_width()/2.0, r["mean_min"] + 0.02,
                    f"μ={r['mean_min']:.2f}\n(n={int(r['n'])})",
                    ha="center", va="bottom", fontsize=8)

        ax.set_title(abbr, pad=6, fontsize=14)
        ax.set_xlabel("")
        ax.set_xticklabels(explainers, rotation=30, ha="right", fontsize=10)

    axes[0].set_ylabel("Mean of normalized minimum distance (0–1)", fontsize=11)
    for ax in axes[1:]:
        ax.set_ylabel("")

    for ax in axes:
        ax.set_ylim(0, 1.0)
        sns.despine(ax=ax)

    out = "./evaluations/rq3_meanmins_all.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[saved] {out}")
def summarize_rq3_skewness_and_plot():
    """
    Summarize skewness of normalized feasibility minima in a compact way and plot a single heatmap.

    - Classic explainers are read from: ./evaluations/feasibility/mahalanobis
    - DiCE uses ONLY SELECTED rows from: ./evaluations/feasibility/mahalanobis/selected
      (files like: {abbr}_DiCE_*_selected.csv)

    Outputs:
      ./evaluations/mahalanobis_skew_summary.csv
      ./evaluations/mahalanobis_skew_heatmap.png

    Skew index definition used in the heatmap:
      right_skew_index = mass(0–0.2) − mass(0.8–1.0)
      -> Higher means more mass near 0 (i.e., more right-skewed).
      -> Range: [-1, 1], with 0 meaning balanced tails.
    """
    import glob
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import skew as fisher_skew
    from pathlib import Path

    plt.rcParams["font.family"] = "Times New Roman"

    # Models and explainers (DiCE appended at the end)
    models_to_plot = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
        "LightGBM": "LGBM",
        "CatBoost": "CatB",
    }
    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"]

    classic_dir = "./evaluations/feasibility/mahalanobis"
    dice_sel_dir = "./evaluations/feasibility/mahalanobis/selected"  # selected-only

    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    rows = []
    for model_full, abbr in models_to_plot.items():
        for expl in explainers:
            # --- load data
            if expl == "DiCE":
                files = glob.glob(f"{dice_sel_dir}/{abbr}_DiCE_*_selected.csv")
                if not files:
                    # No selected DiCE for this model—skip gracefully
                    continue
                parts = []
                for p in files:
                    try:
                        d = pd.read_csv(p)
                        if d is not None and not d.empty:
                            parts.append(d)
                    except Exception:
                        continue
                if not parts:
                    continue
                df = pd.concat(parts, ignore_index=True)
            else:
                path = f"{classic_dir}/{abbr}_{expl}.csv"
                try:
                    df = pd.read_csv(path)
                except FileNotFoundError:
                    continue
                if df is None or df.empty:
                    continue

            if "min" not in df.columns:
                continue

            # normalize/clamp to [0,1]
            vals = pd.to_numeric(df["min"], errors="coerce").dropna().values
            if vals.size == 0:
                continue
            vals = np.clip(vals, 0.0, 1.0)

            # histogram proportions over fixed bins
            counts, _ = np.histogram(vals, bins=bins)
            n = counts.sum()
            if n == 0:
                continue
            props = counts / n

            # robust Bowley skew
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            bowley = (q3 + q1 - 2 * med) / (q3 - q1) if (q3 - q1) > 0 else np.nan
            # Fisher skewness (for reference)
            fskew = fisher_skew(vals, bias=False) if vals.size >= 3 else np.nan

            left_mass  = float(props[0])   # [0.0, 0.2)
            right_mass = float(props[-1])  # [0.8, 1.0]
            # Right-skew index: higher => more mass near 0 => more right-skewed
            right_skew_index = left_mass - right_mass  # in [-1, 1]

            rows.append({
                "Model": model_full,
                "ModelAbbr": abbr,
                "Explainer": expl,
                "N": int(n),
                "left_mass_0_0.2": left_mass,
                "right_mass_0.8_1.0": right_mass,
                "right_skew_index": float(right_skew_index),  # main metric for heatmap
                "bowley_skew": float(bowley),
                "fisher_skew": float(fskew),
                "p_0_0.2": float(props[0]),
                "p_0.2_0.4": float(props[1]),
                "p_0.4_0.6": float(props[2]),
                "p_0.6_0.8": float(props[3]),
                "p_0.8_1.0": float(props[4]),
                "median": float(med),
            })

    if not rows:
        print("No data found for skewness summary.")
        return

    summary = pd.DataFrame(rows)
    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    summary.to_csv("./evaluations/mahalanobis_skew_summary.csv", index=False)
    print("Saved ./evaluations/mahalanobis_skew_summary.csv")

    # --- Heatmap of right_skew_index (rows: explainers, columns: models) ---
    heat = summary.pivot_table(
        index="Explainer",
        columns="Model",
        values="right_skew_index",
        aggfunc="mean"
    )

    # enforce consistent ordering and allow missing entries
    heat = heat.reindex(index=explainers, columns=list(models_to_plot.keys()))

    plt.figure(figsize=(9.2, 4.1))
    ax = sns.heatmap(
        heat,
        annot=True, fmt=".2f",
        linewidths=0.3, linecolor="0.85",
        cmap="vlag", vmin=-1, vmax=1, center=0,
        cbar_kws={"label": "Right-skew index (left − right)"},
        square=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Right-skew index of Normalized Feasibility Distances\n(higher = more mass near 0)")
    plt.tight_layout()
    plt.savefig("./evaluations/mahalanobis_skew_heatmap.png", dpi=300)
    print("Saved ./evaluations/mahalanobis_skew_heatmap.png")

def plot_skew_slopegraph():
    """
    Pretty slopegraph of skew index by Explainer × Model.

    Reads:  ./evaluations/mahalanobis_skew_summary.csv
            (produced by your summarize function)
    Saves:  ./evaluations/mahalanobis_skew_slope.png
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import seaborn as sns
    from pathlib import Path

    plt.rcParams["font.family"] = "Times New Roman"

    # --- load summary ---
    path = Path("./evaluations/mahalanobis_skew_summary.csv")
    if not path.exists():
        print("Missing ./evaluations/mahalanobis_skew_summary.csv — run the summarizer first.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("Summary CSV is empty.")
        return

    # Which column to plot (support either name you ended up with)
    xcol = "right_skew_index" if "right_skew_index" in df.columns else "skew_index"
    if xcol not in df.columns:
        print("No skew index column found in summary CSV.")
        return

    # Order
    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"]
    models = ["RandomForest", "XGBoost", "SVM", "LightGBM", "CatBoost"]

    df = df[df["Explainer"].isin(explainers) & df["Model"].isin(models)].copy()
    if df.empty:
        print("No matching rows for the chosen explainers/models.")
        return

    df["Explainer"] = pd.Categorical(df["Explainer"], categories=explainers, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=models, ordered=True)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    xs = np.arange(len(models))
    # pleasant, distinct colors
    colors = dict(zip(explainers, sns.color_palette("Set2", len(explainers))))

    # background guides
    ax.axhline(0, color="0.85", lw=1)
    ax.axhspan(-0.2, 0.2, color="0.95", zorder=0)  # “near symmetric” band

    # scale marker sizes by N (sample size); clamp for readability
    global_max_n = max(1, np.nanmax(df["N"].values))
    def size_from_n(n):
        if not np.isfinite(n): return 40
        s = 30 + 120 * (n / global_max_n)
        return float(np.clip(s, 30, 150))

    # plot one line per explainer
    for expl in explainers:
        sub = df[df["Explainer"] == expl].set_index("Model")
        y = [sub.loc[m, xcol] if m in sub.index else np.nan for m in models]
        n = [sub.loc[m, "N"] if m in sub.index else np.nan for m in models]
        y = np.array(y, dtype=float)
        n = np.array(n, dtype=float)

        # line across models (breaks across NaNs automatically)
        ax.plot(xs, y, color=colors[expl], lw=2, alpha=0.95, zorder=2)

        # scatter with size ∝ N
        valid = np.isfinite(y)
        sizes = [size_from_n(v) for v in n[valid]]
        ax.scatter(xs[valid], y[valid],
                   s=sizes,
                   color=colors[expl],
                   edgecolor="white",
                   linewidth=0.8,
                   zorder=3)

        # label the rightmost available point inline (so we don't need a legend)
        if np.any(valid):
            last_i = np.where(valid)[0][-1]
            ax.annotate(expl,
                        xy=(xs[last_i], y[last_i]),
                        xytext=(8, 0),
                        textcoords="offset points",
                        va="center",
                        ha="left",
                        fontsize=11,
                        color=colors[expl],
                        fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # axes cosmetics
    ax.set_xlim(xs.min() - 0.2, xs.max() + 0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=10)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.8)

    # titles / captions
    ax.set_ylabel("Skew index  =  mass[0–0.2] − mass[0.8–1.0]", fontsize=12)
    ax.set_title("Skew of Normalized Feasibility Distances\n(higher = more mass near 0)", fontsize=13, pad=10)

    # small caption about point sizes
    ax.text(0.01, 0.01,
            "Marker size ∝ sample size (N).",
            transform=ax.transAxes, fontsize=9, color="0.35")

    fig.tight_layout()
    out = "./evaluations/mahalanobis_skew_slope.png"
    plt.savefig(out, dpi=300)
    print(f"Saved {out}")


def visualize_rq3():
    import glob
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from matplotlib.patches import Patch

    plt.rcParams["font.family"] = "Times New Roman"

    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DiCE"]
    models_to_plot = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
        "LightGBM": "LGBM",
        "CatBoost": "CatB",
    }

    # Base dirs
    distance_dir = "./evaluations/feasibility/mahalanobis"
    dice_selected_dir = "./evaluations/feasibility/mahalanobis/selected"  # <-- selected-only DiCE

    total_df = pd.DataFrame()

    for model_full, abbr in models_to_plot.items():
        # 1) Classic explainers from the standard directory
        for explainer in ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]:
            path = f"{distance_dir}/{abbr}_{explainer}.csv"
            try:
                df = pd.read_csv(path)
                if df is None or df.empty:
                    print(f"[RQ3] Empty file: {path}")
                    continue
                df["Model"] = model_full
                df["Explainer"] = explainer
                total_df = pd.concat([total_df, df], ignore_index=True)
            except FileNotFoundError:
                print(f"Warning: feasibility file not found for {abbr}_{explainer}")
                continue

        # 2) DiCE — ONLY selected instances saved by RQ3
        dice_sel_files = glob.glob(f"{dice_selected_dir}/{abbr}_DiCE_*_selected.csv")
        if dice_sel_files:
            parts = []
            for p in dice_sel_files:
                try:
                    d = pd.read_csv(p)
                    if d is not None and not d.empty:
                        parts.append(d)
                except Exception as e:
                    print(f"Warning: could not read {p}: {e}")
            if parts:
                dcat = pd.concat(parts, ignore_index=True)
                dcat["Model"] = model_full
                dcat["Explainer"] = "DiCE"
                total_df = pd.concat([total_df, dcat], ignore_index=True)
        else:
            print(f"Warning: no SELECTED DiCE files found for model {abbr} in {dice_selected_dir}")

    if total_df.empty:
        print("No feasibility data found.")
        return

    # Use normalized distances (clip defensively)
    plot_df = total_df.loc[:, ["Model", "Explainer", "min"]].copy()
    plot_df["min"] = pd.to_numeric(plot_df["min"], errors="coerce")
    plot_df.dropna(subset=["min"], inplace=True)
    plot_df["min_norm"] = plot_df["min"].clip(0, 1)

    # Plot (same style as before)
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
        label = f".{y:.2f}".replace("0.", ".")
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

    colors = seaborn.color_palette("crest", len(models_to_plot))
    legend_elements = [
        Patch(facecolor=colors[i], edgecolor="black",
              label=models_to_plot[list(models_to_plot.keys())[i]])
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


def group_diff(d1, d2):
    d1 = d1.dropna()
    d2 = d2.dropna()
    if len(d1) == 0 or len(d2) == 0:
        return 0, 0
    _, p = ranksums(d1, d2)
    _, size = cliffs_delta(d1, d2)
    return p, size


def list_status(
    model_type="XGBoost",
    explainers=["TimeLIME", "LIME-HPO", "LIME", "SQAPlanner_confidence", "DiCE"],
):
    projects = read_dataset()
    table = []
    headers = ["Project"] + [exp[:8] for exp in explainers] + ["common", "left"]
    total = 0
    total_left = 0
    projects = sorted(projects.keys())
    for project in projects:
        row = {}
        table_row = [project]
        for explainer in explainers:
            flipped_path = Path(
                f"flipped_instances/{project}/{model_type}/{explainer}_all.csv"
            )

            if not flipped_path.exists():
                print(f"{flipped_path} not exists")
                row[explainer] = set()
            else:
                flipped = pd.read_csv(flipped_path, index_col=0)
                computed_names = set(flipped.index)
                row[explainer] = computed_names
        plan_path = Path(
            f"proposed_changes/{project}/{model_type}/{explainers[0]}/plans_all.json"
        )
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


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--rq1", action="store_true")
    argparser.add_argument("--rq2", action="store_true")
    argparser.add_argument("--rq3", action="store_true")
    argparser.add_argument("--implications", action="store_true")
    args = argparser.parse_args()

    if args.rq1:
        visualize_rq1()
    if args.rq2:
        visualize_rq2()
    if args.rq3:
        visualize_rq3()
        # visualize_rq3_histograms_per_combo_selected()
        # visualize_rq3_hist_per_model()
        plot_skew_slopegraph()

        # summarize_rq3_skewness_and_plot()
    if args.implications:
        visualize_implications()

        table = []
        # Updated to include all models
        for model in ["XGB", "RF", "SVM", "LGBM", "CatB"]:
            table.append(compare_changes(model=model, ex1="LIME", ex2="LIME-HPO"))
            table.append(compare_changes(model=model, ex1="LIME", ex2="TimeLIME"))
            table.append(compare_changes(model=model, ex1="LIME", ex2="SQAPlanner"))
            table.append(compare_changes(model=model, ex1="LIME", ex2="DiCE"))
        print(
            tabulate(
                table,
                headers=[
                    "Model",
                    "Explainer 1",
                    "Explainer 2",
                    "p-value",
                    "Effect Size",
                ],
                tablefmt="grid",
            )
        )