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




def visualize_rq3():
    plt.rcParams["font.family"] = "Times New Roman"
    explainers = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]
    
    # Choose which models to visualize (original 3 only to match your original plot)
    # models_to_plot = {" RandomForest": "RF", "XGBoost": "XGB", "SVM": "SVM"}
    models_to_plot = {"RandomForest": "RF", "XGBoost": "XGB", "SVM": "SVM", "LightGBM": "LGBM", "CatBoost": "CatB"}
    
    total_df = pd.DataFrame()
    for model in models_to_plot:
        for explainer in explainers:
            try:
                df = pd.read_csv(
                    f"./evaluations/feasibility/mahalanobis/{models_to_plot[model]}_{explainer}.csv"
                )
                df["Model"] = model
                df["Explainer"] = explainer
                if df is None or df.empty:
                    print(f"[RQ3] Missing file: {file_path}")
                    continue

                if "min" in df.columns:  # Updated to match original format
                    y = pd.to_numeric(df["min"], errors="coerce").dropna()
                    print(len(y))
                total_df = pd.concat([total_df, df], ignore_index=True)
            except FileNotFoundError:
                print(f"Warning: feasibility file not found for {models_to_plot[model]}_{explainer}")
                continue

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
        palette=["red"] * len(models_to_plot),  # Fixed: use models_to_plot
        dodge=0.8 - 0.8 / len(models_to_plot),  # Fixed: use models_to_plot
        errorbar=None,
        markers="x",
        markersize=4,
        linestyles="none",
        legend=False,
        zorder=10,
    )

    mean_df = test_df.groupby(["Model", "Explainer"]).mean()
    for i, row in mean_df.iterrows():
        model_idx = list(models_to_plot.keys()).index(i[0])  # Fixed: use models_to_plot
        val_str = f'.{row["min"]:.2f}'
        val_str = "." + val_str[3:]
        
        # Text positioning for 3 models (original layout)
        # if model_idx == 0:
        #     offset = -0.3
        # elif model_idx == 1:
        #     offset = -0.05
        # else:
        #     offset = 0.25
        offsets = [-0.4, -0.2, 0, 0.2, 0.4]  # Evenly spaced for 5 models
        model_idx = list(models_to_plot.keys()).index(i[0])
        offset = offsets[model_idx]
        ax.text(
            explainers.index(i[1]) + offset,
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
    plt.ylim(0, 1.5)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    colors = sns.color_palette("crest", len(models_to_plot))  # Fixed: use models_to_plot
    legend_elements = [
        Patch(
            facecolor=colors[i], edgecolor="black", label=models_to_plot[list(models_to_plot.keys())[i]]
        )
        for i in range(len(models_to_plot))  # Fixed: use models_to_plot
    ]

    # fig.legend(
    #     handles=legend_elements,
    #     title="",
    #     loc="upper center",
    #     fontsize=12,
    #     frameon=False,
    #     ncols=3,  # Back to 3 columns for 3 models
    #     bbox_to_anchor=(0.525, 0.94),
    # )
    fig.legend(
        handles=legend_elements,
        title="",
        loc="upper center",
        fontsize=10,  # Smaller font for 5 models
        frameon=False,
        ncols=5,  # 5 columns for 5 models
        bbox_to_anchor=(0.525, 0.94),
    )

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