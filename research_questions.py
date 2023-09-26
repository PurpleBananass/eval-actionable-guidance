import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import read_dataset, load_historical_changes
from hyparams import PLOTS, PLANS, EXPERIMENTS

EXPLAINER_ORDER = ["LIMEHPO", "SQAPlanner", "TimeLIME"]
STRATEGY_ORDER = ["coverage", "confidence", "lift"]
OTHERS_FILE_NAME = ["LIMEHPO", "TimeLIME"]
SQAPLANNERS_FILE_NAME = [
    "SQAPlanner_confidence",
    "SQAPlanner_coverage",
    "SQAPlanner_lift",
]
DISPLAY_NAME = {
    "LIMEHPO": "LIME-HPO",
    "SQAPlanner": "SQAPlanner",
    "TimeLIME": "TimeLIME",
}


def common_plotting_function(
    data,
    x,
    y,
    hue=None,
    xlabel=None,
    ylabel=None,
    ylim=None,
    fname=None,
    annotations=None,
):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})

    ax = sns.boxplot(
        x=x, y=y, hue=hue, data=data, palette="rocket" if hue is None else "crest"
    )

    # Annotations
    if annotations:
        for annotation in annotations:
            ax.annotate(**annotation)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold", labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", labelpad=10)
    if ylim:
        ax.set_ylim(*ylim)

    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=300, transparent=True, format="svg")
    plt.show()

