# %% Import packages ##############################################################################################################
import json
from matplotlib.text import Text
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import (
    get_true_positives,
    read_dataset,
    load_historical_changes,
    get_release_names,
    get_release_ratio,
)
from hyparams import MODELS, PLOTS, PROPOSED_CHANGES, EXPERIMENTS, RESULTS


# %%
ALL_EXPLAINERS = [
    "DeFlip",
    "DeFlip_actionable",
    "LIMEHPO",
    "TimeLIME",
    "SQAPlanner_coverage",
    "SQAPlanner_confidence",
    "SQAPlanner_lift",
]
EXPLAINER_ORDER = ["LIME-HPO", "TimeLIME", "SQAPlanner"]
STRATEGY_ORDER = ["coverage", "confidence", "lift"]
OTHERS_NAME = ["LIME-HPO", "TimeLIME"]
OTHERS_FILE_NAME = ["LIMEHPO", "TimeLIME"]
SQAPLANNERS_FILE_NAME = [
    "SQAPlanner_confidence",
    "SQAPlanner_coverage",
    "SQAPlanner_lift",
]
DISPLAY_NAME = {
    "DeFlip": "DeFlip",
    "DeFlip_actionable": "DeFlip (actionable)",
    "LIMEHPO": "LIME-HPO",
    "SQAPlanner_confidence": "SQAPlanner",
    "SQAPlanner_coverage": "SQAPlanner",
    "SQAPlanner_lift": "SQAPlanner",
    "TimeLIME": "TimeLIME",
}

DISPLAY_NAME_2 = {
    "DeFlip": "DeFlip",
    "LIMEHPO": "LIME-HPO",
    "SQAPlanner_confidence": "SQAPlanner(conf.)",
    "SQAPlanner_coverage": "SQAPlanner(cove.)",
    "SQAPlanner_lift": "SQAPlanner(lift)",
    "TimeLIME": "TimeLIME",
}

GET_STRATEGY = {
    "LIMEHPO": None,
    "SQAPlanner_coverage": "coverage",
    "TimeLIME": None,
    "SQAPlanner_confidence": "confidence",
    "SQAPlanner_lift": "lift",
}


def mean_accuracy_project(project, explainer):
    plan_path = Path(PROPOSED_CHANGES) / project / explainer / "plans_all.json"
    with open(plan_path) as f:
        plans = json.load(f)
    projects = read_dataset()
    train, test = projects[project]
    test_instances = test.drop(columns=["target"])

    # read the flipped instances
    exp_path = Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
    flipped_instances = pd.read_csv(exp_path, index_col=0)
    flipped_instances = flipped_instances.dropna()
    if len(flipped_instances) == 0:
        return None
    results = []
    for index, flipped in flipped_instances.iterrows():
        current = test_instances.loc[index]
        changed_features = list(plans[str(index)].keys())
        diff = current != flipped
        diff = diff[diff == True]

        score = mean_accuracy_instance(
            current[changed_features], flipped[changed_features], plans[str(index)]
        )

        if score is None:
            continue

        results.append(score)
    # median of results
    results_np = np.array(results)
    return {
        "median": np.median(results_np),
        "mean": np.mean(results_np),
        "std": np.std(results_np),
        "min": np.min(results_np),
        "max": np.max(results_np),
    }


def compute_score(a1, a2, b1, b2, is_int):
    intersection_start = max(a1, b1)
    intersection_end = min(a2, b2)

    if intersection_start > intersection_end:
        return 1.0  # No intersection

    if is_int:
        intersection_cnt = intersection_end - intersection_start + 1
        union_cnt = (a2 - a1 + 1) + (b2 - b1 + 1) - intersection_cnt
        score = 1 - (intersection_cnt / union_cnt)
    else:
        intersection_length = intersection_end - intersection_start
        union_length = (a2 - a1) + (b2 - b1) - intersection_length
        if union_length == 0:
            return 0.0  # Identical intervals
        score = 1 - (intersection_length / union_length)

    return score


def mean_accuracy_instance(current: pd.Series, flipped: pd.Series, plans):
    scores = []
    for feature in plans:
        flipped_changed = flipped[feature] - current[feature]
        if flipped_changed == 0.0:
            continue

        min_val = min(plans[feature])
        max_val = max(plans[feature])

        a1, a2 = (
            (min_val, flipped[feature])
            if current[feature] < flipped[feature]
            else (flipped[feature], max_val)
        )

        score = compute_score(
            min_val, max_val, a1, a2, current[feature].dtype == "int64"
        )
        assert 0 <= score <= 1, f"Invalid score {score} for feature {feature}"
        scores.append(score)

    return np.mean(scores)


def set_plotting_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})


def annotate_mean(ax, df, explainer_order, strategy_order=None, offset_order=[]):
    i = 0
    for explainer in df["Explainer"].unique():
        means = df.groupby(["Explainer"])["Value"].mean()
        mean_val = means[explainer]
        x_pos = explainer_order.index(explainer)

        if strategy_order:
            for strategy in strategy_order:
                filtered_df = df[df["Strategy"] == strategy]
                mean_val = filtered_df["Value"].mean()
                x_pos = explainer_order.index(explainer)

                # Calculate the offset for each hue
                num_strategies = len(strategy_order)
                width = 0.9  # The width of the boxes in the boxplot
                offset = width / num_strategies
                strategy_index = strategy_order.index(strategy)
                x_pos = (
                    x_pos + (strategy_index - num_strategies / 2) * offset + offset / 2
                )

                ax.annotate(
                    f"{mean_val:.2f}",
                    xy=(x_pos, mean_val),
                    xytext=(0, offset_order[i]),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="white",
                    fontweight="bold",
                )
                i += 1
        else:
            ax.annotate(
                f"{mean_val:.2f}",
                xy=(x_pos, mean_val),
                xytext=(0, offset_order[i]),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=15,
                color="white",
                fontweight="bold",
            )
            i += 1


def annotate_median(ax, df, explainer_order, strategy_order=None, offset_order=[]):
    i = 0
    for explainer in df["Explainer"].unique():
        medians = df.groupby(["Explainer"])["Value"].median()
        median_val = medians[explainer]
        x_pos = explainer_order.index(explainer)

        if strategy_order:
            for strategy in strategy_order:
                filtered_df = df[df["Strategy"] == strategy]
                median_val = filtered_df["Value"].median()
                x_pos = explainer_order.index(explainer)

                # Calculate the offset for each hue
                num_strategies = len(strategy_order)
                width = 0.9  # The width of the boxes in the boxplot
                offset = width / num_strategies
                strategy_index = strategy_order.index(strategy)
                x_pos = (
                    x_pos + (strategy_index - num_strategies / 2) * offset + offset / 2
                )

                ax.annotate(
                    f"{median_val:.2f}",
                    xy=(x_pos, median_val),
                    xytext=(0, offset_order[i]),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="white",
                    fontweight="bold",
                )
                i += 1
        else:
            ax.annotate(
                f"{median_val:.2f}",
                xy=(x_pos, median_val),
                xytext=(0, offset_order[i]),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=15,
                color="white",
                fontweight="bold",
            )
            i += 1


# %% RQ1 ##############################################################################################################


def get_rq1(ex):
    rq1 = []
    for explainer in OTHERS_FILE_NAME + SQAPLANNERS_FILE_NAME:
        result = pd.read_csv(
            Path(RESULTS) / f"{explainer}{'' if ex == 1 else '_all'}.csv", index_col=0
        )
        for value in result["Flip_Rate"].values:
            rq1.append(
                {
                    "Explainer": DISPLAY_NAME[explainer],
                    "Value": value,
                    "Strategy": GET_STRATEGY[explainer],
                }
            )
    return pd.DataFrame(rq1)


rq1_ex1 = get_rq1(1)
rq1_ex2 = get_rq1(2)

# %% Plot RQ1 ##############################################################################################################
fix, axs = plt.subplots(1, 2, figsize=(15, 6))

set_plotting_style()
others_df = rq1_ex1[rq1_ex1["Explainer"].isin(OTHERS_NAME)]
sqa_df = rq1_ex1[rq1_ex1["Explainer"] == "SQAPlanner"]

# plt.subplots(figsize=(15, 6))
ax = axs[1]
sns.boxplot(
    x="Explainer",
    y="Value",
    data=others_df,
    dodge=True,
    ax=ax,
    width=0.7,
    palette="rocket",
    order=EXPLAINER_ORDER,
)

sns.boxplot(
    x="Explainer",
    y="Value",
    hue="Strategy",
    data=sqa_df,
    ax=ax,
    width=0.9,
    palette="crest",
    dodge=True,
    order=EXPLAINER_ORDER,
    hue_order=STRATEGY_ORDER,
)
annotate_median(
    ax, others_df, EXPLAINER_ORDER, strategy_order=None, offset_order=[10, 10, 10]
)
annotate_median(ax, sqa_df, EXPLAINER_ORDER, STRATEGY_ORDER, offset_order=[10, 10, 10])
ax.set_xlabel("(b)", fontsize=22, fontweight="bold", labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_ylabel("")
ax.tick_params(
    axis="y", which="both", left=False, labelleft=False, right=False, labelright=False
)
ax.legend("")
sns.despine(left=True, right=True, bottom=True, top=True)
# plt.tight_layout()
# plt.savefig(Path(PLOTS) / 'Fig5b.svg', format='svg')
# plt.show()

others_df = rq1_ex2[rq1_ex2["Explainer"].isin(OTHERS_NAME)]
sqa_df = rq1_ex2[rq1_ex2["Explainer"] == "SQAPlanner"]

ax2 = axs[0]
sns.boxplot(
    x="Explainer",
    y="Value",
    data=others_df,
    width=0.7,
    ax=ax2,
    palette="rocket",
    order=EXPLAINER_ORDER,
)

sns.boxplot(
    x="Explainer",
    y="Value",
    hue="Strategy",
    data=sqa_df,
    ax=ax2,
    width=0.9,
    palette="crest",
    dodge=True,
    order=EXPLAINER_ORDER,
    hue_order=STRATEGY_ORDER,
)
annotate_median(
    ax2,
    others_df,
    EXPLAINER_ORDER,
    offset_order=[
        15,
        -10,
    ],
)
annotate_median(ax2, sqa_df, EXPLAINER_ORDER, STRATEGY_ORDER, offset_order=[10, 10, 10])

ax2.set_ylabel(
    "Flip Rate", fontsize=22, fontweight="bold", labelpad=10, fontfamily="sans-serif"
)

ax2.set_xlabel(
    "(a)", fontsize=22, fontweight="bold", labelpad=10, fontfamily="sans-serif"
)
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=20)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=15)

ax2.legend(
    loc="upper left",
    bbox_to_anchor=(0.05, 0.4),
    fontsize=17,
    frameon=True,
    title_fontsize=17,
).set_title("Search Strategy")
ax2.get_legend().get_title().set_fontweight("bold")
sns.despine(left=True, right=True, bottom=True, top=True)

plt.tight_layout()
plt.savefig(Path(PLOTS) / "Fig5.svg", format="svg")
plt.show()

# %% RQ2 ##############################################################################################################
projects = read_dataset()
rq2 = []
for explainer in OTHERS_FILE_NAME + ["SQAPlanner_confidence"]:
    for project in projects:
        release_ratio = get_release_ratio(project)
        score = mean_accuracy_project(project, explainer)
        rq2.append(
            {
                "Project": project,
                "Release_Ratio": release_ratio,
                "Explainer": DISPLAY_NAME[explainer],
                "Value": score["mean"] if score else np.nan,
            }
        )
# %%
rq2_df = pd.DataFrame(rq2)
# rq2_df = rq2_df.sort_values(by="Project")
rq2_df = rq2_df.sort_values(by="Release_Ratio")
rq2_df = rq2_df.drop("Release_Ratio", axis=1)
rq2_df
# %% Plot RQ2 ##############################################################################################################

# %%
set_plotting_style()
plt.figure(figsize=(15, 10))

ax = sns.lineplot(
    x="Project",
    y="Value",
    hue="Explainer",
    data=rq2_df,
    palette="rocket",
    marker="o",
    markersize=10,
    hue_order=EXPLAINER_ORDER,
)

colors = sns.color_palette("rocket", len(EXPLAINER_ORDER))

ax.set_xlabel("")
ax.set_ylabel("Mean Accuracy", fontsize=22, fontweight="bold", labelpad=10)

plt.ylim(0, 1)
ax.set_xticklabels(
    [get_release_names(project.get_text()) for project in ax.get_xticklabels()],
    fontsize=12,
    rotation=45,
    ha="right",
)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)

legend = ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.05, 0.25),
    fontsize=17,
    frameon=True,
    title_fontsize=17,
)
legend.set_title("  Explainer (median)")
legend.get_title().set_fontweight("bold")

medians = rq2_df.groupby(["Explainer"])["Value"].median()
medians = medians.reindex(EXPLAINER_ORDER)

for i, text in enumerate(legend.get_texts()):
    text.set_text(text.get_text() + f" ({medians.iloc[i]:.2f})")

sns.despine(left=True, right=True, bottom=True, top=True, offset=10, trim=True)

plt.tight_layout()
plt.savefig(Path(PLOTS) / "Fig6.svg", format="svg")
plt.show()
# %%


# %% RQ3 ##############################################################################################################
projects = read_dataset()
project_list = list(sorted(projects.keys()))
total_feasibilities = []

for explainer in ALL_EXPLAINERS:
    for project in project_list:
        train, test = projects[project]
        test_instances = test.drop(columns=["target"])
        historical_mean_changes = load_historical_changes(project)["mean_change"]
        exp_path = (
            Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
            if explainer not in ["DeFlip", "DeFlip_actionable"]
            else Path(EXPERIMENTS) / project / f"{explainer}.csv"
        )
        flipped_instances = pd.read_csv(exp_path, index_col=0)
        flipped_instances = flipped_instances.dropna()
        if len(flipped_instances) == 0:
            continue

        project_feasibilities = []
        for index, flipped in flipped_instances.iterrows():
            current = test_instances.loc[index]
            diff = current != flipped
            diff = diff[diff == True]
            changed_features = diff.index.tolist()

            feasibilites = []
            for feature in changed_features:
                if not historical_mean_changes[feature]:
                    historical_mean_changes[feature] = 0
                flipping_proposed_change = abs(flipped[feature] - current[feature])

                feasibility = 1 - (
                    flipping_proposed_change
                    / (flipping_proposed_change + historical_mean_changes[feature])
                )
                feasibilites.append(feasibility)
            feasibility = np.mean(feasibilites)
            project_feasibilities.append(feasibility)
        feasibility = np.mean(project_feasibilities)
        total_feasibilities.append(
            {
                "Explainer": DISPLAY_NAME[explainer],
                "Value": feasibility,
                "Project": project,
            }
        )

# %%
mean_feasibilities = pd.DataFrame(total_feasibilities)
mean_feasibilities = mean_feasibilities.sort_values(by="Project")
mean_feasibilities

# %% Plot RQ3 ##############################################################################################################
rq3_df = mean_feasibilities.copy()
rq3_df = rq3_df[rq3_df["Explainer"].isin(OTHERS_NAME + ["SQAPlanner"])]
plt.figure(figsize=(8, 6))
set_plotting_style()
ax = sns.boxplot(
    x="Explainer",
    y="Value",
    data=rq3_df,
    palette="rocket",
    width=0.7,
    order=EXPLAINER_ORDER,
)
ax.set_xlabel("")
ax.set_ylabel("Feasibility", fontsize=22, fontweight="bold", labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
annotate_median(ax, rq3_df, EXPLAINER_ORDER, offset_order=[10, 10, 10])

sns.despine(left=True, right=True, bottom=True, top=True)
plt.tight_layout()
plt.savefig(Path(PLOTS) / "Fig7_RQ3.svg", format="svg")
plt.show()
# %%
rq4_flip_rate = []
for explainer in ALL_EXPLAINERS[:2]:
    deflip_flip_rates = pd.read_csv(Path(RESULTS) / f"{explainer}.csv", index_col=0)
    deflip_flip_rates = deflip_flip_rates["Flip_Rate"].sort_index()
    for project in deflip_flip_rates.index:
        rq4_flip_rate.append({
            "Explainer": DISPLAY_NAME[explainer],
            "Value": deflip_flip_rates[project],
            "Project": project
        })


rq4_flip_rate_df = pd.DataFrame(rq4_flip_rate)
rq4_flip_rate_df = rq4_flip_rate_df.sort_values(by="Project")
rq4_flip_rate_df
# %% Plot RQ4 
DEFLIP_ORDER = ['DeFlip', 'DeFlip (actionable)']
rq4_feasibility = mean_feasibilities.copy()
rq4_feasibility = rq4_feasibility[rq4_feasibility["Explainer"].isin(DEFLIP_ORDER)]
rq4_feasibility = rq4_feasibility.dropna()

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
set_plotting_style()
sns.boxplot(
    x="Explainer",
    y="Value",
    data=rq4_feasibility,
    palette="viridis",
    width=0.7,
    order=DEFLIP_ORDER,
    ax = axs[1]
)
axs[1].set_xlabel("")
axs[1].set_ylabel("Feasibility", fontsize=22, fontweight="bold", labelpad=10)
axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=20)
axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=20)
annotate_median(axs[1], rq4_feasibility, DEFLIP_ORDER, offset_order=[10, 10, 10])
axs[1].set_xlabel(
    "(b)", fontsize=22, fontweight="bold", labelpad=10, fontfamily="sans-serif"
)

sns.boxplot(x="Explainer", y="Value", data=rq4_flip_rate_df, palette="viridis", width=0.7, ax=axs[0])
axs[0].set_xlabel("")
axs[0].set_ylabel("Flip Rate", fontsize=22, fontweight="bold", labelpad=10)
axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=20)
axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=20)
annotate_median(axs[0], rq4_flip_rate_df, DEFLIP_ORDER, offset_order=[-9, -9])
axs[0].set_xlabel(
    "(a)", fontsize=22, fontweight="bold", labelpad=10, fontfamily="sans-serif"
)
sns.despine(left=True, right=True, bottom=True, top=True)
plt.tight_layout()
plt.savefig(Path(PLOTS) / "Discussion.svg", format="svg")
plt.show()







# %% #### Mann Whitney U Test ##############################################################################################################

deflip_feasibility_df = feasible_rq4_df[feasible_rq4_df["Explainer"] == "DeFlip"]
deflip_feasibility_df = deflip_feasibility_df.sort_values(by="Project")
deflip_feasibility_df = deflip_feasibility_df.drop("Explainer", axis=1)
deflip_feasibility_df.set_index("Project", inplace=True)
deflip_feasibility_df = deflip_feasibility_df.dropna()
deflip_feasibility_df

limehpo_feasibility_df = feasible_rq4_df[feasible_rq4_df["Explainer"] == "LIME-HPO"]
limehpo_feasibility_df = limehpo_feasibility_df.sort_values(by="Project")
limehpo_feasibility_df = limehpo_feasibility_df.drop("Explainer", axis=1)
limehpo_feasibility_df.set_index("Project", inplace=True)
limehpo_feasibility_df = limehpo_feasibility_df.dropna()
limehpo_feasibility_df

from scipy.stats import mannwhitneyu

stat, p = mannwhitneyu(deflip_feasibility_df.values, limehpo_feasibility_df.values)
print("Statistics=%.3f, p=%.3f" % (stat, p))

# %% #### Mann Whitney U Test ##############################################################################################################

projects = read_dataset()

timelime_accuracy = []
for project in projects:
    score = mean_accuracy_project(project, "SQAPlanner_confidence")
    if score:
        timelime_accuracy.append(score["mean"])

print(np.median(timelime_accuracy))
stat, p = mannwhitneyu([1.0] * len(timelime_accuracy), timelime_accuracy)
print("Statistics=%.3f, p=%.3f" % (stat, p))

# %% #### Mann Whitney U Test ##############################################################################################################
explainer = "DeFlip"
deflip_flip_rates = pd.read_csv(Path(RESULTS) / f"{explainer}.csv", index_col=0)
deflip_flip_rates = deflip_flip_rates["Flip_Rate"].sort_index()
deflip_flip_rates

sqaplanner_mpc_flip_rates = pd.read_csv(
    Path(RESULTS) / f"SQAPlanner_confidence.csv", index_col=0
)
sqaplanner_mpc_flip_rates = sqaplanner_mpc_flip_rates["Flip_Rate"].sort_index()
sqaplanner_mpc_flip_rates

stat, p = mannwhitneyu(deflip_flip_rates.values, sqaplanner_mpc_flip_rates.values)
print("Statistics=%.3f, p=%.3f" % (stat, p))

# %% #### Mann Whitney U Test ##############################################################################################################
limehpo_fpc_flip_rates = pd.read_csv(Path(RESULTS) / f"LIMEHPO_all.csv", index_col=0)
limehpo_fpc_flip_rates = limehpo_fpc_flip_rates["Flip_Rate"].sort_index()
limehpo_fpc_flip_rates

stat, p = mannwhitneyu(deflip_flip_rates.values, limehpo_fpc_flip_rates.values)
print("Statistics=%.3f, p=%.3f" % (stat, p))
# %%

projects = read_dataset()
total_cosimilarities = []
for project in projects:
    model_path = Path(f"{MODELS}/{project}/RandomForest.pkl")
    train, test = projects[project]
    true_positives = get_true_positives(model_path, test)

    for test_name_1 in true_positives.index:
        for test_name_2 in true_positives.index:
            if test_name_1 == test_name_2:
                continue
            test_instance_1 = true_positives.loc[test_name_1]
            test_instance_2 = true_positives.loc[test_name_2]
            test_instance_1 = test_instance_1.values.reshape(1, -1)
            test_instance_2 = test_instance_2.values.reshape(1, -1)
            similarity = cosine_similarity(test_instance_1, test_instance_2)[0][0]
            total_cosimilarities.append(similarity)
    
# %%
np.mean(total_cosimilarities)
# %%
