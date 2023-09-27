# %% Import packages ##############################################################################################################
import json
from matplotlib.text import Text
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import read_dataset, load_historical_changes, get_release_names
from hyparams import PLOTS, PLANS, EXPERIMENTS, RESULTS

# %%

EXPLAINER_ORDER = ["LIME-HPO", "TimeLIME",  "SQAPlanner"]
STRATEGY_ORDER = ["coverage", "confidence", "lift"]
OTHERS_NAME = ["LIME-HPO", "TimeLIME"]
OTHERS_FILE_NAME = ["LIMEHPO", "TimeLIME"]
SQAPLANNERS_FILE_NAME = [
    "SQAPlanner_confidence",
    "SQAPlanner_coverage",
    "SQAPlanner_lift",
]
DISPLAY_NAME = {
    "LIMEHPO": "LIME-HPO",
    "SQAPlanner_confidence": "SQAPlanner",
    "SQAPlanner_coverage": "SQAPlanner",
    "SQAPlanner_lift": "SQAPlanner",
    "TimeLIME": "TimeLIME",
}

DISPLAY_NAME_2 = {
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

def excess_rate_project(project, explainer):
    min_plan_path = Path(PLANS) / project / explainer / "plans.json"
    with open(min_plan_path) as f:
        min_plan = json.load(f)
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
        changed_features = list(min_plan[str(index)].keys())
        diff = current != flipped
        diff = diff[diff == True]

        excess_rates = excess_rate(
            current[changed_features], flipped[changed_features], min_plan[str(index)]
        )

        if excess_rates is None:
            continue

        results.append(excess_rates)
    # median of results
    results_np = np.array(results)
    return {
        'median': np.median(results_np),
        'mean': np.mean(results_np),
        'std': np.std(results_np),
        'min': np.min(results_np),
        'max': np.max(results_np)
    }


def excess_rate(current: pd.Series, flipped: pd.Series, min_change: dict):
    incorrectness = []
    for feature, value in min_change.items():
        flipped_changed = flipped[feature] - current[feature]
        if flipped_changed == 0.0:
            continue
        min_changed = value - current[feature]
        if min_changed == 0.0:
            continue

        score = abs(flipped_changed) / abs(min_changed)

        incorrectness.append(score)
    if len(incorrectness) == 0:
        return None
    return sum(incorrectness) / len(incorrectness)

# %%

def correctness_project(project, explainer):
    plan_path = Path(PLANS) / project / explainer / "plans_all.json"
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

        score = correctness(
            current[changed_features], flipped[changed_features], plans[str(index)]
        )

        if score is None:
            continue

        results.append(score)
    # median of results
    results_np = np.array(results)
    return {
        'median': np.median(results_np),
        'mean': np.mean(results_np),
        'std': np.std(results_np),
        'min': np.min(results_np),
        'max': np.max(results_np)
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

def correctness(current: pd.Series, flipped: pd.Series, plans):
    scores = []
    for feature in plans:
        flipped_changed = flipped[feature] - current[feature]
        if flipped_changed == 0.0 :
            continue
        
        min_val = min(plans[feature])
        max_val = max(plans[feature])

        a1, a2 = (min_val, flipped[feature]) if current[feature] < flipped[feature] else (flipped[feature], max_val)

        score = compute_score(min_val, max_val, a1, a2, current[feature].dtype == 'int64')
        assert 0 <= score <= 1, f"Invalid score {score} for feature {feature}"
        scores.append(score)
    
    return np.mean(scores)  

def set_plotting_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})

def annotate_median(ax, df, explainer_order, strategy_order=None, offset_order=[]):
    i = 0
    for explainer in df['Explainer'].unique():
        medians = df.groupby(['Explainer'])['Value'].median()
        median = medians[explainer]
        x_pos = explainer_order.index(explainer)

        if strategy_order:
            for strategy in strategy_order:
                filtered_df = df[df['Strategy'] == strategy]
                median = filtered_df['Value'].median()
                x_pos = explainer_order.index(explainer)

                # Calculate the offset for each hue
                num_strategies = len(strategy_order)
                width = 0.9  # The width of the boxes in the boxplot
                offset = width / num_strategies
                strategy_index = strategy_order.index(strategy)
                x_pos = x_pos + (strategy_index - num_strategies / 2) * offset + offset / 2

            
                ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, offset_order[i]), 
                            textcoords='offset points', ha='center', va='center', fontsize=15, color='white', fontweight='bold')
                i += 1
        else:
            ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, offset_order[i]), 
                        textcoords='offset points', ha='center', va='center', fontsize=15, color='white', fontweight='bold')
            i += 1

# %% RQ1 ##############################################################################################################

def get_rq1(ex):
    rq1 = []
    for explainer in OTHERS_FILE_NAME + SQAPLANNERS_FILE_NAME:
        result = pd.read_csv(Path(RESULTS) / f"{explainer}{'' if ex == 1 else '_all'}.csv", index_col=0)
        for value in result['Flip_Rate'].values:
            rq1.append({
                'Explainer': DISPLAY_NAME[explainer],
                'Value': value,
                'Strategy': GET_STRATEGY[explainer]
            })
    return pd.DataFrame(rq1)

rq1_ex1 = get_rq1(1)
rq1_ex2 = get_rq1(2)

# %%

# %% Plot RQ1 ##############################################################################################################
fix, axs = plt.subplots(1, 2, figsize=(15, 6))

set_plotting_style()
others_df = rq1_ex1[rq1_ex1['Explainer'].isin(OTHERS_NAME)]
sqa_df = rq1_ex1[rq1_ex1['Explainer'] == 'SQAPlanner']

# plt.subplots(figsize=(15, 6))
ax = axs[1]
sns.boxplot(x="Explainer", y="Value", data=others_df, dodge=True, ax=ax, width=0.7, palette="rocket", order=EXPLAINER_ORDER)

sns.boxplot(x="Explainer", y="Value", hue="Strategy", data=sqa_df, ax=ax, width=0.9, palette="crest", dodge=True, order=EXPLAINER_ORDER, hue_order=STRATEGY_ORDER)
annotate_median(ax, others_df, EXPLAINER_ORDER, strategy_order=None, offset_order=[10, 10, 10])
annotate_median(ax, sqa_df, EXPLAINER_ORDER, STRATEGY_ORDER, offset_order=[10, 10, 10])
ax.set_xlabel('(b)', fontsize=22, fontweight='bold', labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(),  fontsize=20)
ax.set_ylabel('')
ax.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
ax.legend('')
sns.despine(left=True, right=True, bottom=True, top=True)
# plt.tight_layout()
# plt.savefig(Path(PLOTS) / 'Fig5b.svg', format='svg')
# plt.show()

others_df = rq1_ex2[rq1_ex2['Explainer'].isin(OTHERS_NAME)]
sqa_df = rq1_ex2[rq1_ex2['Explainer'] == 'SQAPlanner']

ax2 = axs[0]
sns.boxplot(x="Explainer", y="Value", data=others_df, width=0.7, ax=ax2, palette="rocket", order=EXPLAINER_ORDER)

sns.boxplot(x="Explainer", y="Value", hue="Strategy", data=sqa_df, ax=ax2, width=0.9, palette="crest", dodge=True, order=EXPLAINER_ORDER, hue_order=STRATEGY_ORDER)
annotate_median(ax2, others_df, EXPLAINER_ORDER, offset_order=[15, -10, -10,])
annotate_median(ax2, sqa_df, EXPLAINER_ORDER, STRATEGY_ORDER, offset_order=[10, 10, 10])

ax2.set_ylabel('Flip Rate', fontsize=22, fontweight='bold', labelpad=10, fontfamily='sans-serif')

ax2.set_xlabel('(a)', fontsize=22, fontweight='bold', labelpad=10, fontfamily='sans-serif')
ax2.set_xticklabels(ax2.get_xticklabels(),  fontsize=20)
ax2.set_yticklabels(ax2.get_yticklabels(),  fontsize=15)

ax2.legend(loc='upper left', bbox_to_anchor=(0.05,0.4), fontsize=17, frameon=True, title_fontsize=17).set_title('Search Strategy')
ax2.get_legend().get_title().set_fontweight('bold')
sns.despine(left=True, right=True, bottom=True, top=True)

plt.tight_layout()
plt.savefig(Path(PLOTS) / 'Fig5.svg', format='svg')
plt.show()

# %% RQ2 ##############################################################################################################
projects = read_dataset()
rq2 = []
for explainer in OTHERS_FILE_NAME + ['SQAPlanner_confidence']:
    for project in projects:
        score = correctness_project(project, explainer)
        rq2.append({
            'Project': project,
            'Explainer': DISPLAY_NAME[explainer],
            'Value': score['mean'] if score else np.nan,
        })
# %%
rq2_df = pd.DataFrame(rq2)
rq2_df = rq2_df.sort_values(by='Project')
rq2_df

# %%

def DFC_MPC_project(project, explainer):
    min_plan_path = Path(PLANS) / project / explainer / "plans.json"
    with open(min_plan_path) as f:
        min_plan = json.load(f)
    projects = read_dataset()
    train, test = projects[project]
    min_values = train.min()
    max_values = train.max()

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
        changed_features = list(min_plan[str(index)].keys())
        diff = current != flipped
        diff = diff[diff == True]

        dfc, mpc = get_DFC_MPC(
            current[changed_features], flipped[changed_features], min_plan[str(index)], min_values, max_values
        )
        results.append({
            'Instance': index,
            'Explainer': DISPLAY_NAME[explainer],
            'DFC': dfc,
            'MPC': mpc
        })
    return results

def get_DFC_MPC(current: pd.Series, flipped: pd.Series, min_change: dict, min_values, max_values):
    dfcs = []
    mpcs = []
    for feature, value in min_change.items():
        def normalize(x):
            return (x - min_values[feature]) / (max_values[feature] - min_values[feature])
        flipped_changed = normalize(flipped[feature]) - normalize(current[feature])
        if flipped_changed == 0.0:
            continue
        min_changed = normalize(value) - normalize(current[feature])
        if min_changed == 0.0:
            continue

        dfcs.append(abs(flipped_changed))
        mpcs.append(abs(min_changed))
    if len(dfcs) == 0:
        return None
    return sum(dfcs) / len(dfcs), sum(mpcs) / len(mpcs)

# %%
# results_a = []
# for explainer in OTHERS_FILE_NAME + ['SQAPlanner_confidence']:
#     for project in projects:
#         result = DFC_MPC_project(project, explainer)
#         if result is None:
#             continue
#         results_a += result

# # %%
# results_a_df = pd.DataFrame(results_a)
# # results_a_df.set_index('Instance', inplace=True)
# results_a_df['Instance'] = results_a_df['Instance'].astype(str) 
# results_a_df = results_a_df.sort_values(by='MPC')
# results_a_df
# %% Plot RQ2 ##############################################################################################################

# %%
set_plotting_style()
plt.figure(figsize=(15, 10))

# # Line plot (x: project, y: median excess rate, hue: explainer)
ax = sns.lineplot(x="Project", y="Value", hue="Explainer", data=rq2_df, palette="rocket", marker='o', markersize=10, hue_order=EXPLAINER_ORDER)


colors = sns.color_palette("rocket", len(EXPLAINER_ORDER))

ax.set_xlabel('')
ax.set_ylabel('Mean Accuracy', fontsize=22, fontweight='bold', labelpad=10)

plt.ylim(0, 1)
# ax.set_yticks([1, 20, 40, 60, 80, 100, 120])
ax.set_xticklabels([ get_release_names(project.get_text()) for project in ax.get_xticklabels()],  fontsize=20, rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)

legend = ax.legend(loc='upper left', bbox_to_anchor=(0.73, 0.26), fontsize=17, frameon=True, title_fontsize=17)
legend.set_title('Explainer (Mean)')
legend.get_title().set_fontweight('bold')

means = rq2_df.groupby(['Explainer'])['Value'].mean()

for i, text in enumerate(legend.get_texts()):
    text.set_text(text.get_text() + f" ({means.iloc[i]:.2f})")

sns.despine(left=True, right=True, bottom=True, top=True, offset=10, trim=True)

# plot_a = axs[0]
# set_plotting_style()
# # line plot1 (x: Instance, y: DFC, hue: explainer)
# # line plot2 (x: Instance, y: MPC, hue: explainer)

# limehpo = results_a_df[results_a_df['Explainer'] == 'LIME-HPO']
# timelime = results_a_df[results_a_df['Explainer'] == 'TimeLIME']
# sqa = results_a_df[results_a_df['Explainer'] == 'SQAPlanner']

# sns.lineplot(x="Instance", y="MPC", data=limehpo, color=colors[0], ax=plot_a, zorder=3)
# # sns.lineplot(x="Instance", y="MPC", data=timelime, color=colors[1], marker='o', markersize=10, ax=plot_a)
# # sns.lineplot(x="Instance", y="MPC", data=sqa, color=colors[2], marker='o', markersize=10, ax=plot_a)

# sns.lineplot(x="Instance", y="DFC", data=limehpo, color=colors[1], ax=plot_a, zorder=2)
# # sns.lineplot(x="Instance", y="DFC", data=timelime, color=colors[1], marker='o', markersize=10, ax=plot_a)
# # sns.lineplot(x="Instance", y="DFC", data=sqa, color=colors[2], marker='o', markersize=10, ax=plot_a)

# plot_a.set_xlabel('(a)', fontsize=22, fontweight='bold', labelpad=10)
# plot_a.set_ylabel('Median Value of DFC and MPC', fontsize=22, fontweight='bold', labelpad=10)
# plot_a.set_xticklabels('')
# plot_a.set_yticklabels('')
plt.tight_layout()
plt.savefig(Path(PLOTS) / 'Fig6.svg', format='svg')
plt.show()
# %%


# %% RQ3 ##############################################################################################################
projects = read_dataset()
rq3 = []
project_list = list(sorted(projects.keys()))
for explainer in OTHERS_FILE_NAME + ['SQAPlanner_confidence']:
    for project in project_list:
        train, test = projects[project]
        test_instances = test.drop(columns=["target"])
        historical_mean_changes = load_historical_changes(project)["mean_change"]
        exp_path = (
            Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
            if explainer != "DeFlip"
            else Path(EXPERIMENTS) / project / f"{explainer}.csv"
        )
        flipped_instances = pd.read_csv(exp_path, index_col=0)
        flipped_instances = flipped_instances.dropna()
        if len(flipped_instances) == 0:
            continue
        results = []
        for index, flipped in flipped_instances.iterrows():
            current = test_instances.loc[index]
            diff = current != flipped
            diff = diff[diff == True]
            changed_features = diff.index.tolist()
            historical_mean_change = historical_mean_changes[changed_features]
            DFC = flipped[changed_features] - current[changed_features]
            DFC = DFC.abs()
            demand_rates = DFC / historical_mean_change
            demand_rate = demand_rates.mean()
            results.append(demand_rate)
        if len(results) == 0:
            continue
        rq3.append({
            'Project': project,
            'Explainer': DISPLAY_NAME[explainer],
            'Value': np.median(results),
        })

# %%
rq3_df = pd.DataFrame(rq3)
rq3_df = rq3_df.sort_values(by='Project')
rq3_df.loc[rq3_df['Value'].isna(), 'Value'] = np.nan
rq3_df
# %% Plot RQ3 ##############################################################################################################
set_plotting_style()
plt.figure(figsize=(15, 10))

ax = sns.lineplot(x="Project", y="Value", hue="Explainer", data=rq3_df, palette="rocket", marker='o', markersize=10, hue_order=EXPLAINER_ORDER)
colors = sns.color_palette("rocket", 3)
# Median value of median demand rates for each explainer
median_demand_rates = rq3_df.groupby(['Explainer'])['Value'].median()
median_x = []
for explainer in EXPLAINER_ORDER:
    median = median_demand_rates[explainer]
    # find the x position of the median value
    explainer_df = rq3_df.loc[rq3_df['Explainer'] == explainer]
    median_locs = explainer_df.loc[explainer_df['Value'] == median, "Project"]
    median_loc = median_locs.values[0]
    median_x.append(median_loc)

# Plot median value of median demand rates for each project
for i in range(len(EXPLAINER_ORDER)):
    sns.scatterplot(x=[median_x[i]], y=[median_demand_rates.values[i]], ax=ax, color=colors[i], s=100, marker='x', linewidth=5, zorder=10)

 # LIME-HPO, TimeLIME, SQAPlanner
offsets = [-40, -40, -40]
for i, explainer in enumerate(median_demand_rates.index):
    median = median_demand_rates[explainer]
    x_pos = median_x[i]
    ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, offsets[i]), 
                    textcoords='offset points', ha='center', va='center', fontsize=20, color=colors[i], fontweight='bold')


ax.set_xlabel('')
ax.set_ylabel('Median Value of All Demand Rates', fontsize=22, fontweight='bold', labelpad=10)

ax.set_xticklabels([ get_release_names(project.get_text()) for project in ax.get_xticklabels()],  fontsize=20, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)

ax.legend(loc='upper left', bbox_to_anchor=(0.8, 0.97), fontsize=17, frameon=True, title_fontsize=17).set_title('Explainer')
ax.get_legend().get_title().set_fontweight('bold')
sns.despine(left=True, right=True, bottom=True, top=True)

plt.tight_layout()
plt.savefig(Path(PLOTS) / 'Fig7.svg', format='svg')
plt.show()

# %% RQ4 ##############################################################################################################

project_list = list(sorted(projects.keys()))
explainer = "DeFlip"
rq4 = []
for project in project_list:
    train, test = projects[project]
    test_instances = test.drop(columns=["target"])
    historical_mean_changes = load_historical_changes(project)["mean_change"]
    exp_path = Path(EXPERIMENTS) / project / f"{explainer}.csv"
    
    flipped_instances = pd.read_csv(exp_path, index_col=0)
    flipped_instances = flipped_instances.dropna()
    if len(flipped_instances) == 0:
        continue
    results = []
    for index, flipped in flipped_instances.iterrows():
        current = test_instances.loc[index]

        diff = current != flipped
        diff = diff[diff == True]
        changed_features = diff.index.tolist()
        historical_mean_change = historical_mean_changes[changed_features]
        DFC = flipped[changed_features] - current[changed_features]
        DFC = DFC.abs()
        demand_rates = DFC / historical_mean_change
        demand_rate = demand_rates.mean()
        results.append(demand_rate)
    if len(results) == 0:
        continue
    rq4.append({
        'Project': project,
        'Value': np.median(results),
    })

# median of median demand rates for each project
rq4_df = pd.DataFrame(rq4)
median_demand_rate = rq4_df['Value'].median()
median_demand_rate

# %%
