# explainer들의 proposed changes에 속한 룰 중, max 값으로 증가시켜야하는 피쳐들의 분포를 그린다.
import json

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from data_utils import read_dataset
from hyparams import PLANS, PLOTS

def relative_change_rate(old_value, new_value):
    return (new_value - old_value) / old_value * 100

MAX_PERCENTAGE = 9000

plot_path = Path(PLOTS)
plot_path.mkdir(parents=True, exist_ok=True)
projects = read_dataset()
project_list = list(sorted(projects.keys()))
baselines = ['LIMEHPO', 'TimeLIME', 'SQAPlanner/confidence', 'SQAPlanner/coverage', 'SQAPlanner/lift']

max_statistics = {}
statistics = {}
count_statistics = {}
for explainer in baselines:
    max_statistics[explainer] = max_statistics.get(explainer, {})
    statistics[explainer] = statistics.get(explainer, {})
    count_statistics[explainer] = count_statistics.get(explainer, {})
    for project in project_list:
        plan_path = Path(PLANS) / project / explainer / 'plans_all.json'
        with open(plan_path, 'r') as f:
            plans = json.load(f)

        train, test = projects[project]
        test_instances = test.drop(columns=['target'])
        mins = train.min()
        maxs = train.max()

        for instance_name in plans.keys():
            plan = plans[instance_name]
            origin = test_instances.loc[int(instance_name)]
            for feature in plan.keys():
                if origin[feature] == 0:
                    statistics[explainer][feature] = statistics[explainer].get(feature, 0) + 1
                elif relative_change_rate(origin[feature], max(plan[feature])) > MAX_PERCENTAGE:
                    max_statistics[explainer][feature] = max_statistics[explainer].get(feature, 0) + 1
                count_statistics[explainer][feature] = count_statistics[explainer].get(feature, 0) + 1

for explainer in baselines:
    for feature in count_statistics[explainer].keys():
        if feature not in statistics[explainer]:
            statistics[explainer][feature] = 0
        if feature not in max_statistics[explainer]:
            max_statistics[explainer][feature] = 0
        statistics[explainer][feature] = statistics[explainer][feature]

# Max count

for explainer in baselines:
    df_count = pd.DataFrame(count_statistics[explainer], index=['count']).T
    df_count = df_count.sort_values(by='count', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(12, 12))
    
    sns.barplot(x=df_count.index.values, y='count', data=df_count, ax=ax1, label='count')
    plt.xticks(rotation=90)
    ax1.set_ylim([0, 1.1 * df_count['count'].max()])
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    df_zero_count = pd.DataFrame(statistics[explainer], index=['zero_count']).T
    
    df_zero_count = df_zero_count.loc[df_count.index]  # Match the order of features
    df_zero_count = df_zero_count.sort_values(by='zero_count', ascending=False)

    sns.barplot(x=df_zero_count.index.values, y='zero_count', data=df_zero_count, ax=ax2, label='zero_count', color='red', width=0.4)
    for p in ax2.patches:
        p.set_alpha(0.5)
        x = p.get_x() - 0.2
        p.set_x(x)
    plt.xticks(rotation=90)
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_ylim(ax1.get_ylim())
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax2.set_ylabel('')

    df_max_count = pd.DataFrame(max_statistics[explainer], index=['max_count']).T
    
    df_max_count = df_max_count.loc[df_count.index]  # Match the order of features
    df_max_count = df_max_count.sort_values(by='max_count', ascending=False)

    sns.barplot(x=df_max_count.index.values, y='max_count', data=df_max_count, ax=ax3, label='max_count', color='blue', width=0.4)
    for p in ax3.patches:
        p.set_alpha(0.5)
        x = p.get_x() + 0.2
        p.set_x(x)
    plt.xticks(rotation=90)
    ax3.set_yticks(ax1.get_yticks())
    ax3.set_ylim(ax1.get_ylim())
    ax3.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax3.set_ylabel('')

    # legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax1.legend(h1+h2+h3, l1+l2+l3, loc='upper right')

    plt.title(f'{explainer}: # of times each feature is changed when flipped and # of times original value was zero')
    plt.tight_layout()
    filename = explainer.replace('/', '_')
    plt.savefig(plot_path / f'over_plans_{filename}_{MAX_PERCENTAGE}%.png')
