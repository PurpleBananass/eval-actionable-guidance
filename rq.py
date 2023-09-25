# %% Imports
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import read_dataset, load_historical_changes

from hyparams import PLOTS, PLANS, EXPERIMENTS  
# %% RQ1
ex1 = [
    './results/DeFlip.csv',
    './results/LIMEHPO.csv', 
    './results/TimeLIME.csv', 
    './results/SQAPlanner_confidence.csv', 
    './results/SQAPlanner_coverage.csv', 
    './results/SQAPlanner_lift.csv'
]

ex2 = [
    './results/LIMEHPO_all.csv', 
    './results/TimeLIME_all.csv', 
    './results/SQAPlanner_confidence_all.csv', 
    './results/SQAPlanner_coverage_all.csv', 
    './results/SQAPlanner_lift_all.csv'
]

def plot_rq1(results, fname):
    total_others = {}
    others = results[:2]
    sqap = results[2:]
    for result in others:
        result = Path(result)
        df = pd.read_csv(result, index_col=0)
        df['Flip_Rate'] = df['Flip'] / df['TP']
        name = result.stem.replace('_all', '')
        total_others[name] = df['Flip_Rate'].values
    
    project_list = list(df.index)
    total_others_df = pd.DataFrame(total_others, index=project_list)
    total_others_df = total_others_df.melt(var_name='Explainer', value_name='Flip_Rate', ignore_index=False)
    
    sns.set_theme(style="whitegrid") 
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
    
    # Plotting the others
    ax = sns.boxplot(x="Explainer", y="Flip_Rate", data=total_others_df, width=0.3, palette="rocket")
    
    total_sqa = []
    for result in sqap:
        result = Path(result)
        df = pd.read_csv(result, index_col=0)
        df['Flip_Rate'] = df['Flip'] / df['TP']
        name = result.stem.replace('_all', '')
        sqa, strategy = name.split('_')
        df['Explainer'] = sqa
        df['Strategy'] = strategy
        total_sqa.append(df)

    total_sqa_df = pd.concat(total_sqa)

    # Order of the Explainers and Strategies
    explainer_order = total_others_df['Explainer'].unique().tolist() + total_sqa_df['Explainer'].unique().tolist()
    strategy_order = total_sqa_df['Strategy'].unique().tolist()
    
    # Plotting the sqa
    sns.boxplot(x="Explainer", y="Flip_Rate", hue="Strategy", data=total_sqa_df, ax=ax, width=0.9, 
                palette="crest", dodge=True, order=explainer_order, hue_order=strategy_order)
    
    # Annotating median values for 'others'
    for explainer in total_others_df['Explainer'].unique():
        medians = total_others_df.groupby(['Explainer'])['Flip_Rate'].median()
        median = medians[explainer]
        x_pos = explainer_order.index(explainer)
        ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, -5.5), 
                    textcoords='offset points', ha='center', va='center', fontsize=12, color='white')

    # Adding median points for 'sqa'
    for explainer in total_sqa_df['Explainer'].unique():
        for strategy in strategy_order:
            filtered_df = total_sqa_df[(total_sqa_df['Explainer'] == explainer) & (total_sqa_df['Strategy'] == strategy)]
            median = filtered_df['Flip_Rate'].median()
            x_pos = explainer_order.index(explainer)
            
            # Calculate the offset for each hue
            num_strategies = len(strategy_order)
            width = 0.9  # The width of the boxes in the boxplot
            offset = width / num_strategies
            strategy_index = strategy_order.index(strategy)
            x_pos = x_pos + (strategy_index - num_strategies / 2) * offset + offset / 2

            ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, -6), 
                        textcoords='offset points', ha='center', va='center', fontsize=12, color='white')
    
    ax.set_xlabel('Explainer', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Flip Rate', fontsize=12, fontweight='bold', labelpad=10)

    # Adjusting the legend location
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    
    # title of legend
    leg = ax.get_legend()
    leg.set_title('Search Strategy')

    # Removing the spines
    sns.despine(left=True, right=True, bottom=True, top=True)
    
    plt.tight_layout()
    plt.savefig(Path(PLOTS, fname), bbox_inches='tight', dpi=300, transparent=True, format='svg')
    plt.show()
    plt.close()

plot_rq1(ex1, 'rq1_ex1_tp.svg')
plot_rq1(ex2, 'rq1_ex2_tp.svg')
# %% RQ2
# Calculating the  flipped P.C. / min. P.C. ratio, called 'incorrectness' 
# incorrectness per one plan is average of incorrectness of consisting features
# incorrectness per one project is average of incorrectness of all plans
# incorrectness per one explainer is average of incorrectness of all projects

def incorrectness_per_project(project, explainer):
    min_plan_path = Path(PLANS) / project / explainer / 'plans.json'
    with open(min_plan_path) as f:
        min_plan = json.load(f)
    projects = read_dataset()
    train, test = projects[project]
    test_instances = test.drop(columns=['target'])

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
        try:
            assert set(diff.index.tolist()).issubset(set(changed_features))
        except AssertionError:
            print(project, explainer, index)
            print(diff.index.tolist())
            print(changed_features)

            raise ValueError

        incorrectness_per_plan = incorrectness(current[changed_features], flipped[changed_features], min_plan[str(index)])
        
        if incorrectness_per_plan is None:
            continue

        results.append(incorrectness_per_plan)
    # median of results
    results_np = np.array(results)
    return np.median(results_np)


def incorrectness(current: pd.Series, flipped: pd.Series, min_change: dict):
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


import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from hyparams import PLOTS  # Assuming PLOTS is correctly imported from hyparams
# %%
projects = read_dataset()
baselines = ['LIMEHPO', 'TimeLIME', 'SQAPlanner_confidence', 'SQAPlanner_coverage', 'SQAPlanner_lift']
data = []
for explainer in baselines:
    for project in projects:
        score = incorrectness_per_project(project, explainer)
        if score is None:
            continue
        # Parse explainer and strategy
        if 'SQAPlanner' in explainer:
            sqa, strategy = explainer.split('_')
            data.append({'Explainer': sqa, 'Strategy': strategy, 'Ratio': score})
        else:
            data.append({'Explainer': explainer, 'Strategy': None, 'Ratio': score})

results_df = pd.DataFrame(data)
results_df.to_csv('./results/rq2.csv')
# %%
def plot_rq2(results_df):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
    
    # Define the order and separate the data
    explainer_order = ['LIMEHPO', 'TimeLIME','SQAPlanner']
    strategy_order = ['confidence', 'coverage', 'lift']
    others = ['LIMEHPO', 'TimeLIME']
    sqa = ['SQAPlanner_confidence', 'SQAPlanner_coverage', 'SQAPlanner_lift']
    
    others_df = results_df[results_df['Explainer'].isin(others)]
    sqa_df = results_df[results_df['Explainer'] == 'SQAPlanner']
    # fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting the others
    ax = sns.boxplot(x="Explainer", y="Ratio", data=others_df, width=0.4, palette="rocket")
    
    # Plotting the sqa
    sns.boxplot(x="Explainer", y="Ratio", hue="Strategy", data=sqa_df, ax=ax, width=0.9, palette="crest", dodge=True, order=explainer_order, hue_order=strategy_order)
    
    # Annotating median values for 'others'
    for explainer in others_df['Explainer'].unique():
        medians = others_df.groupby(['Explainer'])['Ratio'].median()
        median = medians[explainer]
        x_pos = explainer_order.index(explainer)
        ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, +7), 
                    textcoords='offset points', ha='center', va='center', fontsize=12, color='white')

    # Adding median points for 'sqa'
    for explainer in sqa_df['Explainer'].unique():
        for strategy in strategy_order:
            filtered_df = sqa_df[(sqa_df['Explainer'] == explainer) & (sqa_df['Strategy'] == strategy)]
            median = filtered_df['Ratio'].median()
            x_pos = explainer_order.index(explainer)
            
            # Calculate the offset for each hue
            num_strategies = len(strategy_order)
            width = 0.9  # The width of the boxes in the boxplot
            offset = width / num_strategies
            strategy_index = strategy_order.index(strategy)
            x_pos = x_pos + (strategy_index - num_strategies / 2) * offset + offset / 2

            ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, +7), 
                        textcoords='offset points', ha='center', va='center', fontsize=12, color='black')
    
    ax.set_xlabel('Explainer', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Ratio', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0, 20)

    # Adjusting the legend location
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    leg = ax.get_legend()
    leg.set_title('Search Strategy')

    plt.tight_layout()
    plt.savefig(Path(PLOTS) / 'rq2.svg', bbox_inches='tight', dpi=300, transparent=True, format='svg')
    plt.show()

# %%Call the function
plot_rq2(results_df)
# %% RQ3
projects = read_dataset()
baselines = ['LIMEHPO', 'TimeLIME', 'SQAPlanner_confidence', 'SQAPlanner_coverage', 'SQAPlanner_lift']
data = []
for explainer in baselines:
    for project in projects:
        train, test = projects[project]
        test_instances = test.drop(columns=['target'])
        historical_mean_changes = load_historical_changes(project)['mean_change']
        exp_path = Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
        flipped_instances = pd.read_csv(exp_path, index_col=0)
        flipped_instances = flipped_instances.dropna()
        if len(flipped_instances) == 0:
            continue
        scores = []
        for index, flipped in flipped_instances.iterrows():
            current = test_instances.loc[index]
            diff = current != flipped
            diff = diff[diff == True]
            changed_features = diff.index.tolist()
            historical_mean_change = historical_mean_changes.loc[changed_features]
            DFC = flipped[changed_features] - current[changed_features]
            DFC = DFC.abs()
            DFC = DFC / historical_mean_change
            score = DFC.mean()
            scores.append(score)
        score = np.median(np.array(scores))

        if 'SQAPlanner' in explainer:
            sqa, strategy = explainer.split('_')
            data.append({'Explainer': sqa, 'Strategy': strategy, 'Ratio': score})
        else:
            data.append({'Explainer': explainer, 'Strategy': None, 'Ratio': score})
results_df = pd.DataFrame(data)
results_df.to_csv('./results/rq3.csv')

# %%
def plot_rq3(results_df):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
    
    # Define the order and separate the data
    explainer_order = ['LIMEHPO', 'TimeLIME','SQAPlanner']
    strategy_order = ['confidence', 'coverage', 'lift']
    others = ['LIMEHPO', 'TimeLIME']
    sqa = ['SQAPlanner_confidence', 'SQAPlanner_coverage', 'SQAPlanner_lift']
    
    others_df = results_df[results_df['Explainer'].isin(others)]
    sqa_df = results_df[results_df['Explainer'] == 'SQAPlanner']
    # fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting the others
    ax = sns.boxplot(x="Explainer", y="Ratio", data=others_df, width=0.4, palette="rocket")
    
    # Plotting the sqa
    sns.boxplot(x="Explainer", y="Ratio", hue="Strategy", data=sqa_df, ax=ax, width=0.9, palette="crest", dodge=True, order=explainer_order, hue_order=strategy_order)
    
    # Annotating median values for 'others'
    for explainer in others_df['Explainer'].unique():
        medians = others_df.groupby(['Explainer'])['Ratio'].median()
        median = medians[explainer]
        x_pos = explainer_order.index(explainer)
        ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, +7), 
                    textcoords='offset points', ha='center', va='center', fontsize=12, color='white')

    # Adding median points for 'sqa'
    for explainer in sqa_df['Explainer'].unique():
        for strategy in strategy_order:
            filtered_df = sqa_df[(sqa_df['Explainer'] == explainer) & (sqa_df['Strategy'] == strategy)]
            median = filtered_df['Ratio'].median()
            x_pos = explainer_order.index(explainer)
            
            # Calculate the offset for each hue
            num_strategies = len(strategy_order)
            width = 0.9  # The width of the boxes in the boxplot
            offset = width / num_strategies
            strategy_index = strategy_order.index(strategy)
            x_pos = x_pos + (strategy_index - num_strategies / 2) * offset + offset / 2

            ax.annotate(f'{median:.2f}', xy=(x_pos, median), xytext=(0, +6), 
                        textcoords='offset points', ha='center', va='center', fontsize=12, color='white')
    
    ax.set_xlabel('Explainer', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Ratio', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0, 15)

    # Adjusting the legend location
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    leg = ax.get_legend()
    leg.set_title('Search Strategy')

    plt.tight_layout()
    plt.savefig(Path(PLOTS) / 'rq.svg', bbox_inches='tight', dpi=300, transparent=True, format='svg')
    plt.show()

# %%
plot_rq3(results_df)
# %% RQ4 (DeFlip vs Winners of each RQ1~3)
