from argparse import ArgumentParser
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import json
from pathlib import Path
from data_utils import read_dataset 
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_rq1():
    # Data preparation
    df = pd.read_csv("./evaluations/flip_rates.csv")

    df = df.sort_values(by='Flip Rate', ascending=False)
    sns.set_palette("crest")
    plt.rcParams['font.family'] = 'Times New Roman'

    explainers = ['LIME', 'LIME-HPO','TimeLIME', 'SQAPlanner', "All"]
    # Creating a grouped bar plot
    plt.figure(figsize=(4, 3.5))
    ax = sns.barplot(y='Model', x='Flip Rate', hue='Explainer', data=df, edgecolor='0.2', hue_order=explainers)

    for i, container in enumerate(ax.containers):
        
        
        for bar in container:
            bar.set_height(bar.get_height() * 0.8)
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
            
            left_text = plt.text(-0.02, bar.get_y() + bar.get_height() / 2, f'{explainers[i]}', va='center', ha='right', fontsize=10)

            value_text = plt.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}', va='center', ha='right', fontsize=8)

            if i != 4:
                bar.set_alpha(0.4)
            else:
                left_text.set_fontweight('bold')
                value_text.set_color('white')
                value_text.set_fontweight('bold')

    # Customizing the plot
    ax.margins(x=0, y=0.01)
    plt.xlim(0, 1)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.ylabel('')
    plt.xlabel('')
    # disable legend
    plt.legend([],[], frameon=False)
    plt.xticks(fontsize=10, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    plt.yticks(fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./evaluations/flip_rate_plot.png", dpi=300)


def visualize_rq2():
    explainers = ['LIME', 'LIME-HPO', 'TimeLIME', 'SQAPlanner']
    models = {'RandomForest': 'RF ', 'XGBoost': 'XGB', 'SVM': 'SVM'}
    total_df = pd.DataFrame()
    plt.rcParams['font.family'] = 'Times New Roman'
    for model in models:
        for explainer in explainers:
            df = pd.read_csv(f"./evaluations/accuracy/{model}_{explainer}.csv")
            df['Model'] = model
            df['Explainer'] = explainer
            total_df = pd.concat([total_df, df], ignore_index=True)

    colors = sns.color_palette("crest", len(models))
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=models[model]) for i, model in enumerate(models)]

    # median values (for each model and explainer)
    median_df = total_df.groupby(['Model', 'Explainer']).median().reset_index()



    
    # Grouped histogram plot for each explainer (total 4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i, explainer in enumerate(explainers):
        ax = axes[i // 2, i % 2]
        df = total_df[total_df['Explainer'] == explainer]
        sns.histplot(data=df, x='score', hue='Model', multiple='dodge', ax=ax, palette='crest', stat='percent')
        ax.set_title(explainer)
        ax.set_xlabel('')
        ax.set_ylim(0, 32)
        if i % 2 == 0:
            ax.set_ylabel('Percentages (%)', rotation=90, labelpad=3)
        else:
            ax.set_ylabel('')
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        
        # ax.get_legend().set_title('Median')
        
        # # only text legend
        # ax.get_legend().set_frame_on(False)
        # for handle in ax.get_legend().legend_handles:
        #     handle.set_visible(False)
        

        # for text in ax.get_legend().texts:
        #     text.set_fontsize(10)
        #     model_name = models[text.get_text()]
        #     text.set_text( f'{model_name} {median_df[(median_df["Explainer"] == explainer) & (median_df["Model"] == text.get_text())]["score"].values[0]:.2f}')

        ax.get_legend().remove()
        # ax.set_yticks()
        # ax.legend(title='Model', loc='upper left')
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, title='', frameon=False)


    plt.tight_layout()
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 하단 여백 확보

    plt.savefig("./evaluations/accuracy_plot.png", dpi=300)


def visualize_rq3():
    explainers = ['LIME', 'LIME-HPO','TimeLIME', 'SQAPlanner']
    models = ['RandomForest', 'XGBoost']
    total_df = pd.DataFrame()
    for model in models:
        for explainer in explainers:
            df = pd.read_csv(f"./evaluations/feasibility/{model}_{explainer}.csv")
            df['Model'] = model
            df['Explainer'] = explainer
            total_df = pd.concat([total_df, df], ignore_index=True)

    # Grouped violin plot
    plt.figure(figsize=(8, 5))
    # df_melted = total_df.melt(id_vars=['Model', 'Explainer'], value_vars=['min', 'mean'], var_name='Dist', value_name='Value')
    # ax = sns.violinplot(x='Model', y='Value', hue='Explainer', data=df_melted, split=False, inner='quartile', density_norm='count', palette='crest')
    # print(total_df)
    test_df = total_df.loc[:, ['Model', 'Explainer', 'min']]
    # ax = sns.violinplot(x='Model', y='min', hue='Explainer', data=test_df, split=False, inner='quartile', palette='crest')
    plt.ylim(0, 1)
    ax = sns.boxplot(x='Model', y='min', hue='Explainer', data=test_df, palette='crest')
    # plt.ylabel('Value')
    # plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig("./evaluations/feasibility_plot.png", dpi=300)




def list_status(model_type="XGBoost", explainers=["TimeLIME", "LIME-HPO", "LIME", "SQAPlanner_confidence"]):
    projects = read_dataset()
    table = []
    headers = ["Project"] + [ exp[:8] for exp in explainers ] + ["common", "left"]
    total = 0
    total_left = 0
    projects = sorted(projects.keys())
    for project in projects:
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
        with open(plan_path, "r") as f:
            plans = json.load(f)
            total_names = set(plans.keys())
        # common names between explainers
        common_names = row[explainers[0]]
        for explainer in explainers[1:]:
            common_names = common_names.intersection(row[explainer])
        row["common"] = common_names
        row["total"] = total_names
        for explainer in explainers:
            table_row.append(len(row[explainer]))
        table_row.append(f"{len(common_names)}/{len(total_names)}")
        table_row.append(len(total_names)-len(common_names))
        table.append(table_row)
        total += len(common_names)
        total_left += len(total_names)-len(common_names)
    table.append(["Total"] + [""] * len(explainers) + [total, total_left])
    print(f"Model: {model_type}")
    print(tabulate(table, headers=headers))




if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--rq1", action="store_true")
    argparser.add_argument("--rq2", action="store_true")
    argparser.add_argument("--rq3", action="store_true")
    args = argparser.parse_args()
    # list_status(model_type="RandomForest")
    # list_status(model_type="XGBoost")
    # visualize_rq1()
    if args.rq1:
        visualize_rq1()
    if args.rq2:
        visualize_rq2()
    if args.rq3:
        visualize_rq3()
            
            