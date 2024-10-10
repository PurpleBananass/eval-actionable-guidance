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
    
    # sort the data by flip rate
    df = df.sort_values(by='Flip Rate', ascending=False)

    # Setting the pastel color palette and Times New Roman font
    sns.set_palette("pastel")
    plt.rcParams['font.family'] = 'Times New Roman'

    explainers = ['LIME', 'LIME-HPO','TimeLIME', 'SQAPlanner', "All"]
    # Creating a grouped bar plot
    plt.figure(figsize=(4, 6))
    ax = sns.barplot(y='Model', x='Flip Rate', hue='Explainer', data=df, edgecolor='0.2', hue_order=explainers)

    for i, container in enumerate(ax.containers):
        print(i)
        for bar in container:
            bar.set_height(bar.get_height() * 0.8)
            bar.set_edgecolor('black')
            bar.set_linewidth(0.4)
            
            # add y tick labels
            text = plt.text(-0.02, bar.get_y() + bar.get_height() / 2, f'{explainers[i]}', va='center', ha='right', fontsize=10)


            # add alpha value to the bar
            if i != 4:
                bar.set_alpha(0.4)
            else:
                text.set_fontweight('bold')

            

    # Customizing the plot
    plt.xlim(0, 1)
    
    # move y-axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.ylabel('')
    plt.xlabel('Change Rate', fontsize=12)
    # disable legend
    plt.legend([],[], frameon=False)
    plt.xticks(fontsize=10, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    plt.yticks(fontsize=10)

    plt.tight_layout()
    # Display the plot
    plt.savefig("./evaluations/flip_rate_plot.png", dpi=300)

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
    # list_status(model_type="RandomForest")
    # list_status(model_type="XGBoost")
    visualize_rq1()
            
            