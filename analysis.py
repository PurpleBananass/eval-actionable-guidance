import pandas as pd
import json
from pathlib import Path
from data_utils import read_dataset 
from tabulate import tabulate

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
    list_status(model_type="RandomForest")
    list_status(model_type="XGBoost")
            
            