# Output/*/LIMEHPO_ 에 있는 모든 csv 파일을 읽어서, 각각 DataFrame으로 만들고, importance_ratio를 DataFrame마다 더한 값을 모든 파일에 대해 구한다.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import json
from hyparams import PLANS, EXPERIMENTS
from data_utils import read_dataset

def left_instances(project, explainer):
    plan_path = Path(PLANS) / project / explainer / "plans_all.json"
    exp_path = Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
    with open(plan_path, "r") as f:
        plan = json.load(f)
    exp = pd.read_csv(exp_path, index_col=0)
    left_indices= list(set(plan.keys()) - set(exp.index.tolist()))
    return left_indices

EXPLAINERS = ["LIMEHPO", "TimeLIME", "SQAPlanner_coverage", "SQAPlanner_confidence", "SQAPlanner_lift"]
 
# %%
projects = read_dataset()
for project in projects:
    train, test = projects[project]
    for explainer in EXPLAINERS:
        print(explainer)
        print(left_instances(project, explainer))


# %%
