# Output/*/LIMEHPO_ 에 있는 모든 csv 파일을 읽어서, 각각 DataFrame으로 만들고, importance_ratio를 DataFrame마다 더한 값을 모든 파일에 대해 구한다.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import json
from hyparams import PLANS, EXPERIMENTS
from data_utils import load_historical_changes, read_dataset
# %%
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
    if "Added_lines" in train.columns:
        print(train["Added_lines"].max())
        


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 샘플 데이터 생성
import pandas as pd
import numpy as np

np.random.seed(0)
df = pd.DataFrame({
    'time': np.tile(np.arange(10), 3),
    'value': np.random.randn(30),
    'group': np.repeat(['A', 'B', 'C'], 10)
})

# lineplot 그리기
plt.figure(figsize=(10,6))
sns.lineplot(x='time', y='value', hue='group', data=df)

plt.show()

# %%
projects = read_dataset()
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
# %%
