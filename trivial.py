# Output/*/LIMEHPO_ 에 있는 모든 csv 파일을 읽어서, 각각 DataFrame으로 만들고, importance_ratio를 DataFrame마다 더한 값을 모든 파일에 대해 구한다.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import json

project = "activemq@0"
explainer = "LIMEHPO"
plan_path = f"plans/{project}/{explainer}/plans_all.json"
with open(plan_path, "r") as f:
    plan = json.load(f)
df = pd.read_csv(f"experiments/{project}/{explainer}_all.csv", index_col=0)
set(list(plan.keys())) - set(df.index.tolist())
 
# %%
plan
# %%
