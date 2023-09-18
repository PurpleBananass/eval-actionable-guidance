# %%
# Load plans

import json
from pathlib import Path
import pickle

import numpy as np

import pandas as pd
from data_utils import read_dataset

from hyparams import MODELS, OUTPUT, PLANS, SEED


def run_single_project(train, test, project_name, model_type, explainer_type):
    model_path = Path(f"{MODELS}/{project_name}/{model_type}.pkl") 
    output_path = Path(f"{OUTPUT}/{project_name}/{explainer_type}")
    plans_path = Path(f"{PLANS}/{project_name}/{explainer_type}")
    output_path.mkdir(parents=True, exist_ok=True)
    plans_path.mkdir(parents=True, exist_ok=True)

    

    if not (plans_path / "plans.json").exists():
        return
    
    with open(plans_path / "plans.json", "r") as f:
        plans = json.load(f)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    for test_idx in plans:
        test_instance = test.loc[int(test_idx)]
        result = []
        for plan in plans[test_idx]:
            for feature in plan:
                cur_value = test_instance[feature]
                ranges = plan[feature]
                ranges = list(filter(lambda x: x != cur_value, ranges))
                print(f"{feature} : {cur_value} -> {ranges}")
        break

        

            
# %%
# DICE

import dice_ml

np.random.seed(SEED)
projects = read_dataset()
project = 'derby@0'
train, test = projects[project]
target_instance = 108
test_instance = test.loc[target_instance, test.columns != "target"]
test_instance

# %%
model_path = Path(f"{MODELS}/{project}/RandomForest.pkl") 
with open(model_path, "rb") as f:
    model = pickle.load(f)

train_x = train.loc[:, train.columns != "target"]

d = dice_ml.Data(dataframe=train, continuous_features=list(train_x.columns), outcome_name='target')

m = dice_ml.Model(model=model, backend="sklearn")
exp = dice_ml.Dice(d, m)
# %%
# Test 108 is predicted as 1
model.predict(test_instance.values.reshape(1, -1))
# %%
query = test.loc[108, test.columns != 'target']
query = query.to_frame().T
dice_exp = exp.generate_counterfactuals(query, total_CFs=100, desired_class="opposite")
dice_exp.visualize_as_dataframe(show_only_changes=True)
final = dice_exp.cf_examples_list[0].final_cfs_df
final = pd.concat([final, query ], axis=0)
final.to_csv('dice100.csv')
# %%
