# %%
from itertools import product
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from data_utils import read_dataset

from hyparams import MODELS, PLANS, SEED

PROJ = 'derby@0'
INSTANCE = 108
np.random.seed(SEED)
projects = read_dataset()
train, test = projects[PROJ]

query = test.loc[INSTANCE, test.columns != "target"].to_frame().T

model_path = Path(f"{MODELS}/{PROJ}/RandomForest.pkl") 
with open(model_path, "rb") as f:
    model = pickle.load(f)

assert model.predict(query.values) == test.loc[INSTANCE, "target"]

explainers = ['LIMEHPO', 'TimeLIME', 'SQAPlanner']
plan_paths = [ Path(f"{PLANS}/{PROJ}/{explainer}/plans.json") for explainer in explainers ]

plan_q = []
for plan_path in plan_paths:
    with open(plan_path, "r") as f:
        plans = json.load(f)
    if isinstance(plans)
    plan_q.append(plans[str(INSTANCE)])
        
# %%
def find_counterfactual_all_features(instance: pd.Series, top_features, training_data: pd.DataFrame, classifier):

    for values in product(*changeable_ranges):
        modified_instance = instance.copy()
        for feature, value in zip(top_features, values):
            modified_instance[feature] = value

        new_pred = classifier.predict_proba(modified_instance.values.reshape(1, -1))[:, 0]
        if new_pred >= 0.5:
            counterfactual = modified_instance
            break

    return counterfactual