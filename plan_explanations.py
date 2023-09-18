import json
import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from math import ceil, floor
from data_utils import read_dataset
from hyparams import MODELS, OUTPUT, PLANS

# Aussme there are generated explanations
def run_single_project(train, test, project_name, model_type, explainer_type):
    model_path = Path(f"{MODELS}/{project_name}/{model_type}.pkl") 
    output_path = Path(f"{OUTPUT}/{project_name}/{explainer_type}")
    plans_path = Path(f"{PLANS}/{project_name}/{explainer_type}")
    output_path.mkdir(parents=True, exist_ok=True)
    plans_path.mkdir(parents=True, exist_ok=True)

    train_min = train.min()
    train_max = train.max()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(test.loc[:, test.columns != "target"].values)

    all_plans = {}
    for i in tqdm(range(len(test)), desc=f"{project_name}", leave=False):
        test_instance = test.iloc[i, :]
        test_idx = test_instance.name
        if test_instance["target"] == 0 or predictions[i] == 0:
            continue

        if explainer_type == "LIMEHPO" or explainer_type == "LIME":
            explanation_path = Path(f"{output_path}/{test_idx}.csv")
            if not explanation_path.exists():
                continue
            explanation = pd.read_csv(explanation_path)
      
            plan = []
            for row in range(len(explanation)):
                feature, value, importance, min_val, max_val, rule = explanation.iloc[row].values
                proposed_changes = flip_feature_range(feature, min_val, max_val, importance, rule)
                plan.append(proposed_changes)

            top_perturb_features = []
            perturb_features = {}
            for proposed_changes in plan:
                if isinstance(proposed_changes[0], list):
                    assert proposed_changes[0][1] == proposed_changes[1][1]
                    feature = proposed_changes[0][1]
                    dtype = train.dtypes[feature]
                    perturbations = perturb_feature(proposed_changes[0][0], proposed_changes[0][2], dtype) + perturb_feature(proposed_changes[1][0], proposed_changes[1][2], dtype)
                    perturbations = list(set(perturbations))
                    perturbations = sorted(perturbations, key=lambda x: abs(x - test_instance[feature]))
                    perturb_features[feature] = perturbations[:20]
                else:
                    assert len(proposed_changes) == 3
                    feature = proposed_changes[1]
                    dtype = train.dtypes[feature]
                    perturbations = perturb_feature(proposed_changes[0], proposed_changes[2], dtype)
                    perturbations = sorted(perturbations, key=lambda x: abs(x - test_instance[feature]))
                    perturb_features[feature] = perturbations[:20]
            top_perturb_features.append(perturb_features)

        elif explainer_type == "TimeLIME":
            explanation_path = Path(f"{output_path}/{test_idx}.csv")
            if not explanation_path.exists():
                continue
            explanation = pd.read_csv(explanation_path)

            plan = []
            for row in range(len(explanation)):
                feature, value, importance, left, right, rec, rule, min_val, max_val = explanation.iloc[row].values
                plan.append([left, feature, right])

            top_perturb_features = []
            perturb_features = {}
            for proposed_changes in plan:
                assert len(proposed_changes) == 3
                feature = proposed_changes[1]
                dtype = train.dtypes[feature]
                perturbations = perturb_feature(proposed_changes[0], proposed_changes[2], dtype)
                perturbations = sorted(perturbations, key=lambda x: abs(x - test_instance[feature]))
                perturb_features[feature] = perturbations[:20]
            top_perturb_features.append(perturb_features)

        elif explainer_type == "SQAPlanner":
            try:
                confidience_plan = pd.read_csv(plans_path / f"confidence/{test_idx}.csv")
                coverage_plan = pd.read_csv(plans_path / f"coverage/{test_idx}.csv")
            except pd.errors.EmptyDataError:
                continue
            if len(confidience_plan) == 0 or len(coverage_plan) == 0:
                continue
         
            top_plans = [
                split_inequality(confidience_plan.loc[:, "Antecedent"][0]),
                split_inequality(coverage_plan.loc[:, "Antecedent"][0])
            ]
            
            for proposed_changes in top_plans:
                if isinstance(proposed_changes[0], list):
                    for sub_plan in proposed_changes:
                        if sub_plan[0] is None:
                            sub_plan[0] = train_min[sub_plan[1]]
                        if sub_plan[2] is None:
                            sub_plan[2] = train_max[sub_plan[1]]
                else:
                    if proposed_changes[0] is None:
                        proposed_changes[0] = train_min[proposed_changes[1]]
                    if proposed_changes[2] is None:
                        proposed_changes[2] = train_max[proposed_changes[1]]

            top_perturb_features = []
            for plan in top_plans:
                perturb_features = {}
                if not isinstance(plan[0], list):
                    feature = plan[1]
                    dtype = train.dtypes[feature]
                    perturbations = perturb_feature(plan[0], plan[2], dtype)
                    perturbations = sorted(perturbations, key=lambda x: abs(x - test_instance[feature]))
                    perturb_features[feature] = perturbations[:20]
                else:
                    for sub_plan in plan:
                        feature = sub_plan[1]
                        dtype = train.dtypes[feature]
                        perturbations = perturb_feature(sub_plan[0], sub_plan[2], dtype)
                        perturbations = sorted(perturbations, key=lambda x: abs(x - test_instance[feature]))
                        perturb_features[feature] = perturbations[:20]
                top_perturb_features.append(perturb_features)
        
        all_plans[int(test_idx)] = top_perturb_features

    # Save top perturb features per test instance into a json file
    with open(plans_path / "plans.json", "w") as f:
        json.dump(all_plans, f, indent=4)

def perturb_feature(low, high, dtype):
    if dtype == "int64":    
        low = int(ceil(float(low)))
        high = int(floor(float(high)))
        step = 1
        perturbations = list(range(low, high + 1, step))
        
    elif dtype == "float64":
        low = float(low)
        high = float(high)
        step = 0.05
        perturbations = list(np.arange(low, high + step, step))

    return perturbations

def split_inequality(rule): 
    if '&' in rule:
        return [split_inequality(r) for r in rule.split('&')]    
     
    pattern = re.compile(r'([-]?[\d.]+)?\s*(<|>)?\s*([a-zA-Z_]+)\s*(<=|>=|<|>)?\s*([-]?[\d.]+)?')
    match = pattern.search(rule)
    if match is None:
        print(rule)
    v1, c1, feature, c2, v2 = match.groups()
    if v1 is None:
        if c2 == '>':
            return [v2, feature, None]
        elif c2 == '<=':
            return [None, feature, v2]
        else:
            return [None, feature, None]
    else:
        return [v1, feature, v2]
        
def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    # Case: a < feature <= b
    match = re.search(r'([\d.]+) < ' + re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if match:
        a, b = map(float, match.groups())
        return [[min_val, feature, a], [b, feature, max_val]]
    
    # Case: feature > a
    match = re.search(re.escape(feature) + r' > ([\d.]+)', rule_str)
    if match:
        a = float(match.group(1))
        return [min_val, feature, a]
    
    # Case: feature <= b
    match = re.search(re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if match:
        b = float(match.group(1))
        return [b, feature, max_val]

def run_all_project(model_type, explainer_type):
    projects = read_dataset()
    for project in tqdm(projects, desc="Projects", leave=True):
        train, test = projects[project]
        run_single_project(train, test, project, model_type, explainer_type)



    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--explainer_type", type=str, default="LIMEHPO")
    parser.add_argument("--project", type=str, default="all")
    args = parser.parse_args()

    if args.project == "all":
        run_all_project(args.model_type, args.explainer_type)
    else:
        projects = read_dataset()
        train, test = projects[args.project]
        run_single_project(train, test, args.project, args.model_type, args.explainer_type)