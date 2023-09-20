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
def run_single_project(train, test, project_name, model_type, explainer_type, search_strategy, only_minimum=True, verbose=False):
    model_path = Path(f"{MODELS}/{project_name}/{model_type}.pkl") 
    output_path = Path(f"{OUTPUT}/{project_name}/{explainer_type}")
    plans_path = Path(f"{PLANS}/{project_name}/{explainer_type}")
    if search_strategy is not None:
        plans_path = plans_path / search_strategy
        output_path = output_path / search_strategy
    output_path.mkdir(parents=True, exist_ok=True)
    plans_path.mkdir(parents=True, exist_ok=True)

    file_name = "plans.json" if only_minimum else "plans_all.json"

    if Path(plans_path / file_name).exists():
        if verbose:
            print(f"Skipping {project_name}")
        return

    pattern = re.compile(r'([-]?[\d.]+)?\s*(<|>)?\s*([a-zA-Z_]+)\s*(<=|>=|<|>)?\s*([-]?[\d.]+)?')

    train_min = train.min()
    train_max = train.max()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(test.loc[:, test.columns != "target"].values)

    all_plans = {}
    for i in tqdm(range(len(test)), desc=f"{project_name}", leave=False, disable=not verbose):
        test_instance = test.iloc[i, :]
        test_idx = test_instance.name
        if test_instance["target"] == 0 or predictions[i] == 0:
            continue

        match explainer_type:
            case "LIMEHPO" | "LIME":
                explanation_path = Path(f"{output_path}/{test_idx}.csv")
                if not explanation_path.exists():
                    continue
                explanation = pd.read_csv(explanation_path)
        
                plan = []
                for row in range(len(explanation)):
                    feature, value, importance, min_val, max_val, rule = explanation.iloc[row].values
                    proposed_changes = flip_feature_range(feature, min_val, max_val, importance, rule)
                    plan.append(proposed_changes)

                perturb_features = {}
                for proposed_changes in plan:
                    feature = proposed_changes[1]
                    dtype = train.dtypes[feature]
                    perturbations = perturb_feature(proposed_changes[0], proposed_changes[2], test_instance[feature], dtype, only_minimum)
                    perturb_features[feature] = perturbations

            case "TimeLIME":
                explanation_path = Path(f"{output_path}/{test_idx}.csv")
                if not explanation_path.exists():
                    continue
                explanation = pd.read_csv(explanation_path)

                plan = []
                for row in range(len(explanation)):
                    feature, value, importance, left, right, rec, rule, min_val, max_val = explanation.iloc[row].values
                    plan.append([left, feature, right])

                perturb_features = {}
                for proposed_changes in plan:
                    feature = proposed_changes[1]
                    dtype = train.dtypes[feature]
                    perturbations = perturb_feature(proposed_changes[0], proposed_changes[2],test_instance[feature], dtype, only_minimum)
                    perturb_features[feature] = perturbations

            case "SQAPlanner":
                try:
                    plan = pd.read_csv(output_path / f"{test_idx}.csv")
                except pd.errors.EmptyDataError:
                    if verbose:
                        print(f"EmptyDataError: {project_name} {test_idx}")
                    continue

                if len(plan) == 0:
                    continue
            
                perturb_features = {}
                for _, row in plan.iterrows():
                    if len(perturb_features) > 0:
                        break
                    best_rule = row["Antecedent"]
                    for rule in best_rule.split('&'):
                        feature, ranges = split_inequality(rule, train_min, train_max, pattern)
                        if ranges[0] > ranges[1]:
                            break
                        perturbations = perturb_feature(*ranges, test_instance[feature], train.dtypes[feature], only_minimum)
                        
                        perturb_features[feature] = perturbations
                    
        all_plans[int(test_idx)] = perturb_features

    
    with open(plans_path / file_name, "w") as f:
        json.dump(all_plans, f, indent=4)

def perturb_feature(low, high, current, dtype, only_minimum=False):
    if dtype == "int64":    
        low = int(ceil(float(low)))
        high = int(floor(float(high)))
        step = 1
        perturbations = list(range(low, high + 1, step))
        
    elif dtype == "float64":
        low = float(low)
        high = float(high)
        step = 0.05
        if low <= current <= high:
            perturbations = list(np.arange(current, low, -1 * step)) + list(np.arange(current, high + step, step))
        else:
            perturbations = list(np.arange(low, high + step, step))

    if perturbations == [current]:
        pass
    elif current in perturbations and only_minimum:
        perturbations.remove(current)

    sorted_publications = sorted(perturbations, key=lambda x: abs(x - current))

    return sorted_publications[0] if only_minimum else sorted_publications

def split_inequality(rule, min_val, max_val, pattern): 
    match pattern.search(rule).groups():
        case v1, "<", feature_name, "<=", v2:
            l, r = float(v1), float(v2)
        case None, None, feature_name, ">", v1:
            l, r = float(v1), max_val[feature_name]
        case None, None, feature_name, "<=", v2:
            l, r = min_val[feature_name], float(v2)
    return feature_name, [l, r]
        
def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    # Case: a < feature <= b
    matches = re.search(r'([\d.]+) < ' + re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if matches:
        a, b = map(float, matches.groups())
        if importance > 0:
            return [min_val, feature, a] 
        return [b, feature, max_val]
    
    # Case: feature > a
    matches = re.search(re.escape(feature) + r' > ([\d.]+)', rule_str)
    if matches:
        a = float(matches.group(1))
        return [min_val, feature, a]
    
    # Case: feature <= b
    matches = re.search(re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if matches:
        b = float(matches.group(1))
        return [b, feature, max_val]
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--explainer_type", type=str, default="LIMEHPO")
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--search_strategy", type=str, default=None)
    parser.add_argument("--only_minimum", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.project == "all":
        projects = read_dataset()
        for project in tqdm(projects, desc="Projects", leave=True, disable=not args.verbose):
            train, test = projects[project]
            run_single_project(train, test, project, args.model_type, args.explainer_type, args.search_strategy, args.only_minimum, args.verbose)
    else:
        projects = read_dataset()
        train, test = projects[args.project]
        run_single_project(train, test, args.project, args.model_type, args.explainer_type, args.search_strategy, args.only_minimum, args.verbose)