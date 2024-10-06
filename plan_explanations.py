import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from math import ceil, floor
from data_utils import get_true_positives, load_model, read_dataset, get_output_dir, get_model_file
from hyparams import PROPOSED_CHANGES

# Aussme there are generated explanations
def run_single_project(train, test, project_name, model_type, explainer_type, search_strategy, only_minimum=False, verbose=False):
    model_path = get_model_file(project_name, model_type)
    output_path = get_output_dir(project_name, explainer_type, model_type)
    proposed_change_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}")
    if search_strategy is not None:
        proposed_change_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}_{search_strategy}")
        output_path = output_path / search_strategy
        output_path.mkdir(parents=True, exist_ok=True)
    proposed_change_path.mkdir(parents=True, exist_ok=True)

    file_name = "plans.json" if only_minimum else "plans_all.json"

    pattern = re.compile(r'([-]?[\d.]+)?\s*(<|>)?\s*([a-zA-Z_]+)\s*(<=|>=|<|>)?\s*([-]?[\d.]+)?')

    train_min = train.min()
    train_max = train.max()

    all_plans = {}

    true_positives = get_true_positives(model_path, train, test)

    for test_idx in tqdm(true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose):
        test_instance = test.loc[test_idx]
        assert test_instance["target"] == 1
        

        match explainer_type:
                
            case "LIME" | "LIME-HPO":
                explanation_path = output_path / f"{test_idx}.csv"
                
                if not explanation_path.exists():
                    print("????")
                    continue
                explanation = pd.read_csv(explanation_path)
                
                plan = []
                for row in range(len(explanation)):
                    feature,value,importance,min_val,max_val,rule,importance_ratio = explanation.iloc[row].values
                    proposed_changes = flip_feature_range(feature, min_val, max_val, importance, rule)
                    if proposed_changes:
                        plan.append(proposed_changes)

                perturb_features = {}
               
                for proposed_changes in plan:
                    feature = proposed_changes[1]
                    dtype = train.dtypes[feature]
                    perturbations = perturb_feature(proposed_changes[0], proposed_changes[2], test_instance[feature], dtype)
                    if not perturbations:
                        
                        continue
                    if only_minimum:
                        perturb_features[feature] = perturbations[0]
                    else:
                        perturb_features[feature] = perturbations
                
                

            case "TimeLIME":
                explanation_path = Path(f"{output_path}/{test_idx}.csv")

                if not explanation_path.exists():
                    print("????")
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
                    perturbations = perturb_feature(proposed_changes[0], proposed_changes[2],test_instance[feature], dtype)
                    if not perturbations:
                        print("why", feature, dtype, proposed_changes)
                        
                        continue
                    if only_minimum:
                        perturb_features[feature] = perturbations[0]
                    else:                        
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
                        if ranges[0] < train_min[feature]:
                            ranges[0] = max(0, train_min[feature])
                     
                        perturbations = perturb_feature(ranges[0], ranges[1], test_instance[feature], train.dtypes[feature])
                        if not perturbations:
                            continue
                        if only_minimum:
                            perturb_features[feature] = perturbations[0]
                        else:
                            perturb_features[feature] = perturbations

                    
        all_plans[int(test_idx)] = perturb_features
    
    def convert_int64(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    with open(proposed_change_path / file_name, "w") as f:
        json.dump(all_plans, f, indent=4, default=convert_int64)


def perturb_feature(low, high, current, dtype):
    
    if dtype == "int64":    
        if low == high:
            return [low]
        low = int(ceil(float(low)))
        high = int(floor(float(high)))
        step = 1
        perturbations = list(range(low, high + 1, step))
        
    elif dtype == "float64":
        if low == high:
            return [low]
        low = float(low)
        high = float(high)
        step = 0.05
    
        if low <= current <= high:
            perturbations = list(np.arange(current, low, -1 * step)) + list(np.arange(current, high + step, step))
        else:
            perturbations = list(np.arange(low, high + step, step))
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if current in perturbations:
        perturbations.remove(current)

    sorted_publications = sorted(perturbations, key=lambda x: abs(x - current))

    return sorted_publications
          
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
    print("Not Available", rule_str)
    return None
    
    raise ValueError(f"Unknown rule: {rule_str}")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--explainer_type", type=str, default="LIMEHPO")
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--search_strategy", type=str, default=None)
    parser.add_argument("--only_minimum", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    projects = read_dataset()  

    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split(" ")
    for project in tqdm(project_list, desc="Projects", leave=True, disable=not args.verbose):
        train, test = projects[project]
        print(project)
        run_single_project(train, test, project, args.model_type, args.explainer_type, args.search_strategy, args.only_minimum, args.verbose)
   