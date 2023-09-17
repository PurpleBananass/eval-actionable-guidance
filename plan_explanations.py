import pickle
import re
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

from data_utils import read_dataset
from TimeLIME import extract_name_from_condition
from hyparams import MODELS, OUTPUT, PLANS

# Aussme there are generated explanations
def run_single_project(train, test, project_name, model_type, explainer_type):
    model_path = Path(f"{MODELS}/{project_name}/{model_type}.pkl") 
    output_path = Path(f"{OUTPUT}/{project_name}/{explainer_type}")
    plans_path = Path(f"{PLANS}/{project_name}/{explainer_type}")
    output_path.mkdir(parents=True, exist_ok=True)
    plans_path.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(test.loc[:, test.columns != "target"].values)
    for i in range(len(test)):
        test_instance = test.iloc[i, :]
        test_idx = test_instance.name
        if test_instance["target"] == 0 or predictions[i] == 0:
            continue

        # Load the explanation
        explanation_path = Path(f"{output_path}/{test_idx}.csv")
        if not explanation_path.exists():
            continue

        
        print(f"Test instance: {test_idx}")
        if explainer_type == "LIMEHPO" or explainer_type == "LIME":
            explanation = pd.read_csv(explanation_path)
            top_plans = []
            for row in range(len(explanation)):
                feature, value, importance, min_val, max_val, rule = explanation.iloc[row].values
                str_plan, plan = flip_feature_range(feature, min_val, max_val, importance, rule)
                top_plans.append(plan)
                

        elif explainer_type == "TimeLIME":
            explanation = pd.read_csv(explanation_path)
            top_plans = []
            for row in range(len(explanation)):
                feature, value, importance, left, right, rec, rule, min_val, max_val = explanation.iloc[row].values
                str_plan = f'{left} <= {feature} <= {right}'
                top_plans.append(str_plan)
                
        elif explainer_type == "SQAPlanner":
            confidience_plan = pd.read_csv(plans_path / "confidence", header=None)
            coverage_plan = pd.read_csv(plans_path / "coverage", header=None)
            lift_plan = pd.read_csv(plans_path / "lift", header=None)
            top_plans = [
                confidience_plan.iloc[0, "Antecedent"],
                coverage_plan.iloc[0, "Antecedent"],
                lift_plan.iloc[0, "Antecedent"],
            ]
            

        
def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    # Case: a < feature <= b
    match = re.search(r'([\d.]+) < ' + re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if match:
        a, b = map(float, match.groups())
        if importance > 0:
            return f'{min_val} <= {feature} <= {a}', (min_val, a)
        else:
            return f'{b} < {feature} <= {max_val}', (b, max_val)
    
    # Case: feature > a
    match = re.search(re.escape(feature) + r' > ([\d.]+)', rule_str)
    if match:
        a = float(match.group(1))
        return f'{min_val} <= {feature} <= {a}', (min_val, a)
    
    # Case: feature <= b
    match = re.search(re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if match:
        b = float(match.group(1))
        return f'{b} < {feature} <= {max_val}', (b, max_val)

def run_all_project(model_type, explainer_type):
    projects = read_dataset()
    for project in projects:
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