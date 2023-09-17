import pickle
import re
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

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
    print(sum(predictions))
    for i in range(len(test)):
        test_instance = test.iloc[i, :]
        test_idx = test_instance.name
        if test_instance["target"] == 0 or predictions[i] == 0:
            continue

        # # Load the explanation
        # explanation_path = Path(f"{output_path}/{test_idx}.csv")
        # if not explanation_path.exists():
        #     continue

        print(f"Test instance: {test_idx}")
        if explainer_type == "LIMEHPO" or explainer_type == "LIME":
            explanation_path = Path(f"{output_path}/{test_idx}.csv")
            if not explanation_path.exists():
                continue
            explanation = pd.read_csv(explanation_path)
            top_plans = []
            for row in range(len(explanation)):
                feature, value, importance, min_val, max_val, rule = explanation.iloc[row].values
                str_plan, plan = flip_feature_range(feature, min_val, max_val, importance, rule)
                top_plans.append(plan)
                

        elif explainer_type == "TimeLIME":
            explanation_path = Path(f"{output_path}/{test_idx}.csv")
            if not explanation_path.exists():
                continue
            explanation = pd.read_csv(explanation_path)
            top_plans = []
            for row in range(len(explanation)):
                feature, value, importance, left, right, rec, rule, min_val, max_val = explanation.iloc[row].values
                str_plan = f'{left} <= {feature} <= {right}'
                top_plans.append(str_plan)

        elif explainer_type == "SQAPlanner":
            confidience_plan = pd.read_csv(plans_path / f"confidence/{test_idx}.csv")
            coverage_plan = pd.read_csv(plans_path / f"coverage/{test_idx}.csv")

            if len(confidience_plan) == 0 or len(coverage_plan) == 0:
                continue
         
            plans = [
                split_inequality(confidience_plan.loc[:, "Antecedent"][0]),
                split_inequality(coverage_plan.loc[:, "Antecedent"][0])
            ]
            
            for plan in plans:
                if plan[0] is not None and len(plan[0]) == 3:
                    for sub_plan in plan:
                        if sub_plan[0] is None:
                            sub_plan[0] = train_min[sub_plan[1]]
                        if sub_plan[2] is None:
                            sub_plan[2] = train_max[sub_plan[1]]
                else:
                    if plan[0] is None:
                        plan[0] = train_min[plan[1]]
                    if plan[2] is None:
                        plan[2] = train_max[plan[1]]

            print(plans)
            


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