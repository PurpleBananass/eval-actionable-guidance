from argparse import ArgumentParser
from itertools import product
import json
import os
from pathlib import Path
import pickle
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_utils import read_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

from hyparams import MODELS, PLANS, SEED, EXPERIMENTS

np.random.seed(SEED)
def find_smallest_perturbation(original_instance, features, changeable_features, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    for values in product(*changeable_features):
        modified_instance = original_instance.copy()
        for feature, value in zip(features, values):
            modified_instance[feature] = value
        prediction = model.predict_proba(modified_instance.values.reshape(1, -1))[:, 0]
        if prediction >= 0.5:
            return modified_instance
    return None  # Return None if no perturbation flips the prediction

def flip_single_project(test, project_name, explainer_type, search_strategy, only_minimum, verbose=False):
    
    model_path = Path(f"{MODELS}/{project_name}/RandomForest.pkl") 
   
    match (only_minimum, search_strategy):
        case (True, None):
            plan_path = Path(f"{PLANS}/{project_name}/{explainer_type}/plans.json")
            exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}.csv")
        case (False, None):
            plan_path = Path(f"{PLANS}/{project_name}/{explainer_type}/plans_all.json")
            exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}_all.csv")
        case (True, _):
            plan_path = Path(f"{PLANS}/{project_name}/{explainer_type}_{search_strategy}/plans.json")
            exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}_{search_strategy}.csv")
        case (False, _):
            plan_path = Path(f"{PLANS}/{project_name}/{explainer_type}_{search_strategy}/plans_all.json")
            exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}_{search_strategy}_all.csv")
        
    
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    
    if exp_path.exists():
        computed_test_names = set(pd.read_csv(exp_path, index_col=0).index.astype(str))
    else:
        computed_test_names = set()

    with open(plan_path, "r") as f:
        plans = json.load(f)

    test_names = list(plans.keys())
    flipped_instances = {}

    if only_minimum:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        for test_name in tqdm(test_names, desc=f"{project_name}", leave=False, disable=not verbose):
            if test_name not in plans:
                continue
            original_instance = test.loc[int(test_name), test.columns != "target"]
            flipped_instance = original_instance.copy()
            features = list(plans[test_name].keys())

        
            flipped_instance[features] = [ plans[test_name][feature] for feature in features ]
            prediction = model.predict_proba(flipped_instance.values.reshape(1, -1))[:, 0]
            if prediction >= 0.5:
                flipped_instances[test_name] = flipped_instance

    else:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {}
            for test_name in tqdm(test_names, desc=f"{project_name}", leave=False, disable=not verbose):
                if test_name in computed_test_names:
                    continue

                if test_name not in plans:
                    continue

                original_instance = test.loc[int(test_name), test.columns != "target"]
                features = list(plans[test_name].keys())

                changeable_features = []
                for feature in features:
                
                    if original_instance[feature] <= min(plans[test_name][feature]):
                        changeable_features = [plans[test_name][feature]] + changeable_features
                    else:
                        changeable_features = changeable_features + [plans[test_name][feature]]
                # Submitting the task for parallel execution
                future = executor.submit(find_smallest_perturbation, original_instance, features, changeable_features, model_path)
                futures[future] = test_name
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{project_name}", leave=False, disable=not verbose):
                try:
                    test_name = futures[future]
                    flipped_instance = future.result()

                    if flipped_instance is not None:
                        flipped_instances[test_name] = flipped_instance
                    else:  # if flipped_instance is None
                        flipped_instances[test_name] = pd.Series([np.nan] * len(original_instance), index=original_instance.index)

                    # Save each completed test_name immediately, including None cases
                    pd.DataFrame(flipped_instances).T.to_csv(exp_path)

                except Exception as e:
                    tqdm.write(f"Error occurred: {e}")
                    traceback.print_exc()
                    exit()
    
    if verbose:
        tqdm.write(f"| {project_name} | {len(flipped_instances)} | {len(test_names)} |")
    df = pd.DataFrame(flipped_instances).T
    df.to_csv(exp_path)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--explainer_type", type=str, required=True)
    argparser.add_argument("--search_strategy", type=str, default=None)
    argparser.add_argument("--only_minimum", action="store_true")
    argparser.add_argument("--verbose", action="store_true")

    args = argparser.parse_args()
    projects = read_dataset()
    tqdm.write("| Project | Flipped | Total |")
    tqdm.write("| ------- | ------- | ----- |")
    if args.project == "all":
        for project in tqdm(projects, desc="Projects", leave=True, disable=not args.verbose):
            _, test = projects[project]
            flip_single_project(test, project, args.explainer_type, args.search_strategy, args.only_minimum, verbose=args.verbose)
    else:
        _, test = projects[args.project]
        flip_single_project(test, args.project, args.explainer_type, args.search_strategy, args.only_minimum, verbose=args.verbose)