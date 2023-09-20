from argparse import ArgumentParser
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_utils import read_dataset

from hyparams import MODELS, PLANS, SEED, EXPERIMENTS

np.random.seed(SEED)

def flip_single_project(test, project_name, explainer_type, search_strategy, only_minimum, verbose=False):
    model_path = Path(f"{MODELS}/{project_name}/RandomForest.pkl") 
    exp_path = Path(f"{EXPERIMENTS}/{project_name}")
    if only_minimum:
        plan_path = Path(f"{PLANS}/{project_name}/{explainer_type}/plans.json")
    else:
        plan_path = Path(f"{PLANS}/{project_name}/{explainer_type}/plans_all.json")

    if search_strategy is not None:
        plan_path = Path(f"{PLANS}/{project_name}/{explainer_type}/{search_strategy}/plans.json")
        exp_path = Path(f"{EXPERIMENTS}/{project_name}/{search_strategy}")
        
    exp_path.mkdir(parents=True, exist_ok=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(plan_path, "r") as f:
        plans = json.load(f)

    test_names = list(plans.keys())
    flipped_instances = {}
    for test_name in tqdm(test_names, desc=f"{project_name}", leave=False, disable=not verbose):
        original_instance = test.loc[int(test_name), test.columns != "target"]
        flipped_instance = original_instance.copy()
        features = list(plans[test_name].keys())

        if only_minimum:
            flipped_instance[features] = [ plans[test_name][feature] for feature in features ]
            prediction = model.predict_proba(flipped_instance.values.reshape(1, -1))[:, 0]
            if prediction >= 0.5:
                flipped_instances[test_name] = flipped_instance
        else:
            # binary search 
            pass

    if verbose:
        print(f"Number of flipped instances: {len(flipped_instances)} / {len(test_names)}")
    df = pd.DataFrame(flipped_instances).T
    df.to_csv(exp_path / f"{explainer_type}{'_all' if not only_minimum else ''}.csv")
        
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--explainer_type", type=str, required=True)
    argparser.add_argument("--search_strategy", type=str, default=None)
    argparser.add_argument("--only_minimum", action="store_true")
    argparser.add_argument("--verbose", action="store_true")

    args = argparser.parse_args()
    projects = read_dataset()
    if args.project == "all":
        for project in tqdm(projects, desc="Projects", leave=True, disable=not args.verbose):
            _, test = projects[project]
            flip_single_project(test, project, args.explainer_type, args.search_strategy, args.only_minimum, verbose=args.verbose)
    else:
        _, test = projects[args.project]
        flip_single_project(test, args.project, args.explainer_type, args.search_strategy, args.only_minimum, verbose=args.verbose)