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
from data_utils import read_dataset, get_true_positives
from concurrent.futures import ProcessPoolExecutor, as_completed

from hyparams import MODELS, PROPOSED_CHANGES, SEED, EXPERIMENTS, RESULTS

np.random.seed(SEED)

def flip_instance(
    original_instance, features, changeable_features, model_path
):
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

def get_flip_rates(explainer_type, search_strategy, only_minimum):
    
    projects = read_dataset()
    result = {
        "Project": [],
        "Flipped": [],
        "Plan": [],
        "TP": [],
    }
    for project_name in projects:
        _, test = projects[project_name]
        model_path = Path(f"{MODELS}/{project_name}/RandomForest.pkl")
        
        match (only_minimum, search_strategy):
            case (True, None):
                plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}/plans.json")
                exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}.csv")
                result_path = Path(RESULTS) / f"{explainer_type}.csv"
            case (False, None):
                plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}/plans_all.json")
                exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}_all.csv")
                result_path = Path(RESULTS) / f"{explainer_type}_all.csv"
            case (True, _):
                plan_path = Path(
                    f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}_{search_strategy}/plans.json"
                )
                exp_path = Path(
                    f"{EXPERIMENTS}/{project_name}/{explainer_type}_{search_strategy}.csv"
                )
                result_path = Path(RESULTS) / f"{explainer_type}_{search_strategy}.csv"
            case (False, _):
                plan_path = Path(
                    f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}_{search_strategy}/plans_all.json"
                )
                exp_path = Path(
                    f"{EXPERIMENTS}/{project_name}/{explainer_type}_{search_strategy}_all.csv"
                )
                result_path = Path(RESULTS) / f"{explainer_type}_{search_strategy}_all.csv"
        
        result_path.parent.mkdir(parents=True, exist_ok=True)
        file = pd.read_csv(exp_path, index_col=0)
        flipped_instances = {
            test_name: file.loc[test_name, :] for test_name in file.index
        }
        with open(plan_path, "r") as f:
            plans = json.load(f)
        
        true_positives = get_true_positives(model_path, test)
        df = pd.DataFrame(flipped_instances).T
        result["Project"].append(project_name)
        result["Flipped"].append(len(df.dropna()))
        result["Plan"].append(len(plans.keys()))

        result["TP"].append(len(true_positives))
        left_indices = list(set(df.index.astype(str)) - set(plans.keys()))
        if len(left_indices) > 0:
            print(f"{project_name}: {left_indices}")

    result_df = pd.DataFrame(result, index=result["Project"]).drop("Project", axis=1)
    result_df = result_df.dropna()
    result_df['Flip_Rate'] = result_df['Flipped'] / result_df['TP']
    result_df.to_csv(result_path)



def flip_single_project(
    test, project_name, explainer_type, search_strategy, only_minimum, verbose=False, load=True
):
    model_path = Path(f"{MODELS}/{project_name}/RandomForest.pkl")

    match (only_minimum, search_strategy):
        case (True, None):
            plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}/plans.json")
            exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}.csv")
        case (False, None):
            plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}/plans_all.json")
            exp_path = Path(f"{EXPERIMENTS}/{project_name}/{explainer_type}_all.csv")
        case (True, _):
            plan_path = Path(
                f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}_{search_strategy}/plans.json"
            )
            exp_path = Path(
                f"{EXPERIMENTS}/{project_name}/{explainer_type}_{search_strategy}.csv"
            )
        case (False, _):
            plan_path = Path(
                f"{PROPOSED_CHANGES}/{project_name}/{explainer_type}_{search_strategy}/plans_all.json"
            )
            exp_path = Path(
                f"{EXPERIMENTS}/{project_name}/{explainer_type}_{search_strategy}_all.csv"
            )

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    if exp_path.exists() and load:
        file = pd.read_csv(exp_path, index_col=0)
        computed_test_names = set(file.index.astype(str))
        flipped_instances = {
            test_name: file.loc[test_name, :] for test_name in file.index
        }
        
    else:
        computed_test_names = set()
        flipped_instances = {}

    with open(plan_path, "r") as f:
        plans = json.load(f)

    test_names = list(plans.keys())
    
    true_positives = get_true_positives(model_path, test)
    if verbose and load and len(flipped_instances) > 0:
        df = pd.DataFrame(flipped_instances).T
        if len(df) < len(test_names):
            tqdm.write(f"| {project_name} | {len(df.dropna())} | {len(df)} |{len(test_names)} | {len(df.dropna()) / len(df):.3f} | {len(true_positives)} | Loaded !")
    
    if only_minimum:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        for test_name in tqdm(
            test_names, desc=f"{project_name}", leave=False, disable=not verbose
        ):
            if test_name in computed_test_names:
                continue

            original_instance = test.loc[int(test_name), test.columns != "target"]
            flipped_instance = original_instance.copy()
            features = list(plans[test_name].keys())

            flipped_instance[features] = [
                plans[test_name][feature] for feature in features
            ]
            prediction = model.predict_proba(flipped_instance.values.reshape(1, -1))[
                :, 0
            ]
            if prediction >= 0.5:
                flipped_instances[test_name] = flipped_instance
            else:
                flipped_instances[test_name] = pd.Series(
                    [np.nan] * len(original_instance),
                    index=original_instance.index,
                )

    else:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {}
            max_perturbations = []
            for test_name in tqdm(test_names, desc=f"{project_name} queing...", leave=False, disable=not verbose):
                # Calcuate the number of perturbations (len(A) * len(B) * ...)
                features = list(plans[test_name].keys())
                computation = 1
                for feature in features:
                    computation * (len(plans[test_name][feature]) + 1) # +1 for original value
                max_perturbations.append([test_name, computation])
            max_perturbations = sorted(max_perturbations, key=lambda x: x[1])

            # Start from the test with the smallest number of perturbations
            test_indices = [x[0] for x in max_perturbations]

            for test_name in tqdm(
                test_indices, desc=f"{project_name}", leave=False, disable=not verbose
            ):
                if test_name in computed_test_names:
                    continue

                original_instance = test.loc[int(test_name), test.columns != "target"]
                features = list(plans[test_name].keys())

                changeable_features = []
                for feature in features:
                    if original_instance[feature] <= min(plans[test_name][feature]):
                        changeable_features = [
                            [original_instance[feature]] + plans[test_name][feature]
                        ] + changeable_features
                    else:
                        changeable_features = changeable_features + [
                            [original_instance[feature]] + plans[test_name][feature]
                        ]
                # Submitting the task for parallel execution
                future = executor.submit(
                    flip_instance,
                    original_instance,
                    features,
                    changeable_features,
                    model_path,
                )
                futures[future] = test_name
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{project_name}",
                leave=False,
                disable=not verbose,
            ):
                try:
                    test_name = futures[future]
                    flipped_instance = future.result()

                    if flipped_instance is not None:
                        flipped_instances[test_name] = flipped_instance
                    else:  # if flipped_instance is None
                        flipped_instances[test_name] = pd.Series(
                            [np.nan] * len(original_instance),
                            index=original_instance.index,
                        )

                    # Save each completed test_name immediately, including None cases
                    pd.DataFrame(flipped_instances).T.to_csv(exp_path)

                except Exception as e:
                    tqdm.write(f"Error occurred: {e}")
                    traceback.print_exc()
                    exit()

    df = pd.DataFrame(flipped_instances).T
    if verbose:
        tqdm.write(f"| {project_name} | {len(df.dropna())} | {len(df)} |{len(test_names)} | {len(df.dropna()) / len(df):.3f} | {len(true_positives)} |")
    df.to_csv(exp_path)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--explainer_type", type=str, required=True)
    argparser.add_argument("--search_strategy", type=str, default=None)
    argparser.add_argument("--only_minimum", action="store_true")
    argparser.add_argument("--verbose", action="store_true")
    argparser.add_argument("--new", action="store_true")
    argparser.add_argument("--get_flip_rate", action="store_true")

    args = argparser.parse_args()

    if args.get_flip_rate:
        # RQ 1
        get_flip_rates(
            args.explainer_type, args.search_strategy, args.only_minimum
        )
    else:
        tqdm.write("| Project | Flipped | Computed | Plan | Flip Rate | TP |")
        tqdm.write("| ------- | ------- | -------- | ---- | --------- | -- |")
        projects = read_dataset()
        if args.project == "all":
            project_list = list(sorted(projects.keys()))
        else:
            project_list = args.project.split(" ")

        for project in tqdm(
            project_list, desc="Projects", leave=True, disable=not args.verbose
        ):
            _, test = projects[project]
            flip_single_project(
                test,
                project,
                args.explainer_type,
                args.search_strategy,
                args.only_minimum,
                verbose=args.verbose,
                load=not args.new
            )
