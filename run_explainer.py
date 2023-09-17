
from pathlib import Path
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
from LIME_HPO import LIME_HPO, LIME_Planner
from LORMIKA import LORMIKA
from TimeLIME import TimeLIME
from data_utils import read_dataset
from hyparams import *

# Aussme there are trained models
def run_single_project(train, test, project_name, model_type, explainer_type):
    model_path = Path(f"{MODELS}/{project_name}/{model_type}.pkl") 
    output_path = Path(f"{OUTPUT}/{project_name}/{explainer_type}")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(test.loc[:, test.columns != "target"].values)

    if explainer_type == "LIMEHPO":
        for i in tqdm(range(len(test)), desc=f"{project_name}", leave=False):
            test_instance = test.iloc[i, :]
            test_idx = test_instance.name
            result_path = output_path / f"{test_idx}.csv"
            if result_path.exists():
                continue
            result_path.parent.mkdir(parents=True, exist_ok=True)

            if test_instance["target"] == 0 or predictions[i] == 0:
                continue

            LIME_HPO(
                X_train=train.loc[:, train.columns != "target"],
                test_instance=test.loc[test_idx, test.columns != "target"],
                training_labels=train[["target"]],
                model=model,
                path=result_path,
            )
    elif explainer_type == "LIME":
        for i in tqdm(range(len(test)), desc=f"{project_name}", leave=False):
            test_instance = test.iloc[i, :]
            test_idx = test_instance.name
            result_path = output_path / f"{test_idx}.csv"
            if result_path.exists():
                continue
            result_path.parent.mkdir(parents=True, exist_ok=True)

            if test_instance["target"] == 0 or predictions[i] == 0:
                continue

            LIME_Planner(
                X_train=train.loc[:, train.columns != "target"],
                test_instance=test.loc[test_idx, test.columns != "target"],
                training_labels=train[["target"]],
                model=model,
                path=result_path,
            )

    elif explainer_type == "TimeLIME_":
        TimeLIME(train, test, model, output_path)

    elif explainer_type == "SQAPlanner":
        # 1. Generate instances
        gen_instances_path = output_path.parent / "generated"
        gen_instances_path.mkdir(parents=True, exist_ok=True)
        lormika = LORMIKA(
            train_set=train.loc[:, train.columns != "target"],
            train_class=train[["target"]],
            cases=test.loc[:, test.columns != "target"],
            model=model,
            output_path=gen_instances_path,
        )
        lormika.instance_generation()

        # 2. Generate Association Rules -> generate_plans_SQA.py
        


def run_all_project(model_type, explainer_type):
    projects = read_dataset()
    for project in tqdm(projects, desc="Generating Explanations ...", leave=True):
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