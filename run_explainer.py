
from pathlib import Path
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
from Explainer.LIME_HPO import LIME_HPO
from Explainer.SQAPlanner.LORMIKA import LORMIKA
from Explainer.TimeLIME import TimeLIME
from data_utils import get_true_positives, load_model, read_dataset, get_model_file, get_output_dir
from hyparams import *

def run_single_project(train_data, test_data, project_name, model_type, explainer_type, verbose=True):
    model_file = get_model_file(project_name, model_type)
    output_path = get_output_dir(project_name, explainer_type)
    model = load_model(model_file)

    true_positives = get_true_positives(model_file, test_data)

    match explainer_type:
        case "LIME-HPO":
            for test_idx in tqdm(true_positives.index, desc=f"{project_name}", leave=False, disable=not verbose):
                test_instance = true_positives.loc[test_idx, :]
                output_file = output_path / f"{test_idx}.csv"

                if output_file.exists():
                    continue

                LIME_HPO(
                    X_train=train_data.drop(columns=["target"]),
                    test_instance=test_instance,
                    training_labels=train_data[["target"]],
                    model=model,
                    path=output_file,
                )
                
        case "TimeLIME":
            TimeLIME(train_data, test_data, model, output_path)

        case "SQAPlanner":
            # 1. Generate instances
            gen_instances_path = output_path / "generated_instances"
            gen_instances_path.mkdir(parents=True, exist_ok=True)
            lormika = LORMIKA(
                train_set=train_data.loc[:, train_data.columns != "target"],
                train_class=train_data[["target"]],
                cases=test_data.loc[:, test_data.columns != "target"],
                model=model,
                output_path=gen_instances_path,
            )
            lormika.instance_generation()

            # 2. Generate Association Rules on BigML -> generate_plans_SQA.py

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--model_type", type=str, default="RandomForest")
    argparser.add_argument("--explainer_type", type=str, default="LIME-HPO")
    argparser.add_argument("--project", type=str, default="all")
    args = argparser.parse_args()

    projects = read_dataset()

    if args.project == "all":
        projct_list = list(sorted(projects.key()))
    else:
        project_list = args.project.split(" ")

    for project in tqdm(project_list, desc="Generating Explanations ...", leave=True):
        train, test = projects[project]
        run_single_project(train, test, args.project, args.model_type, args.explainer_type)