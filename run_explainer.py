import concurrent.futures
import warnings
import os
from argparse import ArgumentParser

from Explainer.LIME_HPO import LIME_HPO, LIME_Planner
from sklearn.exceptions import ConvergenceWarning
from Explainer.TimeLIME import TimeLIME
from tqdm import tqdm

from data_utils import get_true_positives, read_dataset, get_output_dir, get_model
from Explainer.SQAPlanner.LORMIKA import LORMIKA

warnings.filterwarnings("ignore", category=ConvergenceWarning)

warnings.filterwarnings("ignore", category=UserWarning)


def process_test_idx(
    test_idx, true_positives, train_data, model, output_path, explainer_type
):
    test_instance = true_positives.loc[test_idx, :]
    output_file = output_path / f"{test_idx}.csv"

    if output_file.exists():
        return None

    print(f"Processing {test_idx} on {os.getpid()}")

    if explainer_type == "LIME":
        LIME_Planner(
            X_train=train_data.drop(columns=["target"]),
            test_instance=test_instance,
            training_labels=train_data[["target"]],
            model=model,
            path=output_file,
        )

    elif explainer_type == "LIME-HPO":
        LIME_HPO(
            X_train=train_data.drop(columns=["target"]),
            test_instance=test_instance,
            training_labels=train_data[["target"]],
            model=model,
            path=output_file,
        )

    return os.getpid()


def run_single_project(
    train_data, test_data, project_name, model_type, explainer_type, verbose=True
):
    output_path = get_output_dir(project_name, explainer_type, model_type)
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train_data, test_data)

    if len(true_positives) == 0:
        return
    print(len(true_positives))

    if explainer_type in ["LIME", "LIME-HPO"]:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_test_idx,
                    test_idx,
                    true_positives,
                    train_data,
                    model,
                    output_path,
                    explainer_type,
                )
                for test_idx in true_positives.index
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{project_name}",
                disable=not verbose,
            ):
                out = future.result()
                if out is not None:
                    tqdm.write(f"Process {out} finished")
    elif explainer_type == "TimeLIME":
        TimeLIME(train_data, test_data, model, output_path)
    elif explainer_type == "SQAPlanner":
        # SQAPlanner specific code
        gen_instances_path = output_path / "generated_instances"
        gen_instances_path.mkdir(parents=True, exist_ok=True)
        lormika = LORMIKA(
            train_set=train_data.loc[:, train_data.columns != "target"],
            train_class=train_data[["target"]],
            cases=true_positives.loc[:, true_positives.columns != "target"],
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
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split(" ")

    for project in tqdm(project_list, desc="Project", leave=True):
        print(project)
        train, test = projects[project]
        run_single_project(train, test, project, args.model_type, args.explainer_type)
