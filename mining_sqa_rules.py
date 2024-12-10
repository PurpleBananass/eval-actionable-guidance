import argparse
import time
import os
import re

import pandas as pd
from dotenv import load_dotenv
from bigml.api import BigML
from tqdm import tqdm

from Explainer.SQAPlanner.bigml_mining import (
    get_or_create_association,
    get_or_create_dataset,
)
from data_utils import get_true_positives, read_dataset, get_output_dir, get_model

load_dotenv()


def comparison(word):
    regexp = re.finditer(r"\w+=[0-9]+", word)

    for m in regexp:
        matched_word = m.group()
        new_word = "==".join(matched_word.split("="))
        word = word.replace(matched_word, new_word)
    return word


def generate_plans(project, search_strategy, model_type):
    username = os.getenv("BIGML_USERNAME")
    api_key = os.getenv("BIGML_API_KEY")
    api = BigML(username, api_key)

    projects = read_dataset()

    train, test = projects[project]
    model = get_model(project, model_type)

    generated_path = (
        get_output_dir(project, "SQAPlanner", model_type) / "generated_instances"
    )
    rules_path = (
        get_output_dir(project, "SQAPlanner", model_type) / f"rules/{search_strategy}"
    )
    rules_path.mkdir(parents=True, exist_ok=True)

    output_path = (
        get_output_dir(project, "SQAPlanner", model_type) / f"{search_strategy}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    true_positives = get_true_positives(model, train, test)

    len_csv = len(list(generated_path.glob("*.csv")))
    if len_csv == 0:
        return

    for csv in tqdm(
        generated_path.glob("*.csv"), desc=f"{project}", leave=False, total=len_csv
    ):
        if int(csv.stem) not in true_positives.index:
            continue
        output_file = output_path / f"{csv.stem}.csv"
        if output_file.exists():
            continue

        case_data = test.loc[int(csv.stem), :]
        x_test = case_data.drop("target")
        real_target = True

        if case_data["target"] == 0 or real_target == 0:
            continue

        dataset_id = get_or_create_dataset(api, str(csv), project)
        options = {
            "name": csv.stem,
            "tags": [project, search_strategy],
            "search_strategy": search_strategy,
            "max_k": 10,
            "max_lhs": 5,
            "rhs_predicate": [{"field": "target", "operator": "=", "value": "0"}],
        }
        file = rules_path / f"{csv.stem}.csv"
        if not file.exists():
            get_or_create_association(api, dataset_id, options, str(file))

        ff_df = pd.DataFrame([])

        real_target = "target==" + str(real_target) + str(".000")

        rules_df = pd.read_csv(file, encoding="utf-8")
        rules_df = rules_df.replace({r"^b'|'$": ""}, regex=True)

        x_test = x_test.to_frame().T

        for index, row in rules_df.iterrows():
            rule = rules_df.iloc[index, 1]
            rule = comparison(rule)

            class_val = rules_df.iloc[index, 2]
            class_val = comparison(class_val)

            if real_target == class_val:
                print("correctly predicted")

            # Practices  to  follow  to  decrease  the  risk  of  having defects
            elif not x_test.eval(rule).all() and real_target != class_val:
                ff_df = pd.concat([ff_df, row.to_frame().T])

        if ff_df.empty:
            df = pd.DataFrame(
                [],
                columns=["Antecedent", "Antecedent Coverage %", "Confidence", "Lift"],
            )
            df = df.reset_index(drop=True)
            df.to_csv(output_path / f"{csv.stem}.csv", index=False)
        else:
            ff_df = ff_df[["Antecedent", "Antecedent Coverage %", "Confidence", "Lift"]]
            ff_df = ff_df.reset_index(drop=True)
            ff_df = ff_df.head(10)
        ff_df.to_csv(output_file, index=False)


def main(projects, search_strategy, model_type):
    for proj in tqdm(projects, desc="Gen Plans ...", leave=True):
        print(f"Generating plans for {proj}")
        generate_plans(proj, search_strategy, model_type)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--project", type=str)
    argparser.add_argument(
        "--search_strategy",
        type=str,
        default="confidence",
        choices=["coverage", "confidence", "lift"],
    )
    argparser.add_argument("--model_type", type=str, default="RandomForest")
    args = argparser.parse_args()
    count = 0
    while True:
        try:
            print(f"Running... ({count})")
            if args.project:
                main(args.project.split(" "), args.search_strategy, args.model_type)
            else:
                projects = read_dataset()
                projects = list(projects.keys())
                projects = sorted(projects)
                main(projects, args.search_strategy, args.model_type)

            break
        except Exception as e:
            print(e)
            print("Error occurred. Restarting...")
            time.sleep(10)
            count += 1
