from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from data_utils import load_historical_changes, read_dataset
from pathlib import Path
from hyparams import PROPOSED_CHANGES, EXPERIMENTS

def get_flip_rate(explainers):
    flip_rates = []
    for explainer in explainers:
        result = pd.read_csv(f"./flip_rates/{explainer}.csv")
        result.index = result["Unnamed: 0"]
        result = result.drop(columns=["Unnamed: 0"])
        result.index.name = "project"
        result['Flip_Rate'].name = explainer
        flip_rates.append(result['Flip_Rate'])
    flip_rates = pd.concat(flip_rates, axis=1)
    flip_rates = flip_rates.sort_values(by='project')
    flip_rates.to_csv('./evaluations/flip_rates.csv')
    return flip_rates

def get_accuracy(explainers):
    accuracies = []
    projects = read_dataset()
    project_list = list(sorted(projects.keys()))
    for explainer in explainers:
        for project in project_list:
            accuracy = mean_accuracy_project(project, explainer)
            if accuracy is None:
                continue
            accuracies.append({
                'project': project,
                'accuracy': accuracy,
                'explainer': explainer
            })
    accuracies = pd.DataFrame(accuracies)  
    accuracies = accuracies.pivot(index='project', columns='explainer', values='accuracy') 
    accuracies.to_csv('./evaluations/accuracies.csv')
    return accuracies

def get_feasibility(explainers):
    projects = read_dataset()
    project_list = list(sorted(projects.keys()))
    total_feasibilities = []

    for explainer in explainers:
        for project in project_list:
            train, test = projects[project]
            test_instances = test.drop(columns=["target"])
            historical_mean_changes = load_historical_changes(project)["mean_change"]
            exp_path =  Path(EXPERIMENTS) / project / f"{explainer}.csv"
                
            flipped_instances = pd.read_csv(exp_path, index_col=0)
            flipped_instances = flipped_instances.dropna()
            if len(flipped_instances) == 0:
                continue

            project_feasibilities = []
            for index, flipped in flipped_instances.iterrows():
                current = test_instances.loc[index]
                diff = current != flipped
                diff = diff[diff == True]
                changed_features = diff.index.tolist()

                feasibilites = []
                for feature in changed_features:
                    if not historical_mean_changes[feature]:
                        historical_mean_changes[feature] = 0
                    flipping_proposed_change = abs(flipped[feature] - current[feature])

                    feasibility = 1 - (
                        flipping_proposed_change
                        / (flipping_proposed_change + historical_mean_changes[feature])
                    )
                    feasibilites.append(feasibility)
                feasibility = np.mean(feasibilites)
                project_feasibilities.append(feasibility)
            feasibility = np.mean(project_feasibilities)
            total_feasibilities.append(
                {
                    "Explainer": explainer,
                    "Value": feasibility,
                    "Project": project,
                }
            )
    total_feasibilities = pd.DataFrame(total_feasibilities)
    total_feasibilities = total_feasibilities.pivot(
        index="Project", columns="Explainer", values="Value"
    )
    total_feasibilities.to_csv('./evaluations/feasibilities.csv')
    return total_feasibilities

def mean_accuracy_project(project, explainer):
    plan_path = Path(PROPOSED_CHANGES) / project / explainer / "plans_all.json"
    with open(plan_path) as f:
        plans = json.load(f)
    projects = read_dataset()
    train, test = projects[project]
    test_instances = test.drop(columns=["target"])

    # read the flipped instances
    exp_path = Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
    flipped_instances = pd.read_csv(exp_path, index_col=0)
    flipped_instances = flipped_instances.dropna()
    if len(flipped_instances) == 0:
        return None
    results = []
    for index, flipped in flipped_instances.iterrows():
        current = test_instances.loc[index]
        changed_features = list(plans[str(index)].keys())
        diff = current != flipped
        diff = diff[diff == True]

        score = mean_accuracy_instance(
            current[changed_features], flipped[changed_features], plans[str(index)]
        )

        if score is None:
            continue

        results.append(score)
    # median of results
    results_np = np.array(results)
    return np.mean(results_np)

def compute_score(a1, a2, b1, b2, is_int):
    intersection_start = max(a1, b1)
    intersection_end = min(a2, b2)

    if intersection_start > intersection_end:
        return 1.0  # No intersection

    if is_int:
        intersection_cnt = intersection_end - intersection_start + 1
        union_cnt = (a2 - a1 + 1) + (b2 - b1 + 1) - intersection_cnt
        score = 1 - (intersection_cnt / union_cnt)
    else:
        intersection_length = intersection_end - intersection_start
        union_length = (a2 - a1) + (b2 - b1) - intersection_length
        if union_length == 0:
            return 0.0  # Identical intervals
        score = 1 - (intersection_length / union_length)

    return score


def mean_accuracy_instance(current: pd.Series, flipped: pd.Series, plans):
    scores = []
    for feature in plans:
        flipped_changed = flipped[feature] - current[feature]
        if flipped_changed == 0.0:
            continue

        min_val = min(plans[feature])
        max_val = max(plans[feature])

        a1, a2 = (
            (min_val, flipped[feature])
            if current[feature] < flipped[feature]
            else (flipped[feature], max_val)
        )

        score = compute_score(
            min_val, max_val, a1, a2, current[feature].dtype == "int64"
        )
        assert 0 <= score <= 1, f"Invalid score {score} for feature {feature}"
        scores.append(score)

    return np.mean(scores)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--flip_rate", action="store_true")
    argparser.add_argument("--accuracy", action="store_true")
    argparser.add_argument("--feasibility", action="store_true")
    argparser.add_argument("--explainer", type=str, default="all")

    args = argparser.parse_args()

    if args.explainer == "all":
        explainers = [file.stem for file in Path("./flip_rates").glob("*.csv")]
    else:
        explainers = args.explainer.split(" ")
    
    if args.flip_rate:
        get_flip_rate(explainers)
    
    if args.accuracy:
        fpc_explainers = ['LIME-HPO', 'TimeLIME', 'SQAPlanner_confidence', 'SQAPlanner_coverage', 'SQAPlanner_lift']
        if args.explainer != "all":
            assert set(explainers).issubset(set(fpc_explainers))
            get_accuracy(explainers)
        else:
            get_accuracy(fpc_explainers)
        
    if args.feasibility:
        get_feasibility(explainers)
