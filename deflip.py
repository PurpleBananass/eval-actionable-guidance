from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
from data_utils import get_true_positives, read_dataset

from sklearn.metrics.pairwise import cosine_similarity
from hyparams import MODELS, RESULTS, SEED, EXPERIMENTS, OUTPUT
import dice_ml

import warnings

np.random.seed(SEED)
warnings.filterwarnings("ignore", message="UserConfigValidationException will be deprecated from dice_ml.utils")


class DeFlip:
    total_CFs = 10
    max_varied_features = 5

    def __init__(self, training_data: pd.DataFrame, model, save_path, verbose=False):
        self.training_data = training_data
        self.model = model
        self.verbose = verbose
        self.save_path = save_path

        self.min_values = self.training_data.min()
        self.max_values = self.training_data.max()
        self.sd_values = self.training_data.std()

        self.data = dice_ml.Data(
            dataframe=self.training_data,
            continuous_features=list(self.training_data.drop("target", axis=1).columns),
            outcome_name="target",
        )
        self.dice_model = dice_ml.Model(model=model, backend="sklearn")
        self.exp = dice_ml.Dice(self.data, self.dice_model, method="random")

    def run(self, query_instances: pd.DataFrame):
        if (self.save_path / "DeFlip.csv").exists():
            return
        result = {}
        dice_exp = self.exp.generate_counterfactuals(
            query_instances,
            total_CFs=self.total_CFs,
            desired_class="opposite",
            random_seed=SEED,
        )
        
        for i, idx in enumerate(query_instances.index):
            single_instance_result = dice_exp.cf_examples_list[i].final_cfs_df
            assert single_instance_result.loc[lambda x: x["target"] == 1].empty
            single_instance_result = single_instance_result.drop("target", axis=1)

            original_instance = query_instances.loc[idx, :]
            candidates = []
            for j, cf_instance in single_instance_result.iterrows():
                num_changed = self.get_num_changed(original_instance, cf_instance)
                if num_changed <= self.max_varied_features:
                    candidates.append(cf_instance)
            if len(candidates) == 0:
                continue
            candidates = pd.DataFrame(candidates)
            candidates["effort"] = candidates.apply(lambda x: self.effort(original_instance, x), axis=1)
            candidates = candidates.sort_values(by="effort")
            result[idx] = candidates.iloc[0, :]
            
        result_df = pd.DataFrame(result).T
        result_df.to_csv(self.save_path / "DeFlip.csv" )
        return result_df
        

    def get_num_changed(self, query_instance: pd.Series, cf_instance: pd.Series):
        num_changed = np.sum(query_instance != cf_instance)
        return num_changed

    def effort(self, query_instance: pd.Series, cf_instance: pd.Series):
        query_instance = query_instance.values.reshape(1, -1)
        cf_instance = cf_instance.values.reshape(1, -1)

        similarity = cosine_similarity(query_instance, cf_instance)[0][0]
        effort = 1 - similarity
        return effort

def get_flip_rates():
    projects = read_dataset()
    result = {
        "Project": [],
        "Flip": [],
        "Plan": [],
        "TP": [],
    }
    
    for project in projects:
        train, test = projects[project]
        model_path = Path(f"{MODELS}/{project}/RandomForest.pkl")
        true_positives = get_true_positives(model_path, test)
    
        exp_path = Path(f"{EXPERIMENTS}/{project}/DeFlip.csv")
        df = pd.read_csv(exp_path, index_col=0)
        flipped_instances = df.drop("effort", axis=1)
        result["Project"].append(project)
        result["Flip"].append(len(flipped_instances))
        result["Plan"].append(len(flipped_instances))
        result["TP"].append(len(true_positives))
    return pd.DataFrame(result, index=result["Project"]).drop("Project", axis=1).to_csv(Path(RESULTS) / "DeFlip.csv")


def run_single_dataset(
    project: str, train: pd.DataFrame, test: pd.DataFrame, verbose: bool = False
):
    model_path = Path(f"{MODELS}/{project}/RandomForest.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    save_path = Path(f"{EXPERIMENTS}/{project}")
    save_path.mkdir(parents=True, exist_ok=True)

    deflip = DeFlip(train, model, save_path, verbose=verbose)

    positives = test[test["target"] == 1]
    predictions = model.predict(positives.drop("target", axis=1))
    true_positives = positives[predictions == 1]
    query_instances = true_positives.drop("target", axis=1)
    flipped_instances = deflip.run(query_instances)

    tqdm.write(f"| {project} | {len(flipped_instances)} | {len(query_instances)} |")

def filpped_instances(project, candidates):
    flip_path = Path(f"{OUTPUT}/{project}/deflip")
    indices = []
    for file in flip_path.glob("*.csv"):
        if file.stem not in candidates:
            continue
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
        except pd.errors.EmptyDataError:
            continue
        indices.append(file.stem)
    return indices

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--verbose", action="store_true")
    argparser.add_argument("--only_flip_rate", action="store_true")

    args = argparser.parse_args()

    if args.only_flip_rate:
        get_flip_rates()
        exit(0)
    projects = read_dataset()
    tqdm.write("| Project | Flip | Plan | TP |")
    tqdm.write("| ------- | ----- | --- | --- |")
    if args.project == "all":
        for project in tqdm(projects, desc="Projects", leave=True, disable=not args.verbose):
            train, test = projects[project]
            run_single_dataset(project, train, test, verbose=args.verbose)
    else:
        train, test = projects[args.project]
        run_single_dataset(args.project, train, test, verbose=args.verbose)