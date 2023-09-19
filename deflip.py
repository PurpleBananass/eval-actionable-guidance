from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
from data_utils import read_dataset

from hyparams import MODELS, SEED, EXPERIMENTS
import dice_ml

import warnings

np.random.seed(SEED)
warnings.filterwarnings("ignore")

class DeFlip():
    total_CFs = 10
    max_varied_features = 5
    def __init__(self, training_data: pd.DataFrame, model, save_path, verbose=False):
        self.training_data = training_data
        self.model = model
        self.verbose = verbose
        self.save_path = save_path

        self.data = dice_ml.Data(dataframe=self.training_data, continuous_features=list(self.training_data.drop("target", axis=1).columns), outcome_name='target')
        self.dice_model = dice_ml.Model(model=model, backend="sklearn")
        self.exp = dice_ml.Dice(self.data, self.dice_model, method='random')

    def run(self, query_instances: pd.DataFrame):
        for i, query_instance in tqdm(query_instances.iterrows(), total=len(query_instances), desc="DeFlip", leave=False):
            exp = self.exp.generate_counterfactuals(query_instance.to_frame().T, total_CFs=self.total_CFs, desired_class="opposite")
            final = exp.cf_examples_list[0].final_cfs_df
            final = final.drop("target", axis=1)
            result = []
            for j, cf_instance in final.iterrows():
                num_changed = self.changed_features(query_instance, cf_instance)
                if num_changed <= self.max_varied_features:
                    result.append(cf_instance)
            result = pd.DataFrame(result)
            result.to_csv(self.save_path / f"{i}.csv", index=False, header=True)
            


    def changed_features(self, query_instance: pd.Series, cf_instance: pd.Series):
        num_changed = np.sum(query_instance != cf_instance)
        return num_changed


def run_single_dataset(project: str, train: pd.DataFrame, test: pd.DataFrame, verbose: bool = False):
    model_path = Path(f"{MODELS}/{project}/RandomForest.pkl") 
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    save_path = Path(f"{EXPERIMENTS}/{project}/deflip")
    save_path.mkdir(parents=True, exist_ok=True)

    deflip = DeFlip(train, model, save_path, verbose=verbose)

    query_instances = test.drop("target", axis=1)
    deflip.run(query_instances)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--verbose", action="store_true")

    args = argparser.parse_args()
    projects = read_dataset()
    if args.project == "all":
        for project in projects:
            train, test = projects[project]
            run_single_dataset(project, train, test, verbose=args.verbose)
    else:
        train, test = projects[args.project]
        run_single_dataset(args.project, train, test, verbose=args.verbose)
            