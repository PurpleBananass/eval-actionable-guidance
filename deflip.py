# Load plans

from argparse import ArgumentParser
import json
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
from data_utils import read_dataset

from hyparams import MODELS, OUTPUT, PLANS, SEED, EXPERIMENTS
import dice_ml

np.random.seed(SEED)

class DeFlip():
    total_CFs = 10
    max_varied_features = 5
    def __init__(self, training_data, model, save_path, verbose=False):
        self.training_data = training_data
        self.model = model
        self.verbose = verbose
        self.save_path = save_path

        self.data = dice_ml.Data(dataframe=train, continuous_features=list(training_data.columns), outcome_name='target')
        self.dice_model = dice_ml.Model(model=model, backend="sklearn")
        self.exp = dice_ml.Dice(self.data, self.dice_model, method='random')

    def run(self, test_data):
        self.exp.generate_counterfactuals(query_instances=test_data, total_CFs=self.total_CFs, desired_class="opposite")

        for i in tqdm(range(len(self.exp.cf_examples_list)), desc=f'{self.save_path.parent.name}', leave=False):
            final = self.exp.cf_examples_list[i].final_cfs_df.drop(columns=["target"])
            result = pd.DataFrame()
            for j, row in final.iterrows():
                num_changed = self.changed_features(test_data.iloc[i, 0], row)
                if num_changed <= self.max_varied_features:
                    result = pd.concat([result, row.to_frame().T], axis=0)
                    if self.verbose:
                        print(f"{test_data.iloc[i, 0]} flipped with {num_changed} feature changes")
            result.to_csv(self.save_path / f"{i}.csv")

    def changed_features(self, test_data, cf_data):
        num_changed = np.sum(test_data != cf_data)
        return num_changed


def run_single_dataset(project, train, test, verbose):
    model_path = Path(f"{MODELS}/{project}/RandomForest.pkl") 
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    save_path = Path(f"{EXPERIMENTS}/{project}/deflip")
    save_path.mkdir(parents=True, exist_ok=True)
    train_x = train.loc[:, train.columns != "target"]
    test_x = test.loc[:, test.columns != "target"]
    deflip = DeFlip(train_x, model, save_path, verbose=verbose)
    deflip.run(test_x)


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
            