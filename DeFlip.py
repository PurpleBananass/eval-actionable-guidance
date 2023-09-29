from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd
from data_utils import get_true_positives, read_dataset, load_model, get_model_file

from sklearn.metrics.pairwise import cosine_similarity
from hyparams import MODELS, RESULTS, SEED, EXPERIMENTS, OUTPUT
import dice_ml

import warnings

np.random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)

class DeFlip:
    total_CFs = 10
    max_varied_features = 5
    incationable_features = [
        "MAJOR_COMMIT",
        "MAJOR_LINE",
        "MINOR_COMMIT",
        "MINOR_LINE",
        "OWN_COMMIT",
        "OWN_LINE",
        "ADEV",
        "Added_lines",
        "Del_lines",
    ]

    def __init__(self, training_data: pd.DataFrame, model, save_path, actionable=False):
        self.training_data = training_data
        self.model = model
        self.save_path = save_path / "DeFlip_actionable.csv" if actionable else save_path / "DeFlip.csv"
        self.actionable = actionable

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

        self.actionable_features = list(set(self.training_data.columns) - set(self.incationable_features) - set(["target"]))

    def run(self, query_instances: pd.DataFrame):
        if self.save_path.exists():
            pass
        result = {}
        if self.actionable:
            dice_exp = self.exp.generate_counterfactuals(
                query_instances,
                total_CFs=self.total_CFs,
                desired_class="opposite",
                random_seed=SEED,
                features_to_vary=self.actionable_features,
            )
        else:
            dice_exp = self.exp.generate_counterfactuals(
                query_instances,
                total_CFs=self.total_CFs,
                desired_class="opposite",
                random_seed=SEED,
            )
        
        for i, idx in enumerate(query_instances.index):
            
            single_instance_result = dice_exp.cf_examples_list[i].final_cfs_df
            if single_instance_result is None:
                continue
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
            candidates["similarity"] = candidates.apply(lambda x: self.get_similarity(original_instance, x), axis=1)
            candidates = candidates.sort_values(by="similarity", ascending=False)
            candidates = candidates.drop("similarity", axis=1)

            result[idx] = candidates.iloc[0, :]
            
        result_df = pd.DataFrame(result).T
        result_df.to_csv(self.save_path)
        return result_df 

    def get_num_changed(self, query_instance: pd.Series, cf_instance: pd.Series):
        num_changed = np.sum(query_instance != cf_instance)
        return num_changed

    def get_similarity(self, query_instance: pd.Series, cf_instance: pd.Series):
        query_instance = query_instance.values.reshape(1, -1)
        cf_instance = cf_instance.values.reshape(1, -1)

        similarity = cosine_similarity(query_instance, cf_instance)[0][0]
        return similarity

def get_flip_rates(actionable=False):
    projects = read_dataset()
    result = {
        "Project": [],
        "Flipped": [],
        "Plan": [],
        "TP": [],
    }
    
    for project in projects:
        train, test = projects[project]
        model_path = Path(f"{MODELS}/{project}/RandomForest.pkl")
        true_positives = get_true_positives(model_path, test)
        if actionable:
            exp_path = Path(f"{EXPERIMENTS}/{project}/DeFlip_actionable.csv")
        else:
            exp_path = Path(f"{EXPERIMENTS}/{project}/DeFlip.csv")
        df = pd.read_csv(exp_path, index_col=0)
        flipped_instances = df
        result["Project"].append(project)
        result["Flipped"].append(len(flipped_instances))
        result["Plan"].append(len(flipped_instances))
        result["TP"].append(len(true_positives))
    result_df = pd.DataFrame(result, index=result["Project"]).drop("Project", axis=1)
    # result_df = result_df.dropna()
    result_df['Flip_Rate'] = result_df['Flipped'] / result_df['TP']
    return result_df.to_csv(Path(RESULTS) / "DeFlip.csv" if not actionable else Path(RESULTS) / "DeFlip_actionable.csv")


def run_single_dataset(
    project: str, train: pd.DataFrame, test: pd.DataFrame, actionable: bool = False
):
    model_file = get_model_file(project)
    model = load_model(model_file)
    save_path = Path(f"{EXPERIMENTS}/{project}")
    save_path.mkdir(parents=True, exist_ok=True)

    deflip = DeFlip(train, model, save_path, actionable=actionable)

    positives = test[test["target"] == 1]
    predictions = model.predict(positives.drop("target", axis=1))
    true_positives = positives[predictions == 1]
    query_instances = true_positives.drop("target", axis=1)
    flipped_instances = deflip.run(query_instances)

    tqdm.write(f"| {project} | {len(flipped_instances)} | {len(query_instances)} |")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--get_flip_rate", action="store_true")
    argparser.add_argument("--actionable", action="store_true")

    args = argparser.parse_args()

    if args.get_flip_rate:
        get_flip_rates(args.actionable)
        exit(0)
    
        
    tqdm.write("| Project | Flip  | #TP |")
    tqdm.write("| ------- | ----- | --- |")

    projects = read_dataset()

    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split(" ")
    
    for project in tqdm(project_list, desc="Projects", leave=True):
        train, test = projects[project]
        run_single_dataset(project, train, test, actionable=args.actionable)