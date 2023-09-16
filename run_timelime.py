from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from TimeLIME import TL, TimeLIME
from data_utils import read_dataset


def run_single_dataset():
    projects = read_dataset()
    models_path = Path("./models")
    output_path = Path("./output/TimeLIME")
    models_path.mkdir(parents=True, exist_ok=True)
    project_list = list(projects.keys())
    i = 0
    for project in tqdm(project_list, desc=f"{project_list[i]}", leave=True):
        train, test, val = projects[project]  
        model_path = models_path / f"{project}.pkl"

        if not Path.exists(model_path):
            model = RandomForestClassifier(n_estimators=100, random_state=1)
            model.fit(
                train.iloc[:, train.columns != "target"].values, train["target"].values
            )
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        with open(model_path, "rb") as f:
            rf_model = pickle.load(f)

        # TimeLIME Planner
        result_path = output_path / project
        result_path.mkdir(parents=True, exist_ok=True)
        TimeLIME(train, test, rf_model, result_path)
        i += 1


if __name__ == "__main__":
    run_single_dataset()
