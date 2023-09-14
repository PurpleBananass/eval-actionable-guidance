from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from TimeLIME import TL
from data_utils import read_dataset


def run_single_dataset():
    projects = read_dataset()
    models_path = Path("./models")
    output_path = Path("./output/TimeLIME")
    models_path.mkdir(parents=True, exist_ok=True)
    for project in tqdm(projects, desc="projects", leave=True):
        if Path(output_path / project).exists():
            continue
        print(f"Working on {project}...")
        train, test, val, inverse = projects[project]  # These are normalized datasets
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
        rec, plans, instances, importances, indices = TL(train, test, rf_model)

        feature_names = train.columns
        for i in range(len(rec)):
            result_path = output_path / project
            result_path.mkdir(parents=True, exist_ok=True)
            supported_plan = []
            importances[i]
            for feature_index in range(len(rec[i])):
                if rec[i][feature_index] != 0:
                    feature_importance = [
                        imp[1] for imp in importances[i] if imp[0] == feature_index
                    ][0]
                    supported_plan.append(
                        [
                            feature_names[feature_index],
                            instances[i][feature_index],
                            feature_importance,
                            plans[i][feature_index][0],
                            plans[i][feature_index][1],
                        ]
                    )
            supported_plan = sorted(
                supported_plan, key=lambda x: abs(x[2]), reverse=True
            )
            df = pd.DataFrame(
                supported_plan,
                columns=["feature", "value", "importance", "left", "right"],
            )
            df.to_csv(result_path / f"{indices[i]}.csv", index=False)


if __name__ == "__main__":
    run_single_dataset()
