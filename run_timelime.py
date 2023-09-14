
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
    models_path.mkdir(parents=True, exist_ok=True)
    for project in tqdm(projects, desc="projects", leave=True):
        train, test, val, inverse = projects[project] # These are normalized datasets
        model_path = models_path / f"{project}.pkl"

        if not Path.exists(model_path):
            model = RandomForestClassifier(n_estimators=100, random_state=1)
            model.fit(train.iloc[:, train.columns != "target"].values, train["target"].values)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        with open(model_path, "rb") as f:
            rf_model = pickle.load(f)
        
        # TimeLIME Planner
        rec, plans, instances, importances, indices = TL(train, test, rf_model)

        # Inverse scale
        plans = np.array(plans)
        plans_left, plans_right = plans[:, :, 0], plans[:, :, 1]
        plans_left = inverse(plans_left)
        plans_right = inverse(plans_right)
        plans = np.stack([plans_left, plans_right], axis=2)

        feature_names = train.columns
        supported_plans = []
        for i in range(len(rec)):
            supported_plan = []
            importances[i]
            for feature_index in range(len(rec[i])):
                if rec[i][feature_index] != 0:
                    feature_importance = [ imp[1] for imp in importances[i] if imp[0] == feature_index ]
                    supported_plan.append(feature_names[feature_index], feature_importance, plans[i][feature_index][0], instances[i][feature_index], plans[i][feature_index][1])
            supported_plan = sorted(supported_plan, key=lambda x: abs(x[1]), reverse=True)
            supported_plans.append(supported_plan)
        print(supported_plans)

        # Save supported plans
        supported_plans_path = Path("./output/supported_plans")
        supported_plans_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(supported_plans, columns=["feature", "importance", "left", "instance", "right"], index=indices)
        df.to_csv(supported_plans_path / f"{project}.csv", index=False)

        




if __name__ == "__main__":
    run_single_dataset()