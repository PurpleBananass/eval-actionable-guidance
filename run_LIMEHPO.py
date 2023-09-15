from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# from tqdm import tqdm
from data_utils import read_dataset
from LIME_HPO import LIME_HPO


def run_single_dataset():
    projects = read_dataset()
    models_path = Path("./models")
    models_path.mkdir(parents=True, exist_ok=True)
    output_path = Path("./output/LIMEHPO")
    project_list = list(projects.keys())

    for project in tqdm(project_list, desc=f"{'LIME-HPO'}", leave=True):
        train, test, val = projects[project]  # These are normalized datasets
        model_path = models_path / f"{project}.pkl"

        if not Path.exists(model_path):
            model = RandomForestClassifier(n_estimators=100, random_state=1)
            model.fit(
                train.iloc[:, train.columns != "target"].values, train["target"].values
            )
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        with open(model_path, "rb") as f:
            blackbox = pickle.load(f)

        predictions = blackbox.predict(test.loc[:, test.columns != "target"].values)

        for i in tqdm(range(len(test)), desc=f"{project}", leave=False):
            test_instance = test.iloc[i, :]
            test_idx = test_instance.name
            result_path = output_path / project / f"{test_idx}.csv"
            if result_path.exists():
                continue
            result_path.parent.mkdir(parents=True, exist_ok=True)

            if test_instance["target"] == 0 or predictions[i] == 0:
                continue

            LIME_HPO(
                X_train=train.loc[:, train.columns != "target"],
                test_instance=test.loc[test_idx, test.columns != "target"],
                training_labels=train[["target"]],
                model=blackbox,
                path=result_path,
            )


if __name__ == "__main__":
    run_single_dataset()
