
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from LORMIKA import LORMIKA
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
            blackbox = pickle.load(f)

        model_kwargs = {
            "train_set": train.loc[:, train.columns != "target"],
            "train_class": train[["target"]],
            "cases": test.loc[:, test.columns != "target"],
            "model": blackbox,
            "output_path": "./output/generated/" + project,
            "inverse": inverse,
        }
        lormika = LORMIKA(**model_kwargs)
        lormika.instance_generation()

if __name__ == "__main__":
    run_single_dataset()