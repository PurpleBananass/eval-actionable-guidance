from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

from data_utils import read_dataset

SEED = 1

def train_all_dataset():
    projects = read_dataset()
    models_path = Path("./models")
    models_path.mkdir(parents=True, exist_ok=True)
    for project in tqdm(projects, desc="projects", leave=True):
        train, test, val, inverse = projects[project]  # These are normalized datasets
        model_path = models_path / f"{project}.pkl"

        if not Path.exists(model_path):
            model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            model.fit(
                train.iloc[:, train.columns != "target"].values, train["target"].values
            )
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        with open(model_path, "rb") as f:
            rf_model = pickle.load(f)

        # Test the model performance
        print(f"Working on {project}...")
        print(f"Train accuracy: {rf_model.score(train.iloc[:, train.columns != 'target'].values, train['target'].values)}")
        print(f"Test accuracy: {rf_model.score(test.iloc[:, test.columns != 'target'].values, test['target'].values)}")

if __name__ == "__main__":
    train_all_dataset()