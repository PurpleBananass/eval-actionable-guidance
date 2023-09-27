from pathlib import Path
import natsort
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from hyparams import PROJECT_DATASET, RELEASE_DATASET

def get_release_names(project_release):
    project, release_idx = project_release.split("@")
    release_idx = int(release_idx)
    releases = [ release.stem for release in (Path(PROJECT_DATASET) / project).glob('*.csv')]
    releases = natsort.natsorted(releases)
    return f'{project} {releases[release_idx + 1]}'

def get_true_positives(model_path, test):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    ground_truth = test.loc[test['target'] == True, test.columns != 'target']
    predictions = model.predict(ground_truth.values)
    true_positives = ground_truth[predictions == True]
    return true_positives


def map_indexes_to_file(df, int_to_file):
    df.index = df.index.map(int_to_file)
    return df

def project_dataset(project: Path):
    releases = []
    for release_csv in project.glob("*.csv"):
        release = release_csv.name
        releases.append(release)
    releases = natsort.natsorted(releases)
    k_releases = []
    window = 2
    for i in range(len(releases) - window + 1):
        k_releases.append(releases[i : i + window])
    return k_releases

def all_dataset(dataset: Path = Path("project_dataset")):
    projects = {}
    for project in dataset.iterdir():
        if project.is_dir():
            k_releases = project_dataset(project)
            projects[project.name] = k_releases
    return projects


def models_pickle(path_model: Path):
    model_load = []
    for file in path_model.glob("*.pkl"):
        model_load.append(file)
    return model_load


def read_dataset(normalize=False) -> dict[str, list[pd.DataFrame]]:
    save_folder = "release_dataset"
    projects = {}
    for project in Path(save_folder).iterdir():
        if not project.is_dir():
            continue
        train = pd.read_csv(project / "train.csv", index_col=0)
        test = pd.read_csv(project / "test.csv", index_col=0)

        if normalize:
            
            original_dtypes = train.dtypes
            X_train = train.iloc[:, train.columns != "target"]
            y_train = train["target"]
            X_test = test.iloc[:, test.columns != "target"]
            y_test = test["target"]

            scaler = MinMaxScaler()
            scaler.fit(X_train.values)
            X_train_norm = scaler.transform(X_train.values)
            X_test_norm = scaler.transform(X_test.values)

            train = pd.DataFrame(X_train_norm, columns=X_train.columns, index=X_train.index)
            train["target"] = y_train
            train = train.astype(original_dtypes)
            test = pd.DataFrame(X_test_norm, columns=X_test.columns, index=X_test.index)
            test["target"] = y_test
            test = test.astype(original_dtypes)
            val = val.astype(original_dtypes)

            projects[project.name] = [train, test, scaler]
        else:
            projects[project.name] = [train, test]
    return projects

def historical_changes():
    save_folder = Path("historical_dataset")
    save_folder.mkdir(parents=True, exist_ok=True)
    history = {}
    for project in Path(RELEASE_DATASET).iterdir():
        if not project.is_dir():
            continue
        history[project.name] = {}
        train = pd.read_csv(project / "train.csv", index_col=0)
        test = pd.read_csv(project / "test.csv", index_col=0)
        exist_indices = train.index.intersection(test.index)
        deltas = test.loc[exist_indices, test.columns != "target"] - train.loc[exist_indices, train.columns != "target"]

        for feature in deltas.columns:
            nonzeros = deltas[deltas[feature] != 0][feature]
            nonzeros = nonzeros.abs()
            history[project.name][feature] = [round(nonzeros.mean(), 2), round(nonzeros.max(), 2), round(nonzeros.max() / nonzeros.mean(), 2), round(train[feature].max(), 2), round(train[feature].max() / nonzeros.mean(), 2), nonzeros.dtype]

    for project, features in history.items():
        df = pd.DataFrame.from_dict(features, orient="index", columns=["mean_change", "max change", "MAXChange/mean", "max value",  "MAX/mean", "dtype"])

        df.to_csv(save_folder / f"{project}.csv")

def load_historical_changes(project):
    save_folder = Path("historical_dataset")
    df = pd.read_csv(save_folder / f"{project}.csv", index_col=0)
    return df
        

def inverse_transform(df: pd.DataFrame, scaler):
    origianl_dtypes = df.dtypes
    inversed = scaler.inverse_transform(df.values)
    inversed_df = pd.DataFrame(inversed, columns=df.columns, index=df.index)
    inversed_df = inversed_df.astype(origianl_dtypes)
    return inversed_df

if __name__ == "__main__":

    historical_changes()
    projects = read_dataset()
    columns = set()
    for project in projects:
        train, test = projects[project]
        columns = columns.union(set(train.columns.tolist()))
    print(columns)

    
   



