from pathlib import Path
import natsort
import pandas as pd
import pickle
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from hyparams import (
    HISTORICAL_DATASET,
    MODELS,
    OUTPUT,
    PROJECT_DATASET,
    RELEASE_DATASET,
)


def get_model_file(project_name: str, model_name: str = "RandomForest") -> Path:
    return Path(f"{MODELS}/{project_name}/{model_name}.pkl")


def get_output_dir(project_name: str, explainer_type: str) -> Path:
    path = Path(OUTPUT) / project_name / explainer_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(model_file: Path):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model


def get_release_ratio(project_release):
    projects = read_dataset()
    totals = []
    target_release_num = None
    for project in projects:
        train, test = projects[project]
        model_path = Path(f"{MODELS}/{project}/RandomForest.pkl")
        true_positives = get_true_positives(model_path, test)
        num_tp = len(true_positives)
        totals.append(num_tp)
        if project_release == project:
            target_release_num = num_tp
    total_tp = sum(totals)
    return target_release_num / total_tp * 100


def get_release_names(project_release, with_num_tp=True):
    project, release_idx = project_release.split("@")
    release_idx = int(release_idx)
    releases = [
        release.stem for release in (Path(PROJECT_DATASET) / project).glob("*.csv")
    ]
    releases = natsort.natsorted(releases)
    if with_num_tp:
        num_tp = get_release_ratio(project_release)

    return f"{project} {releases[release_idx + 1]} ({num_tp:.1f}%)"


def get_true_positives(
    model_file: Path, test_data: DataFrame, label: str = "target"
) -> DataFrame:
    assert label in test_data.columns

    model = load_model(model_file)
    ground_truth = test_data.loc[test_data[label] == True, test_data.columns != label]
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


def read_dataset() -> dict[str, list[pd.DataFrame]]:
    projects = {}
    for project in Path(RELEASE_DATASET).iterdir():
        if not project.is_dir():
            continue
        train = pd.read_csv(project / "train.csv", index_col=0)
        test = pd.read_csv(project / "test.csv", index_col=0)
        projects[project.name] = [train, test]
    return projects


def save_historical_changes():
    save_folder = Path(HISTORICAL_DATASET)
    save_folder.mkdir(parents=True, exist_ok=True)
    history = {}
    for project in Path(RELEASE_DATASET).iterdir():
        if not project.is_dir():
            continue
        history[project.name] = {}
        train = pd.read_csv(project / "train.csv", index_col=0)
        test = pd.read_csv(project / "test.csv", index_col=0)
        exist_indices = train.index.intersection(test.index)
        deltas = (
            test.loc[exist_indices, test.columns != "target"]
            - train.loc[exist_indices, train.columns != "target"]
        )

        for feature in deltas.columns:
            nonzeros = deltas[deltas[feature] != 0][feature]
            nonzeros = nonzeros.abs()
            history[project.name][feature] = [
                round(nonzeros.mean(), 2),
                round(nonzeros.max(), 2),
                round(nonzeros.max() / nonzeros.mean(), 2),
                round(train[feature].max(), 2),
                round(train[feature].max() / nonzeros.mean(), 2),
                nonzeros.dtype,
            ]

    for project, features in history.items():
        df = pd.DataFrame.from_dict(
            features,
            orient="index",
            columns=[
                "mean_change",
                "max change",
                "MAXChange/mean",
                "max value",
                "MAX/mean",
                "dtype",
            ],
        )

        df.to_csv(save_folder / f"{project}.csv")


def load_historical_changes(project):
    save_folder = Path(HISTORICAL_DATASET)
    df = pd.read_csv(save_folder / f"{project}.csv", index_col=0)
    return df

if __name__ == "__main__":
    save_historical_changes()