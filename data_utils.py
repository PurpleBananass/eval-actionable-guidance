import joblib
from pathlib import Path

import xgboost
import natsort
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from hyparams import (
    MODELS,
    OUTPUT,
    PROJECT_DATASET,
    RELEASE_DATASET,
)


def get_model(project_name: str, model_name: str = "RandomForest"):
    if model_name == "XGBoost":
        model_path = Path(f"{MODELS}/{project_name}/XGBoost.xgb")
        model = xgboost.XGBClassifier()
        model.load_model(model_path)
    else:
        model_path = Path(f"{MODELS}/{project_name}/{model_name}.joblib")
        model = joblib.load(model_path)
    return model


def get_output_dir(project_name: str, explainer_type: str, model_type: str) -> Path:
    path = Path(OUTPUT) / project_name / explainer_type / model_type
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    model, train_data: DataFrame, test_data: DataFrame, label: str = "target"
) -> DataFrame:
    assert label in test_data.columns

    ground_truth = test_data.loc[test_data[label], test_data.columns != label]
    scaler = StandardScaler()
    scaler.fit(train_data.drop("target", axis=1))
    ground_truth_scaled = scaler.transform(ground_truth)
    predictions = model.predict(ground_truth_scaled)
    # true_positives = ground_truth[predictions]
    true_positives = ground_truth.iloc[predictions.astype(bool).ravel()]


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


def all_dataset(dataset: Path = Path("./Dataset/project_dataset")):
    projects = {}
    for project in dataset.iterdir():
        if project.is_dir():
            k_releases = project_dataset(project)
            projects[project.name] = k_releases
    return projects


def read_dataset() -> dict[str, list[pd.DataFrame]]:
    projects = {}
    for project in Path(RELEASE_DATASET).iterdir():
        if not project.is_dir():
            continue
        train = pd.read_csv(project / "train.csv", index_col=0)
        test = pd.read_csv(project / "test.csv", index_col=0)

        projects[project.name] = [train, test]
    return projects
