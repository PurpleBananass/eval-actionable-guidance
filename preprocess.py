from pathlib import Path

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, StrVector
from rpy2.robjects.packages import importr

from data_utils import all_dataset


def map_indexes_to_int(train_df, test_df):
    all_files = train_df.index.append(test_df.index).unique()
    file_to_int = {file: i for i, file in enumerate(all_files)}
    int_to_file = {i: file for file, i in file_to_int.items()}

    train_df.index = train_df.index.map(file_to_int)
    test_df.index = test_df.index.map(file_to_int)

    return train_df, test_df, int_to_file


def remove_variables(df, vars_to_remove):
    df.drop(vars_to_remove, axis=1, inplace=True, errors="ignore")


def get_df(project: str, release: str, path_data: str = "./Dataset/project_dataset"):
    df = pd.read_csv(f"{path_data}/{project}/{release}", index_col=0)
    return df


def preprocess(project, releases: list[str]):
    dataset_trn = get_df(project, releases[0])
    dataset_tst = get_df(project, releases[1])

    duplicated_index_trn = dataset_trn.index.duplicated(keep="first")
    duplicated_index_tst = dataset_tst.index.duplicated(keep="first")

    dataset_trn = dataset_trn[~duplicated_index_trn]
    dataset_tst = dataset_tst[~duplicated_index_tst]

    print(f"Project: {project}")
    print(
        f"Release: {releases[0]} total: {len(dataset_trn)} bug: {len(dataset_trn[dataset_trn['RealBug'] == 1])}"
    )
    print(
        f"Release: {releases[1]} total: {len(dataset_tst)} bug: {len(dataset_tst[dataset_tst['RealBug'] == 1])}"
    )

    dataset_tst = dataset_tst.drop(
        dataset_tst.index[
            dataset_tst.isin(dataset_trn.to_dict(orient="list")).all(axis=1)
        ],
        errors="ignore",
    )

    vars_to_remove = ["HeuBug", "RealBugCount", "HeuBugCount"]
    remove_variables(dataset_trn, vars_to_remove)
    remove_variables(dataset_tst, vars_to_remove)

    dataset_trn = dataset_trn.rename(columns={"RealBug": "target"})
    dataset_tst = dataset_tst.rename(columns={"RealBug": "target"})

    dataset_trn["target"] = dataset_trn["target"].astype(bool)
    dataset_tst["target"] = dataset_tst["target"].astype(bool)

    dataset_trn, dataset_tst, mapping = map_indexes_to_int(dataset_trn, dataset_tst)

    features_names = dataset_trn.drop(columns=["target"]).columns.tolist()
    X_train = dataset_trn.loc[:, features_names].copy()

    Rnalytica = importr("Rnalytica")
    with (ro.default_converter + pandas2ri.converter).context():
        r_X_train = ro.conversion.get_conversion().py2rpy(X_train)
    selected_features = Rnalytica.AutoSpearman(r_X_train, StrVector(features_names))
    selected_features = list(selected_features) + ["target"]

    train = dataset_trn.loc[:, selected_features]
    test = dataset_tst.loc[:, selected_features]

    return train, test, mapping


def convert_original_dataset(dataset: Path = Path("./Dataset/original_dataset")):
    for csv in dataset.glob("*.csv"):
        file_name = csv.name
        project, *release = file_name.split("-")
        release = "-".join(release)

        df = pd.read_csv(csv, index_col=0)
        df = df.drop_duplicates()

        Path(f"./Dataset/project_dataset/{project}").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"./Dataset/project_dataset/{project}/{release}")


def organize_original_dataset():
    convert_original_dataset()
    path_truncate(Path("./Dataset/project_dataset/activemq"))
    path_truncate(Path("./Dataset/project_dataset/hbase"))
    path_truncate(Path("./Dataset/project_dataset/hive"))
    path_truncate(Path("./Dataset/project_dataset/lucene"))
    path_truncate(Path("./Dataset/project_dataset/wicket"))


def path_truncate(project, base="src/"):
    print(f"Project: {project.name}")
    for path in project.glob("*.csv"):
        df = pd.read_csv(path, index_col="File")
        df.index = df.index.map(lambda x: split_path(base, x))
        df.to_csv(path)


def split_path(base, path):
    if base in path:
        _, *tail = path.split(base)
        return base + base.join(tail)
    else:
        return path


def prepare_release_dataset():
    projects = all_dataset()
    for project, releases in projects.items():
        for i, release in enumerate(releases):
            dataset_trn, dataset_tst, mapping = preprocess(project, release)
            save_folder = f"./Dataset/release_dataset/{project}@{i}"
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            dataset_trn.to_csv(save_folder + "/train.csv", index=True, header=True)
            dataset_tst.to_csv(save_folder + "/test.csv", index=True, header=True)

            # Save mapping as csv
            mapping_df = pd.DataFrame.from_dict(mapping, orient="index")
            mapping_df.to_csv(save_folder + "/mapping.csv", index=True, header=False)


def preprocess_test():
    projects = all_dataset()
    for project, releases in projects.items():
        for i, release in enumerate(releases):
            dataset_trn, dataset_tst, mapping = preprocess(project, release)

            golden_dataset_trn = pd.read_csv(
                f"./Dataset/release_dataset/{project}@{i}/train.csv", index_col=0
            )

            assert (
                set(golden_dataset_trn.columns) == set(dataset_trn.columns)
            ), f"{set(golden_dataset_trn.columns) - set(dataset_trn.columns)} | {set(dataset_trn.columns) - set(golden_dataset_trn.columns)}"


if __name__ == "__main__":
    organize_original_dataset()
    prepare_release_dataset()
    # preprocess_test()
