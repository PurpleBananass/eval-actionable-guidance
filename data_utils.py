from pathlib import Path
import natsort
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, StrVector
from sklearn.preprocessing import MinMaxScaler

Rnalytica = importr("Rnalytica")


def remove_variables(df, vars_to_remove):
    df.drop(vars_to_remove, axis=1, inplace=True, errors="ignore")


def get_df(project: str, release: str, path_data: str = "project_dataset"):
    df = pd.read_csv(f"{path_data}/{project}/{release}", index_col=0)
    return df


def map_indexes_to_int(train_df, test_df, validation_df):
    all_files = (
        train_df.index.append(test_df.index).append(validation_df.index).unique()
    )
    file_to_int = {file: i for i, file in enumerate(all_files)}
    int_to_file = {i: file for file, i in file_to_int.items()}  # 역매핑

    train_df.index = train_df.index.map(file_to_int)
    test_df.index = test_df.index.map(file_to_int)
    validation_df.index = validation_df.index.map(file_to_int)

    return train_df, test_df, validation_df, int_to_file  # 역매핑도 반환


def map_indexes_to_file(df, int_to_file):
    df.index = df.index.map(int_to_file)
    return df


def preprocess(project, releases: list[str]):
    dataset_trn = get_df(project, releases[0])
    dataset_tst = get_df(project, releases[1])
    dataset_val = get_df(project, releases[2])

    duplicated_index_trn = dataset_trn.index.duplicated(keep="first")
    duplicated_index_tst = dataset_tst.index.duplicated(keep="first")
    duplicated_index_val = dataset_val.index.duplicated(keep="first")

    dataset_trn = dataset_trn[~duplicated_index_trn]
    dataset_tst = dataset_tst[~duplicated_index_tst]
    dataset_val = dataset_val[~duplicated_index_val]

    # dataset_tst에서 dataset_trn에 존재하는 동일한 인덱스와 모든 칼럼의 값이 동일한 행을 제거
    dataset_tst = dataset_tst.drop(
        dataset_tst.index[
            dataset_tst.isin(dataset_trn.to_dict(orient="list")).all(axis=1)
        ],
        errors="ignore",
    )

    # dataset_tst = dataset_tst[dataset_tst["RealBug"] == 1]
    # dataset_val과 dataset_tst는 동일한 파일이 있어야 함
    dataset_val = dataset_val[dataset_val.index.isin(dataset_tst.index)]

    vars_to_remove = ["HeuBug", "RealBugCount", "HeuBugCount"]
    remove_variables(dataset_trn, vars_to_remove)
    remove_variables(dataset_tst, vars_to_remove)
    remove_variables(dataset_val, vars_to_remove)

    dataset_trn = dataset_trn.rename(columns={"RealBug": "target"})
    dataset_tst = dataset_tst.rename(columns={"RealBug": "target"})
    dataset_val = dataset_val.rename(columns={"RealBug": "target"})

    dataset_trn, dataset_tst, dataset_val, int_to_file = map_indexes_to_int(
        dataset_trn, dataset_tst, dataset_val
    )

    features_names = dataset_trn.columns.tolist()[:-1]
    X_train = dataset_trn.loc[:, features_names].copy()
    with (ro.default_converter + pandas2ri.converter).context():
        r_X_train = ro.conversion.get_conversion().py2rpy(
            X_train
        )  # rpy2의 dataframe으로 변환
    selected_features = Rnalytica.AutoSpearman(r_X_train, StrVector(features_names))
    selected_features = list(selected_features) + ["target"]
    train = dataset_trn.loc[:, selected_features]
    test = dataset_tst.loc[:, selected_features]
    val = dataset_val.loc[:, selected_features]

    print(f"{project}/{releases[0]}: {len(selected_features)} features selected")
    return train, test, val


def project_dataset(project: Path):
    releases = []
    for release_csv in project.glob("*.csv"):
        release = release_csv.name
        releases.append(release)
    releases = natsort.natsorted(releases)
    k_releases = []
    window = 3
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


def prepare_release_dataset():
    projects = all_dataset()
    for project, releases in projects.items():
        for i, release in enumerate(releases):
            dataset_trn, dataset_tst, dataset_val = preprocess(project, release)
            save_folder = f"release_dataset/{project}@{i}"
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            dataset_trn.to_csv(save_folder + "/train.csv", index=True, header=True)
            dataset_tst.to_csv(save_folder + "/test.csv", index=True, header=True)
            dataset_val.to_csv(save_folder + "/val.csv", index=True, header=True)


def read_dataset(normalize=False):
    save_folder = "release_dataset"
    projects = {}
    for project in Path(save_folder).iterdir():
        if not project.is_dir():
            continue
        train = pd.read_csv(project / "train.csv", index_col=0)
        test = pd.read_csv(project / "test.csv", index_col=0)
        val = pd.read_csv(project / "val.csv", index_col=0)

        if normalize:
            
            original_dtypes = train.dtypes
            X_train = train.iloc[:, train.columns != "target"]
            y_train = train["target"]
            X_test = test.iloc[:, test.columns != "target"]
            y_test = test["target"]
            X_val = val.iloc[:, val.columns != "target"]
            y_val = val["target"]

            scaler = MinMaxScaler()
            scaler.fit(X_train.values)
            X_train_norm = scaler.transform(X_train.values)
            X_test_norm = scaler.transform(X_test.values)
            X_val_norm = scaler.transform(X_val.values)

            train = pd.DataFrame(X_train_norm, columns=X_train.columns, index=X_train.index)
            train["target"] = y_train
            train = train.astype(original_dtypes)
            test = pd.DataFrame(X_test_norm, columns=X_test.columns, index=X_test.index)
            test["target"] = y_test
            test = test.astype(original_dtypes)
            val = pd.DataFrame(X_val_norm, columns=X_val.columns, index=X_val.index)
            val["target"] = y_val
            val = val.astype(original_dtypes)

            projects[project.name] = [train, test, val, scaler]
        else:
            projects[project.name] = [train, test, val]
    return projects

def inverse_transform(df: pd.DataFrame, scaler):
    origianl_dtypes = df.dtypes
    inversed = scaler.inverse_transform(df.values)
    inversed_df = pd.DataFrame(inversed, columns=df.columns, index=df.index)
    inversed_df = inversed_df.astype(origianl_dtypes)
    return inversed_df

def split_path(base, path):
    if base in path:
        _, *tail = path.split(base)
        return base + base.join(tail)
    else:
        return path


def path_truncate(project, base="src/"):
    print(f"Project: {project.name}")
    for path in project.glob("*.csv"):
        df = pd.read_csv(path, index_col="File")
        df.index = df.index.map(lambda x: split_path(base, x))
        df.to_csv(path)


def convert_original_dataset(dataset: Path = Path("./original_dataset")):
    for csv in dataset.glob("*.csv"):
        file_name = csv.name
        project, *release = file_name.split("-")
        release = "-".join(release)
        # print(project, release)

        df = pd.read_csv(csv, index_col=0)
        df = df.drop_duplicates()

        # save to csv
        Path(f"./project_dataset/{project}").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"./project_dataset/{project}/{release}")


def organize_original_dataset():
    convert_original_dataset()
    path_truncate(Path("./project_dataset/activemq"))
    path_truncate(Path("./project_dataset/hbase"))
    path_truncate(Path("./project_dataset/hive"))
    path_truncate(Path("./project_dataset/lucene"))
    path_truncate(Path("./project_dataset/wicket"))


if __name__ == "__main__":
    # organize_original_dataset()
    # prepare_release_dataset()
    
    # Inverse가 제대로 되는지 확인
    projects = read_dataset()
    project = projects["activemq@0"]
    train, test, val, scaler = project
    X_train = train.iloc[:, train.columns != "target"]
    y_train = train["target"]
    X_test = test.iloc[:, test.columns != "target"]
    y_test = test["target"]

    X_train_inv = inverse_transform(X_train, scaler)
    X_test_inv = inverse_transform(X_test, scaler)

    projects_no_norm = read_dataset(normalize=False)
    project_no_norm = projects_no_norm["activemq@0"]
    train_no_norm, test_no_norm, val_no_norm = project_no_norm
    X_train_no_norm = train_no_norm.iloc[:, train_no_norm.columns != "target"]
    y_train_no_norm = train_no_norm["target"]
    X_test_no_norm = test_no_norm.iloc[:, test_no_norm.columns != "target"]
    y_test_no_norm = test_no_norm["target"]

    print(X_train_inv.equals(X_train_no_norm))
    print(X_test_inv.equals(X_test_no_norm))
    print(y_train.equals(y_train_no_norm))
    print(y_test.equals(y_test_no_norm))



