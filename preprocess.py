import pandas as pd
from data_utils import all_dataset
from pathlib import Path
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, StrVector

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



def AutoSpearman(X_train, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5):
    """
    Code from `PyExplainer <https://github.com/awsm-research/PyExplainer/blob/f6e177c895d3f4570dbc0d35dfe7365a8f92e21d/pyexplainer/pyexplainer_pyexplainer.py#L23>`_.

    An automated feature selection approach that address collinearity and multicollinearity.
    For more information, please kindly refer to the `paper <https://ieeexplore.ieee.org/document/8530020>`_.

    Parameters
    ----------
    X_train : :obj:`pd.core.frame.DataFrame`
        The X_train data to be processed
    correlation_threshold : :obj:`float`
        Threshold value of correalation.
    correlation_method : :obj:`str`
        Method for solving the correlation between the features.
    VIF_threshold : :obj:`int`
        Threshold value of VIF score.
    """
    X_AS_train = X_train.copy()
    AS_metrics = X_AS_train.columns
    count = 1

    # (Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test.
    while True:
        corrmat = X_AS_train.corr(method=correlation_method)
        top_corr_features = corrmat.index
        abs_corrmat = abs(corrmat)

        # identify correlated metrics with the correlation threshold of the threshold
        highly_correlated_metrics = ((corrmat > correlation_threshold) | (corrmat < -correlation_threshold)) & (
            corrmat != 1)
        n_correlated_metrics = np.sum(np.sum(highly_correlated_metrics))
        if n_correlated_metrics > 0:
            # find the strongest pair-wise correlation
            find_top_corr = pd.melt(abs_corrmat, ignore_index=False)
            find_top_corr.reset_index(inplace=True)
            find_top_corr = find_top_corr[find_top_corr['value'] != 1]
            top_corr_index = find_top_corr['value'].idxmax()
            top_corr_i = find_top_corr.loc[top_corr_index, :]

            # get the 2 correlated metrics with the strongest correlation
            correlated_metric_1 = top_corr_i[0]
            correlated_metric_2 = top_corr_i[1]


            # compute their correlation with other metrics outside of the pair
            correlation_with_other_metrics_1 = np.mean(abs_corrmat[correlated_metric_1][
                [i for i in top_corr_features if
                 i not in [correlated_metric_1, correlated_metric_2]]])
            correlation_with_other_metrics_2 = np.mean(abs_corrmat[correlated_metric_2][
                [i for i in top_corr_features if
                 i not in [correlated_metric_1, correlated_metric_2]]])
          
            # select the metric that shares the least correlation outside of the pair and exclude the other
            if correlation_with_other_metrics_1 < correlation_with_other_metrics_2:
                exclude_metric = correlated_metric_2
            else:
                exclude_metric = correlated_metric_1

            count = count + 1
            AS_metrics = list(set(AS_metrics) - set([exclude_metric]))
            X_AS_train = X_AS_train[AS_metrics]
        else:
            break


    # (Part 2) Automatically select non-correlated metrics based on a Variance Inflation Factor analysis.

    # Prepare a dataframe for VIF
    X_AS_train = add_constant(X_AS_train)

    selected_features = X_AS_train.columns
    count = 1
    while True:
        # Calculate VIF scores
        vif_scores = pd.DataFrame([variance_inflation_factor(np.array(X_AS_train.values, dtype=float), i)
                                   for i in range(X_AS_train.shape[1])],
                                  index=X_AS_train.columns)
        # Prepare a final dataframe of VIF scores
        vif_scores.reset_index(inplace=True)
        vif_scores.columns = ['Feature', 'VIFscore']
        vif_scores = vif_scores.loc[vif_scores['Feature'] != 'const', :]
        vif_scores.sort_values(by=['VIFscore'], ascending=False, inplace=True, kind='mergesort')

        # Find features that have their VIF scores of above the threshold
        filtered_vif_scores = vif_scores[vif_scores['VIFscore']
                                         >= VIF_threshold]

        # Terminate when there is no features with the VIF scores of above the threshold
        if len(filtered_vif_scores) == 0:
            break

        # exclude the metric with the highest VIF score
        metric_to_exclude = list(filtered_vif_scores['Feature'].head(1))[0]

        count = count + 1

        selected_features = list(
            set(selected_features) - set([metric_to_exclude]))

        X_AS_train = X_AS_train.loc[:, selected_features]

    X_AS_train = X_AS_train.drop('const', axis=1)
    return X_AS_train


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

    # dataset_tst에서 dataset_trn에 존재하는 동일한 인덱스와 모든 칼럼의 값이 동일한 행을 제거
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


    # X_train_AS = AutoSpearman(X_train, VIF_threshold=5)
    # selected_features = X_train_AS.columns.tolist() + ["target"]

    

    Rnalytica = importr("Rnalytica")
    with (ro.default_converter + pandas2ri.converter).context():
        r_X_train = ro.conversion.get_conversion().py2rpy(
            X_train
        )  # rpy2의 dataframe으로 변환
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
            
            golden_dataset_trn = pd.read_csv(f"./Dataset/release_dataset/{project}@{i}/train.csv", index_col=0)
            golden_dataset_tst = pd.read_csv(f"./Dataset/release_dataset/{project}@{i}/test.csv", index_col=0)

            assert set(golden_dataset_trn.columns) == set(dataset_trn.columns), f"{set(golden_dataset_trn.columns) - set(dataset_trn.columns)} | {set(dataset_trn.columns) - set(golden_dataset_trn.columns)}" 


if __name__ == "__main__":
    organize_original_dataset()
    # prepare_release_dataset()
    preprocess_test()