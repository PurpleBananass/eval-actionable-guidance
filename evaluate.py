from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
from data_utils import get_true_positives, load_historical_changes, read_dataset, read_dataset2, get_model
from pathlib import Path
from hyparams import PROPOSED_CHANGES, EXPERIMENTS
from scipy.spatial.distance import mahalanobis
from itertools import product
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from flip_exp import get_flip_rates

def generate_efficient_combinations(data):
    # 각 피처의 반복 횟수를 계산
    feature_names = list(data.keys())
    feature_values = list(data.values())
    
    # 각 피처의 조합 수 계산
    n_combinations = np.prod([len(values) for values in feature_values])
    combo_dict = {}

    # 각 피처에 대해 조합을 구성하는 방식으로 데이터를 생성
    repeat_factor = n_combinations
    for feature, values in zip(feature_names, feature_values):
        # 현재 피처의 반복 횟수를 설정
        repeat_factor //= len(values)
        tile_factor = n_combinations // (repeat_factor * len(values))

        # numpy를 사용해 효율적으로 반복 및 타일링으로 조합 생성
        combo_dict[feature] = np.tile(np.repeat(values, repeat_factor), tile_factor)

    # 최종 결과를 DataFrame으로 변환
    return pd.DataFrame(combo_dict)

def generate_all_combinations(data):
    # 가능한 모든 조합을 담을 리스트
    combinations = []
    feature_values = []
    for feature in data:
        feature_values.append(data[feature])
    combinations = list(product(*feature_values))
    
    df =  pd.DataFrame(combinations, columns=data.keys())
    return df

def plan_similarity(project, model_type, explainer):
    results = {}
    plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}" / "plans_all.json"
    flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"
    with open(plan_path, "r") as f:
        plans = json.load(f)

    experiment = pd.read_csv(flip_path, index_col=0)
    drops = experiment.dropna().index.to_list()
    model = get_model(project, model_type)
    train, test = read_dataset()[project]
    scaler = StandardScaler()
    scaler.fit(train.drop("target", axis=1).values)
    for test_idx in drops:
        
        if str(test_idx) in plans:
            original = test.loc[test_idx, test.columns != "target"]
            original_scaled = scaler.transform([original])
            pred_o = model.predict(original_scaled)[0]
            row = experiment.loc[[test_idx], :]
            row_scaled = scaler.transform(row.values)  # Scaler에 맞게 변환
            pred = model.predict(row_scaled)[0]  # 예측 수행 및 첫 번째 값 추출
            assert pred_o == 1, pred == 0

            plan = {}
            for feature in plans[str(test_idx)]:
                if experiment.loc[test_idx, feature] != original[feature]:
                    plan[feature] = plans[str(test_idx)][feature]

            flipped =  experiment.loc[test_idx, [feature for feature in plan]]
       
            min_changes = [ plan[feature][0] for feature in plan ]
            min_changes = pd.Series(min_changes, index=flipped.index)
            combi = generate_all_combinations(plan)


            score = normalized_mahalanobis_distance(combi, flipped, min_changes)
            results[test_idx] = { 'score': score }

    return results

def normalized_mahalanobis_distance(df, x, y):
    # 상수 피처 제거
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0
    
    # 데이터 표준화
    standardized_df = (df - df.mean()) / df.std()
    
    # x와 y 벡터도 표준화
    x_standardized = [(x[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns]
    y_standardized = [(y[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns]
    
    # 공분산 행렬의 역행렬 계산
    cov_matrix = np.cov(standardized_df.T)
    if cov_matrix.ndim == 0:
        inv_cov_matrix = np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    # x와 y 간의 마할라노비스 거리 계산
    distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)
    
    min_vector = np.array([min(df[feature]) for feature in df.columns])
    max_vector = np.array([max(df[feature]) for feature in df.columns])

    min_vector_standardized = [(min_vector[i] - df[feature].mean()) / df[feature].std() for i, feature in enumerate(df.columns)]
    max_vector_standardized = [(max_vector[i] - df[feature].mean()) / df[feature].std() for i, feature in enumerate(df.columns)]

    max_vector_distance = mahalanobis(min_vector_standardized, max_vector_standardized, inv_cov_matrix)
    # min_values = standardized_df.min()
    # max_values = standardized_df.max()
    # max_distance = np.linalg.norm(max_values - min_values)

    # normalized_distance = distance / max_distance if max_distance != 0 else 0
    # return normalized_distance
    
    # # 벡터화된 최대 마할라노비스 거리 계산
    # # 모든 쌍에 대해 거리 계산
    # pairwise_distances = cdist(standardized_df, standardized_df, metric='mahalanobis', VI=inv_cov_matrix)
    
    # # 자기 자신과의 거리를 제외한 최대 거리 구하기
    # np.fill_diagonal(pairwise_distances, 0)
    # max_distance = np.max(pairwise_distances)

    # assert round(max_distance, 8) == round(max_vector_distance, 8), f"{max_distance} != {max_vector_distance}"
    
    # 정규화된 거리 계산
    normalized_distance = distance / max_vector_distance if max_vector_distance != 0 else 0
    
    return normalized_distance
def mahalanobis_all(df, x):
    # 상수 피처 제거
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0
    
    # 데이터 표준화
    standardized_df = (df - df.mean()) / df.std()
    
    # x와 y 벡터도 표준화
    x_standardized = [(x[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns]
    
    # 공분산 행렬의 역행렬 계산
    cov_matrix = np.cov(standardized_df.T)
    if cov_matrix.ndim == 0:
        inv_cov_matrix = np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    min_vector = np.array([min(df[feature]) for feature in df.columns])
    max_vector = np.array([max(df[feature]) for feature in df.columns])

    min_vector_standardized = [(min_vector[i] - df[feature].mean()) / df[feature].std() for i, feature in enumerate(df.columns)]
    max_vector_standardized = [(max_vector[i] - df[feature].mean()) / df[feature].std() for i, feature in enumerate(df.columns)]

    max_vector_distance = mahalanobis(min_vector_standardized, max_vector_standardized, inv_cov_matrix)
    
    # x와 모든 y in df 간의 마할라노비스 거리 계산
    distances = []
    for _, y in df.iterrows():
        y_standardized = [(y[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns]
        distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)
        distances.append(distance / max_vector_distance if max_vector_distance != 0 else 0)

    return distances
    

def flip_feasibility(project, explainer, model_type):
    
    plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}" / "plans_all.json"
    flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"
    with open(plan_path, "r") as f:
        plans = json.load(f)

    flipped = pd.read_csv(flip_path, index_col=0)
    flipped = flipped.dropna()
    train, test = read_dataset()[project]
    exist_indices = train.index.intersection(test.index)
    deltas = (
        test.loc[exist_indices, test.columns != "target"]
        - train.loc[exist_indices, train.columns != "target"]
    )
    results = []
    
    for test_idx in flipped.index:
        
        if str(test_idx) in plans:
            original_row = test.loc[test_idx, test.columns != "target"]

            flipped_row = flipped.loc[test_idx, :]

            changed_features = {}
            for feature in plans[str(test_idx)]:
                if flipped_row[feature] != original_row[feature]:
                    changed_features[feature] = flipped_row[feature] - original_row[feature]

            changed_flipped = pd.Series(changed_features)

            nonzeros = deltas.loc[deltas[feature] != 0, changed_features.keys()]

            distances = mahalanobis_all(nonzeros, changed_flipped)
            results.append({ 'min': np.min(distances), 'max': np.max(distances), 'mean': np.mean(distances) })

    return results

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--rq1", action="store_true")
    argparser.add_argument("--rq2", action="store_true")
    argparser.add_argument("--rq3", action="store_true")
    argparser.add_argument("--explainer", type=str, default="all")

    args = argparser.parse_args()

    if args.explainer == "all":
        explainers = ['LIME', 'LIME-HPO', 'TimeLIME', 'SQAPlanner_confidence']
    else:
        explainers = args.explainer.split(" ")
    projects = read_dataset()
    if args.rq1:
        table = []
        for model_type in ["RandomForest", "XGBoost"]:
            for explainer in explainers:
                if explainer == "SQAPlanner_confidence":
                    result = get_flip_rates("SQAPlanner", "confidence", model_type, verbose=False)
                else:
                    result = get_flip_rates(explainer, None, model_type, verbose=False)
                table.append([model_type, explainer, result["Rate"]])

            # Add mean per model
            table.append([model_type, "Mean", np.mean([row[2] for row in table if row[0] == model_type])])
        print(tabulate(table, headers=["Model", "Explainer", "Flip Rate"]))
    
        # table to csv
        table = pd.DataFrame(table, columns=["Model", "Explainer", "Flip Rate"])
        table.to_csv('./evaluations/flip_rates.csv', index=False)
    

    if args.rq2:    
        table = []
        for model_type in ["RandomForest", "XGBoost"]:
            similarities = { explainer: pd.DataFrame() for explainer in explainers }
            for explainer in explainers:
                for project in projects:
                    result = plan_similarity(project, model_type, explainer)
                    df = pd.DataFrame(result).T
                    similarities[explainer] = pd.concat([similarities[explainer], df], axis=0, ignore_index=True)
            
            row = [model_type]
            for explainer in explainers:
                # Svae similarity to csv
                similarities[explainer].to_csv(f"./evaluations/accuracy/{model_type}_{explainer}.csv", index=False)
                row.append(similarities[explainer].median()['score'])
            table.append(row)
        print(tabulate(table, headers=["Model"]+explainers))
        # table to csv
        table = pd.DataFrame(table, columns=["Model"]+explainers)
        table.to_csv('./evaluations/accuracies.csv', index=False)

    if args.rq3:
        table = []
        for model_type in ["RandomForest", "XGBoost"]:
            for explainer in explainers:
                results = []
                for project in projects:
                    result = flip_feasibility(project, explainer, model_type)
                    results.extend(result)
                df = pd.DataFrame(results)
                # save to csv
                df.to_csv(f"./evaluations/feasibility/{model_type}_{explainer}.csv", index=False)
                table.append([model_type, explainer, df['min'].mean(), df['max'].mean(), df['mean'].mean()])
                print(table)
            # Add mean per model
            table.append([model_type, "Mean", np.mean([row[2] for row in table if row[0] == model_type]), np.mean([row[3] for row in table if row[0] == model_type]), np.mean([row[4] for row in table if row[0] == model_type])])
        print(tabulate(table, headers=["Model", "Explainer", "Min", "Max", "Mean"]))
        # table to csv
        table = pd.DataFrame(table, columns=["Model", "Explainer", "Min", "Max", "Mean"])
        table.to_csv('./evaluations/feasibility.csv', index=False)

    # plan_similarity("activemq@0", "RandomForest", "LIME")