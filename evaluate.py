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
    
    min_values = standardized_df.min()
    max_values = standardized_df.max()
    max_distance = np.linalg.norm(max_values - min_values)

    normalized_distance = distance / max_distance if max_distance != 0 else 0
    return normalized_distance
    
    # 벡터화된 최대 마할라노비스 거리 계산
    # 모든 쌍에 대해 거리 계산
    pairwise_distances = cdist(standardized_df, standardized_df, metric='mahalanobis', VI=inv_cov_matrix)
    
    # 자기 자신과의 거리를 제외한 최대 거리 구하기
    np.fill_diagonal(pairwise_distances, 0)
    max_distance = np.max(pairwise_distances)
    
    # 정규화된 거리 계산
    normalized_distance = distance / max_distance if max_distance != 0 else 0
    
    return normalized_distance

def get_accuracy(explainers):
    accuracies = []
    projects = read_dataset()
    project_list = list(sorted(projects.keys()))
    for explainer in explainers:
        for project in project_list:
            accuracy = mean_accuracy_project(project, explainer)
            if accuracy is None:
                continue
            accuracies.append({
                'project': project,
                'accuracy': accuracy,
                'explainer': explainer
            })
    accuracies = pd.DataFrame(accuracies)  
    accuracies = accuracies.pivot(index='project', columns='explainer', values='accuracy') 
    accuracies.to_csv('./evaluations/accuracies.csv')
    return accuracies

def get_feasibility2(explainers):
    projects = read_dataset2()
    project_list = list(sorted(projects.keys()))
    total_feasibilities = []

    for explainer in explainers:
        for project in project_list:
            if len(projects[project]) < 3:
                continue
            train, test, valid = projects[project]
            defect_test = test[test['target'] == 1]
            clean_valid = valid[valid['target'] == 0]
            
            


            common_index = defect_test.index.intersection(clean_valid.index)
            print(set(defect_test.index).intersection(set(clean_valid.index)))
            
            # read the proposed changes
            if "DeFlip" in explainer:
                flip_path = Path(EXPERIMENTS) / project / f"{explainer}.csv"
                flipped_instances = pd.read_csv(flip_path, index_col=0)
                flipped_instances = flipped_instances.dropna()
                if len(flipped_instances) == 0:
                    continue
            else:
                plan_path = Path(PROPOSED_CHANGES) / project / explainer / "plans_all.json"
                with open(plan_path) as f:
                    plans = json.load(f)

            project_feasibilities = []
            print(f"Project: {project} Explainer: {explainer} Common Index: {len(common_index)}")
            for index in common_index:
                current = test.loc[index, test.columns != 'target']
                real_flipped = valid.loc[index, valid.columns != 'target']
                if "DeFlip" in explainer:
                    if index not in flipped_instances.index:
                        continue
                    flipped = flipped_instances.loc[index]
                    changed_features = current[current != flipped].index.tolist()
                else:
                    changed_features = list(plans[str(index)].keys())
                diff = current != real_flipped
                real_changed_features = diff[diff == True].index.tolist()

                # intersection / changed_features
                intersection = len(set(changed_features).intersection(set(real_changed_features)))
                feasibility = intersection / len(changed_features)
                project_feasibilities.append(feasibility)
            feasibility = np.mean(project_feasibilities)
            total_feasibilities.append(
                {
                    "Explainer": explainer,
                    "Value": feasibility,
                    "Project": project,
                }
            )
    total_feasibilities = pd.DataFrame(total_feasibilities)
    total_feasibilities = total_feasibilities.pivot(
        index="Project", columns="Explainer", values="Value"
    )
    total_feasibilities.to_csv('./evaluations/feasibilities2.csv')
    return total_feasibilities
        

def get_feasibility(explainers):
    projects = read_dataset()
    project_list = list(sorted(projects.keys()))
    total_feasibilities = []

    for explainer in explainers:
        for project in project_list:
            train, test = projects[project]
            test_instances = test.drop(columns=["target"])
            historical_mean_changes = load_historical_changes(project)["mean_change"]
            exp_path =  Path(EXPERIMENTS) / project / f"{explainer}.csv"
                
            flipped_instances = pd.read_csv(exp_path, index_col=0)
            flipped_instances = flipped_instances.dropna()
            if len(flipped_instances) == 0:
                continue

            project_feasibilities = []
            for index, flipped in flipped_instances.iterrows():
                current = test_instances.loc[index]
                diff = current != flipped
                diff = diff[diff == True]
                changed_features = diff.index.tolist()

                feasibilites = []
                for feature in changed_features:
                    if not historical_mean_changes[feature]:
                        historical_mean_changes[feature] = 0
                    flipping_proposed_change = abs(flipped[feature] - current[feature])

                    feasibility = 1 - (
                        flipping_proposed_change
                        / (flipping_proposed_change + historical_mean_changes[feature])
                    )
                    feasibilites.append(feasibility)
                feasibility = np.mean(feasibilites)
                project_feasibilities.append(feasibility)
            feasibility = np.mean(project_feasibilities)
            total_feasibilities.append(
                {
                    "Explainer": explainer,
                    "Value": feasibility,
                    "Project": project,
                }
            )
    total_feasibilities = pd.DataFrame(total_feasibilities)
    total_feasibilities = total_feasibilities.pivot(
        index="Project", columns="Explainer", values="Value"
    )
    total_feasibilities.to_csv('./evaluations/feasibilities.csv')
    return total_feasibilities

def mean_accuracy_project(project, explainer):
    plan_path = Path(PROPOSED_CHANGES) / project / explainer / "plans_all.json"
    with open(plan_path) as f:
        plans = json.load(f)
    projects = read_dataset()
    train, test = projects[project]
    test_instances = test.drop(columns=["target"])

    # read the flipped instances
    exp_path = Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
    flipped_instances = pd.read_csv(exp_path, index_col=0)
    flipped_instances = flipped_instances.dropna()
    if len(flipped_instances) == 0:
        return None
    results = []
    for index, flipped in flipped_instances.iterrows():
        current = test_instances.loc[index]
        changed_features = list(plans[str(index)].keys())
        diff = current != flipped
        diff = diff[diff == True]

        score = mean_accuracy_instance(
            current[changed_features], flipped[changed_features], plans[str(index)]
        )

        if score is None:
            continue

        results.append(score)
    # median of results
    results_np = np.array(results)
    return np.mean(results_np)

def compute_score(a1, a2, b1, b2, is_int):
    intersection_start = max(a1, b1)
    intersection_end = min(a2, b2)

    if intersection_start > intersection_end:
        return 1.0  # No intersection

    if is_int:
        intersection_cnt = intersection_end - intersection_start + 1
        union_cnt = (a2 - a1 + 1) + (b2 - b1 + 1) - intersection_cnt
        score = 1 - (intersection_cnt / union_cnt)
    else:
        intersection_length = intersection_end - intersection_start
        union_length = (a2 - a1) + (b2 - b1) - intersection_length
        if union_length == 0:
            return 0.0  # Identical intervals
        score = 1 - (intersection_length / union_length)

    return score


def mean_accuracy_instance(current: pd.Series, flipped: pd.Series, plans):
    scores = []
    for feature in plans:
        flipped_changed = flipped[feature] - current[feature]
        if flipped_changed == 0.0:
            continue

        min_val = min(plans[feature])
        max_val = max(plans[feature])

        a1, a2 = (
            (min_val, flipped[feature])
            if current[feature] < flipped[feature]
            else (flipped[feature], max_val)
        )

        score = compute_score(
            min_val, max_val, a1, a2, current[feature].dtype == "int64"
        )
        assert 0 <= score <= 1, f"Invalid score {score} for feature {feature}"
        scores.append(score)

    return np.mean(scores)


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
            for project in projects:
                # train, test = projects[project]
                # model = get_model(project, model_type)
                # true_positives = get_true_positives(model, train, test)
                # common_indices = set(true_positives.index)
                
                for explainer in explainers:
                    result = plan_similarity(project, model_type, explainer)
                    df = pd.DataFrame(result).T
                    similarities[explainer] = pd.concat([similarities[explainer], df], axis=0)
            
            row = [model_type]
            for explainer in explainers:
                row.append(similarities[explainer].mean()['score'])
            table.append(row)
        print(tabulate(table, headers=["Model"]+explainers))
        # table to csv
        table = pd.DataFrame(table, columns=["Model"]+explainers)
        table.to_csv('./evaluations/accuracies.csv', index=False)
