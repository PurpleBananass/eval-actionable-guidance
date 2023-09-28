# %% import packages
import json
from matplotlib.text import Text
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import (
    get_true_positives,
    read_dataset,
    load_historical_changes,
    get_release_names,
)
from hyparams import PLOTS, PLANS, EXPERIMENTS, RESULTS

DISPLAY_NAME_2 = {
    "DeFlip": "DeFlip",
    "LIMEHPO": "LIME-HPO",
    "SQAPlanner_confidence": "SQAPlanner(conf.)",
    "SQAPlanner_coverage": "SQAPlanner(cove.)",
    "SQAPlanner_lift": "SQAPlanner(lift)",
    "TimeLIME": "TimeLIME",
}


# %% Common index with DeFlip and Other explainers
projects = read_dataset()
project_list = list(sorted(projects.keys()))
explainers = [
    "LIMEHPO",
    "TimeLIME",
    "SQAPlanner_coverage",
    "SQAPlanner_confidence",
    "SQAPlanner_lift",
]

deflip_similarity = []
unactionable_ratio = []
for explainer in explainers:
    for project in project_list:
        _, test = projects[project]
        test = test.drop(columns=["target"])

        exp_path = Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
        exp = pd.read_csv(exp_path, index_col=0)
        exp = exp.dropna()
        exp_index = exp.index.tolist()

        deflip_path = Path(EXPERIMENTS) / project / "DeFlip.csv"
        deflip = pd.read_csv(deflip_path, index_col=0)
        deflip = deflip.dropna()
        deflip_index = deflip.index.tolist()
        common_index = list(set(deflip_index) & set(exp_index))

        # Flipping proposed changes와 DeFlip의 common index

        for test_name in common_index:
            deflip_instance = deflip.loc[test_name]
            exp_instance = exp.loc[test_name]
            test_instance = test.loc[test_name]

            deflip_change_features = test_instance[
                deflip_instance != test_instance
            ].index.tolist()
            exp_change_features = test_instance[
                exp_instance != test_instance
            ].index.tolist()
            # jaccard similarity
            jaccard = len(set(deflip_change_features) & set(exp_change_features)) / len(
                set(deflip_change_features) | set(exp_change_features)
            )
            deflip_similarity.append(
                {
                    "Project": project,
                    "Explainer": DISPLAY_NAME_2[explainer],
                    "Instance": test_name,
                    "Value": jaccard,
                }
            )
# %%
deflip_similarity_df = pd.DataFrame(deflip_similarity)

similarity_with_limehpo = deflip_similarity_df.loc[
    deflip_similarity_df["Explainer"] == "LIME-HPO", "Value"
].mean()
similarity_with_timelime = deflip_similarity_df.loc[
    deflip_similarity_df["Explainer"] == "TimeLIME", "Value"
].mean()
similarity_with_sqa_conf = deflip_similarity_df.loc[
    deflip_similarity_df["Explainer"] == "SQAPlanner(conf.)", "Value"
].mean()
similarity_with_sqa_cov = deflip_similarity_df.loc[
    deflip_similarity_df["Explainer"] == "SQAPlanner(cove.)", "Value"
].mean()
similarity_with_sqa_lift = deflip_similarity_df.loc[
    deflip_similarity_df["Explainer"] == "SQAPlanner(lift)", "Value"
].mean()

print(f"Similarity with LIME-HPO: {similarity_with_limehpo*100:.2f}")
print(f"Similarity with TimeLIME: {similarity_with_timelime*100:.2f}")
print(f"Similarity with SQAPlanner(conf.): {similarity_with_sqa_conf*100:.2f}")
print(f"Similarity with SQAPlanner(cove.): {similarity_with_sqa_cov*100:.2f}")
print(f"Similarity with SQAPlanner(lift): {similarity_with_sqa_lift*100:.2f}")


# %%
UNACTIONABLES = [
    "MAJOR_COMMIT",
    "MAJOR_LINE",
    "MINOR_COMMIT",
    "MINOR_LINE",
    "OWN_COMMIT",
    "OWN_LINE",
    "ADEV",
    "Added_lines",
    "Del_lines",
]
unactionable_ratio = {}
for explainer in ["DeFlip", "DeFlip_actionable"] + explainers:
    unactionable_count = 0
    total_count = 0
    for project in project_list:
        _, test = projects[project]
        test = test.drop(columns=["target"])
        if explainer in ["DeFlip", "DeFlip_actionable"]:
            exp_path = Path(EXPERIMENTS) / project / f"{explainer}.csv"
        else:
            exp_path = Path(EXPERIMENTS) / project / f"{explainer}_all.csv"
        exp = pd.read_csv(exp_path, index_col=0)
        exp = exp.dropna()
        exp_index = exp.index.tolist()

        for test_name in exp_index:
            exp_instance = exp.loc[test_name]
            test_instance = test.loc[test_name]
            exp_change_features = test_instance[
                exp_instance != test_instance
            ].index.tolist()
            if len(exp_change_features) == 0:
                continue
            for feature in exp_change_features:
                if feature in UNACTIONABLES:
                    unactionable_count += 1
                    break

            total_count += 1
    unactionable_ratio[explainer] = unactionable_count / total_count

# %%
unactionable_ratio
# %%
# %% Initialize the list to store all explainer pairs similarity
all_explainers_similarity = []

# %% Compare each explainer with every other explainer
for i, explainer_1 in enumerate(["DeFlip"] + explainers):
    for j, explainer_2 in enumerate(["DeFlip"] + explainers):
        # Skip the comparison if it's with itself or if this pair has been compared already
        if i >= j:
            continue

        for project in project_list:
            _, test = projects[project]
            test = test.drop(columns=["target"])

            exp_path_1 = Path(EXPERIMENTS) / project / f"{explainer_1}.csv"
            exp_1 = pd.read_csv(exp_path_1, index_col=0)
            exp_1 = exp_1.dropna()
            exp_index_1 = exp_1.index.tolist()

            exp_path_2 = Path(EXPERIMENTS) / project / f"{explainer_2}.csv"
            exp_2 = pd.read_csv(exp_path_2, index_col=0)
            exp_2 = exp_2.dropna()
            exp_index_2 = exp_2.index.tolist()

            common_index = list(set(exp_index_1) & set(exp_index_2))

            for test_name in common_index:
                exp_instance_1 = exp_1.loc[test_name]
                exp_instance_2 = exp_2.loc[test_name]
                test_instance = test.loc[test_name]

                exp_change_features_1 = test_instance[
                    exp_instance_1 != test_instance
                ].index.tolist()
                exp_change_features_2 = test_instance[
                    exp_instance_2 != test_instance
                ].index.tolist()

                # Jaccard similarity
                jaccard = len(
                    set(exp_change_features_1) & set(exp_change_features_2)
                ) / len(set(exp_change_features_1) | set(exp_change_features_2))

                all_explainers_similarity.append(
                    {
                        "Project": project,
                        "Explainer_1": DISPLAY_NAME_2[explainer_1],
                        "Explainer_2": DISPLAY_NAME_2[explainer_2],
                        "Instance": test_name,
                        "Value": jaccard,
                    }
                )

# %%
all_explainers_similarity_df = pd.DataFrame(all_explainers_similarity)
all_explainers_similarity_df
# %%
average_similarity = (
    all_explainers_similarity_df.groupby(["Explainer_1", "Explainer_2"])["Value"]
    .mean()
    .reset_index()
)
average_similarity.rename(columns={"Value": "Average Jaccard Similarity"}, inplace=True)
average_similarity = average_similarity.sort_values(
    by=["Average Jaccard Similarity"], ascending=False
)
average_similarity["Average Jaccard Similarity"] = average_similarity[
    "Average Jaccard Similarity"
].apply(lambda x: f"{x:.2f}")
# %%
average_similarity.to_csv(Path(RESULTS) / "average_similarity.csv", index=False)
# %% Discussion: FPC로 제안된 룰 간 자카드 유사도
explainers = [
    "DeFlip",
    "LIMEHPO",
    "TimeLIME",
    "SQAPlanner_confidence",
]
similarities = {}
for explainer in explainers:
    similarities[explainer] = []
    for project in project_list:
        _, test = projects[project]
        test = test.drop(columns=["target"])

        exp_path = Path(EXPERIMENTS) / project / f"{explainer}.csv"
  
        exp = pd.read_csv(exp_path, index_col=0)
        exp = exp.dropna()
        exp_index = exp.index.tolist()

        for i, test_name_1 in enumerate(exp_index):
            for j, test_name_2 in enumerate(exp_index):
                if i >= j:
                    continue
                exp_instance_1 = exp.loc[test_name_1]
                exp_instance_2 = exp.loc[test_name_2]

                exp_change_features_1 = exp_instance_1[
                    exp_instance_1 != test.loc[test_name_1]
                ].index.tolist()
                exp_change_features_2 = exp_instance_2[
                    exp_instance_2 != test.loc[test_name_2]
                ].index.tolist()

                # Jaccard similarity
                jaccard = len(
                    set(exp_change_features_1) & set(exp_change_features_2)
                ) / len(set(exp_change_features_1) | set(exp_change_features_2))
                
                similarities[explainer].append(jaccard)

# %%
print(np.mean(similarities['DeFlip'])*100)
print(np.mean(similarities['LIMEHPO'])*100)
print(np.mean(similarities['TimeLIME'])*100)
print(np.mean(similarities['SQAPlanner_confidence']))
# %% Number of features 
num_features = {}
for explainer in explainers:
    num_features[explainer] = {}
    for project in project_list:
        _, test = projects[project]
        test = test.drop(columns=["target"])

        exp_path = Path(EXPERIMENTS) / project / f"{explainer}.csv"
  
        exp = pd.read_csv(exp_path, index_col=0)
        exp = exp.dropna()
        exp_index = exp.index.tolist()
    
        for test_name in exp_index:
            exp_instance = exp.loc[test_name]
    
            exp_change_features = exp_instance[
                exp_instance != test.loc[test_name]
            ].index.tolist()

            for feature in exp_change_features:
                num_features[explainer][feature] = num_features[explainer].get(feature, 0) + 1

# %%
num_features_df = pd.DataFrame(num_features)
num_features_df = num_features_df.fillna(0)
num_features_df = num_features_df.astype(int)
num_features_df 
# %%
deflip_num_features = num_features_df['DeFlip']
deflip_num_features = deflip_num_features.sort_values(ascending=False)
ratio = deflip_num_features / deflip_num_features.sum()
ratio = ratio.apply(lambda x: f"{x*100:.2f} %")
ratio = ratio[:5]
ratio
# %%
deflip_num_features = num_features_df['TimeLIME']
deflip_num_features = deflip_num_features.sort_values(ascending=False)
ratio = deflip_num_features / deflip_num_features.sum()
ratio = ratio.apply(lambda x: f"{x*100:.2f} %")
ratio = ratio[:5]
for name, value in ratio.items():
    print(f'{name} ({value})', end=' ')
# %%
deflip_num_features = num_features_df['LIMEHPO']
deflip_num_features = deflip_num_features.sort_values(ascending=False)
ratio = deflip_num_features / deflip_num_features.sum()
ratio = ratio.apply(lambda x: f"{x*100:.2f} %")
ratio = ratio[:5]
for name, value in ratio.items():
    print(f'{name} ({value})', end=' ')

# %%
deflip_num_features = num_features_df['SQAPlanner_confidence']
deflip_num_features = deflip_num_features.sort_values(ascending=False)
ratio = deflip_num_features / deflip_num_features.sum()
ratio = ratio.apply(lambda x: f"{x*100:.2f}")
ratio = ratio[:5]
for name, value in ratio.items():
    print(f'{name} ({value})', end=' ')
# %%
