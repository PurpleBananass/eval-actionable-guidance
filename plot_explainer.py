# %%
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from data_utils import historical_changes, load_historical_changes, read_dataset

from hyparams import PLANS


# %%
def num_features_in_plan_of(target_explaienr, threshold=3):
    projects = read_dataset()

    candidates = {}
    for target_project in projects.keys():
        plan_file = Path(PLANS) / target_project / target_explaienr / "plans_all.json"

        with open(plan_file, "r") as f:
            plans = json.load(f)

        train, test = projects[target_project]
        train_x = train.loc[:, train.columns != "target"]
        test_x = test.loc[:, test.columns != "target"]

        for test_name in plans.keys():
            target_plan = plans[test_name]
            number_of_features = len(target_plan)
            flag = False
            for feature in target_plan.keys():
                if len(target_plan[feature]) < 4:
                    flag = True
            if number_of_features >= threshold and not flag:
                candidates[target_project] = candidates.get(target_project, {})
                candidates[target_project][test_name] = target_plan
    return candidates


# %%
explainers = ["SQAPlanner_confidence", "LIMEHPO", "TimeLIME"]
# explainers = ["TimeLIME"]
for explainer in explainers:
    candidates = num_features_in_plan_of(explainer, 2)
    print(explainer, len(candidates))
    for project in candidates.keys():
        if project in ["derby@0"]:
            print(project, len(candidates[project]))
            for test in candidates[project].keys():
                # print(test, list(candidates[project][test].keys()))
                    if test in ["493"]:
                        print(test, len(candidates[project][test]))
                        print(candidates[project][test])
                        print()

# %%
target_project = "derby@0"
target_test = 493
hist_changes = load_historical_changes(target_project)
train, test = read_dataset()[target_project]
target_instance = test.loc[target_test, test.columns != "target"]
target_instance = target_instance.to_frame()
target_instance["mean_change"] = hist_changes["mean_change"]
target_instance
# %%
deflip_ex = pd.read_csv("experiments/derby@0/DeFlip.csv", index_col=0)
deflip_ex = deflip_ex.loc[target_test, :]
deflip_ex - test.loc[target_test, test.columns != "target"]
deflip_ex[['MaxInheritanceTree', 'MaxNesting_Mean']]
# %%

time_lime_data = {
    1957: ["MAJOR_LINE", "CountClassCoupled", "CountDeclClass", "MaxInheritanceTree"],
    1958: ["MAJOR_LINE", "OWN_LINE", "MaxInheritanceTree", "CountDeclClass"],
    1960: ["MAJOR_LINE", "CountClassCoupled", "CountDeclClass", "OWN_LINE"],
    96: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    189: ["OWN_LINE", "MAJOR_LINE", "MaxInheritanceTree", "CountDeclClass"],
    212: ["MaxInheritanceTree", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    213: ["MAJOR_LINE", "MaxInheritanceTree", "CountDeclClass", "CountClassCoupled"],
    216: ["MAJOR_LINE", "MaxInheritanceTree", "CountDeclClass", "CountClassCoupled"],
    356: ["MAJOR_LINE", "MaxInheritanceTree", "CountClassCoupled", "CountDeclClass"],
    386: ["MaxInheritanceTree", "MAJOR_LINE", "CountClassCoupled", "CountDeclClass"],
    420: ["MAJOR_LINE", "MaxInheritanceTree", "CountDeclClass", "CountClassCoupled"],
    493: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    527: ["MAJOR_LINE", "OWN_LINE", "CountDeclClass", "CountClassCoupled"],
    1989: ["MAJOR_LINE", "OWN_LINE", "CountDeclClass", "CountClassCoupled"],
    1992: ["MaxInheritanceTree", "MAJOR_LINE", "CountDeclClass", "OWN_LINE"],
    688: ["MAJOR_LINE", "MaxInheritanceTree", "CountDeclClass", "CountClassCoupled"],
    2002: ["OWN_LINE", "MaxInheritanceTree", "MAJOR_LINE", "CountClassCoupled"],
    732: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    970: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "MaxInheritanceTree"],
    1042: ["OWN_LINE", "MaxInheritanceTree", "MAJOR_LINE", "CountDeclClass"],
    1158: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    1241: ["OWN_LINE", "MAJOR_LINE", "MaxInheritanceTree", "CountDeclClass"],
    1475: ["MAJOR_LINE", "MaxInheritanceTree", "CountClassCoupled", "CountDeclClass"],
    1549: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    1558: ["CountDeclClass", "MAJOR_LINE", "OWN_LINE", "CountClassCoupled"],
    2044: ["OWN_LINE", "MAJOR_LINE", "CountClassCoupled", "CountDeclClass"],
    1639: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    2089: ["OWN_LINE", "MAJOR_LINE", "CountClassCoupled", "CountDeclClass"],
    2103: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    2131: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    1767: ["MAJOR_LINE", "OWN_LINE", "CountDeclClass", "CountClassCoupled"],
    2163: ["MAJOR_LINE", "OWN_LINE", "CountDeclClass", "CountClassCoupled"],
    2166: ["MAJOR_LINE", "OWN_LINE", "MaxInheritanceTree", "CountClassCoupled"],
    2168: ["MAJOR_LINE", "OWN_LINE", "MaxInheritanceTree", "CountClassCoupled"],
    2169: ["OWN_LINE", "MAJOR_LINE", "CountDeclClass", "CountClassCoupled"],
    2173: ["OWN_LINE", "MAJOR_LINE", "CountClassCoupled", "CountDeclClass"],
    1881: ["OWN_LINE", "MAJOR_LINE", "CountClassCoupled", "CountDeclClass"],
}

sqa_planner_confidence_data = {
    27: ["CountLineComment", "Added_lines"],
    169: ["MaxNesting_Mean", "Added_lines"],
    247: ["RatioCommentToCode", "PercentLackOfCohesion"],
    341: ["CountClassDerived", "CountLineComment"],
    464: ["RatioCommentToCode", "Added_lines"],
    493: ["RatioCommentToCode", "CountLineComment"],
    783: ["MaxNesting_Mean", "PercentLackOfCohesion"],
    806: ["CountClassCoupled", "Added_lines"],
    913: ["RatioCommentToCode", "CountLineComment"],
    930: ["CountLineComment", "CountClassBase"],
    960: ["RatioCommentToCode", "Added_lines"],
    1004: ["MaxNesting_Mean", "Added_lines"],
    1005: ["RatioCommentToCode", "CountLineComment"],
    1043: ["CountInput_Mean", "Added_lines"],
    1045: ["CountInput_Mean", "Added_lines"],
    1075: ["CountClassCoupled", "Added_lines"],
    1116: ["RatioCommentToCode", "Added_lines"],
    1139: ["CountInput_Mean", "Added_lines"],
    1215: ["RatioCommentToCode", "Added_lines"],
    1334: ["CountClassDerived", "CountLineComment"],
    1437: ["CountLineComment", "Added_lines"],
    1441: ["MaxNesting_Mean", "Added_lines"],
    1519: ["RatioCommentToCode", "Added_lines"],
    2034: ["CountInput_Mean", "Added_lines"],
    2043: ["MaxNesting_Mean", "Added_lines"],
    2071: ["CountLineComment", "Added_lines"],
    2088: ["CountInput_Mean", "Added_lines"],
    1671: ["CountInput_Mean", "Added_lines"],
    2118: ["RatioCommentToCode", "CountLineComment"],
    2138: ["CountLineComment", "Added_lines"],
    2139: ["AvgLineComment", "Added_lines"],
    2147: ["CountInput_Mean", "Added_lines"],
    1813: ["RatioCommentToCode", "PercentLackOfCohesion"],
    2170: ["PercentLackOfCohesion", "Added_lines"],
    1818: ["RatioCommentToCode", "Added_lines"],
}

common_indices = set(time_lime_data.keys()).intersection(
    set(sqa_planner_confidence_data.keys())
)
common_data = {}
for index in common_indices:
    common_data[index] = {
        "TimeLIME": time_lime_data[index],
        "SQAPlanner_confidence": sqa_planner_confidence_data[index],
    }

# common_data에는 공통 테스트 인덱스와 각각의 특성 리스트가 저장됩니다.
print(common_data)
# %%
{
    "493": {
        "TimeLIME": {
            "OWN_LINE": [0.5815533980582519, 0.531553398058252, 0.481553398058252],
            "MAJOR_LINE": [2, 3, 10],
            "CountDeclClass": [2, 3, 13],
            "CountClassCoupled": [7, 8, 49]
        },
        "SQAPlanner_confidence": {
            "RatioCommentToCode": [4.9, 4.9, 7.299999999999992],
            "CountLineComment": [61, 60, 50]
        },
        "LIMEHPO": {
            "Added_lines": [7, 8, 1159],
            "CountDeclClassVariable": [3, 2, 0],
            "CountLineComment": [146, 145, 15],
            "PercentLackOfCohesion": [99, 98, 0],
            "MaxNesting_Min": [1, 2, 5]
        },
        "DeFlip": {
            "MaxInheritanceTree": [1.0],
            "MaxNesting_Mean": [1.2]
        }
    }
}

