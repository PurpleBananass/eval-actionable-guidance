import pickle
import re
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import read_dataset
from TimeLIME import extract_name_from_condition

# Concat outputs ["LIMEHPO", "SQAPlanner", "TimeLIME"]


def run_single_dataset():
    projects = read_dataset()
    models_path = Path("./models")
    models_path.mkdir(parents=True, exist_ok=True)
    output_limehpo = Path("./output/LIMEHPO")
    output_sqaplanner = Path("./output/SQAPlanner")
    output_timelime = Path("./output/TimeLIME")
    project_list = list(projects.keys())

    for project in output_timelime.iterdir():
        if not project.is_dir():
            continue
        print(project)
        for test_file in project.iterdir():
            instance_number = test_file.stem
            df = pd.read_csv(test_file)
            for i in range(len(df)):
                row = df.iloc[i]
                flipped_range = flip_feature_range(
                    row["feature"], row["min"], row["max"], row["importance"], row["rule"]
                )
                print(f"{row['value']} ({row['left']}, {row['right']}) {flipped_range} {row['rec']}")
                if "Error" in flipped_range:
                    print(instance_number, i)
            print("-------------")
        exit()
            

            
def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    # Case: a < feature <= b
    match = re.search(r'([\d.]+) < ' + re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if match:
        a, b = map(float, match.groups())
        if importance > 0:
            return f'{min_val} <= {feature} <= {a}', (min_val, a)
        else:
            return f'{b} < {feature} <= {max_val}', (b, max_val)
    
    # Case: feature > a
    match = re.search(re.escape(feature) + r' > ([\d.]+)', rule_str)
    if match:
        a = float(match.group(1))
        return f'{min_val} <= {feature} <= {a}', (min_val, a)
    
    # Case: feature <= b
    match = re.search(re.escape(feature) + r' <= ([\d.]+)', rule_str)
    if match:
        b = float(match.group(1))
        return f'{b} < {feature} <= {max_val}', (b, max_val)



if __name__ == "__main__":
    run_single_dataset()
