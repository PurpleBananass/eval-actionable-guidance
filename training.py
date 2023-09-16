from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from tqdm import tqdm

from data_utils import read_dataset
from hyparams import SEED, MODELS, MODEL_EVALUATION

# def train_and_save_model(model, train, test, model_path):
#     model.fit(
#         train.iloc[:, train.columns != "target"].values, train["target"].values
#     )
#     with open(model_path, "wb") as f:
#         pickle.dump(model, f)

# def evaluate_metrics(model, X, y):
#     y_pred = model.predict(X)
#     y_proba = model.predict_proba(X)[:, 1]

#     y_pred = y_pred.astype(bool)
    
#     return {
#         'AUC-ROC': roc_auc_score(y, y_proba),
#         'F1-score': f1_score(y, y_pred),
#         'Precision': precision_score(y, y_pred),
#         'Recall': recall_score(y, y_pred),
#         'TP': sum(y_pred & y) / sum(y),
#         '# of TP': sum(y_pred & y),
#     }

# def load_and_evaluate_model(model_path, train, test):
#     with open(model_path, "rb") as f:
#         loaded_model = pickle.load(f)

#     test_metrics = evaluate_metrics(loaded_model, test.iloc[:, test.columns != 'target'].values, test['target'].values)
    
#     # print(f"Test metrics: {test_metrics}")

#     return test_metrics

# def train_single_project(project, train, test, metrics={}):
#     models_path = Path(f"{MODELS}/{project}") 
#     models_path.mkdir(parents=True, exist_ok=True)

#     # Target label ratio
#     print(f"Target label ratio: {sum(train['target']) / len(train)}")
#     print(f"Target label ratio: {sum(test['target']) / len(test)}")

#     for model, model_name in [
#         (RandomForestClassifier(n_estimators=100, random_state=SEED), "RandomForest"),
#         (XGBClassifier(n_estimators=100, random_state=SEED), "XGBoost"),
#         (CatBoostClassifier(verbose=False, random_state=SEED), "CatBoost"),
#     ]:
#         model_path = models_path / f"{model_name}.pkl"

#         if not Path.exists(model_path):
#             train_and_save_model(model, train, test, model_path)

#         # print(f"Working on {project} with {model_name}...")
#         test_metrics = load_and_evaluate_model(model_path, train, test)
#         metrics[model_name][project] = test_metrics

#     return metrics


# def train_single_project(project, train, test):
#     X_train = train.iloc[:, train.columns != 'target'].values
#     y_train = train['target'].values
#     X_test = test.iloc[:, test.columns != 'target'].values
#     y_test = test['target'].values
    
#     models_path = Path(f"{MODELS}/{project}")
#     models_path.mkdir(parents=True, exist_ok=True)

#     for ratio in [10/1, 5/1, 3/1, 2/1, 1/1]:
#         smote = SMOTE(sampling_strategy=ratio, random_state=SEED)
#         X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
#         for model, model_name in [
#             (RandomForestClassifier(n_estimators=100, random_state=SEED), "RandomForest"),
#             (XGBClassifier(n_estimators=100, random_state=SEED), "XGBoost"),
#             (CatBoostClassifier(random_state=SEED), "CatBoost"),
#         ]:
#             model_path = models_path / f"{model_name}_ratio_{int(ratio)}.pkl"
            
#             if not Path.exists(model_path):
#                 train_and_save_model(model, X_train_resampled, y_train_resampled, X_test, y_test, model_path)
            
#             print(f"Working on {project} with {model_name} at ratio {ratio}...")
#             load_and_evaluate_model(model_path, train, test)

# def train_all_project():
#     projects = read_dataset()
#     Path(MODEL_EVALUATION).mkdir(parents=True, exist_ok=True)
#     metrics = {
#         "RandomForest": {},
#         "XGBoost": {},
#         "CatBoost": {},
#     }
#     for project in projects:
#         train, test = projects[project]
#         metrics = train_single_project(project, train, test, metrics)

#     # Save metrics per model as csv
#     for model_name in metrics:
#         model_metrics = metrics[model_name]
#         df = pd.DataFrame(model_metrics)
#         df.to_csv(f"{MODEL_EVALUATION}/{model_name}.csv")
        
# if __name__ == "__main__":
#     train_all_project()


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return precision_score(y_test, y_pred)

def train_single_project(project, train, test):
    X_train = train.iloc[:, train.columns != 'target'].values
    y_train = train['target'].values
    X_test = test.iloc[:, test.columns != 'target'].values
    y_test = test['target'].values
    
    models_path = Path(f"{MODELS}/{project}")
    models_path.mkdir(parents=True, exist_ok=True)

    best_ratios = {}

    for model, model_name in [
        (RandomForestClassifier(n_estimators=100, random_state=SEED), "RandomForest"),
        (XGBClassifier(n_estimators=100, random_state=SEED), "XGBoost"),
        (CatBoostClassifier(random_state=SEED, verbose=0), "CatBoost"),
    ]:
        best_precision = 0
        best_ratio = None

        for ratio in [10/1, 5/1, 3/1, 2/1, 1/1]:
            smote = SMOTE(sampling_strategy=ratio, random_state=SEED)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            current_precision = train_and_evaluate_model(model, X_train_resampled, y_train_resampled, X_test, y_test)
            
            print(f"Working on {project} with {model_name} at ratio {ratio}... Precision: {current_precision}")
            
            if current_precision > best_precision:
                best_precision = current_precision
                best_ratio = ratio

        best_ratios[model_name] = best_ratio
        print(f"Best ratio for {model_name} on project {project} is {best_ratio} with precision {best_precision}")
        
    return best_ratios

def train_all_project():
    projects = read_dataset()
    best_ratios_per_project = {}

    for project in projects:
        train, test = projects[project]
        best_ratios = train_single_project(project, train, test)
        best_ratios_per_project[project] = best_ratios

    return best_ratios_per_project

if __name__ == "__main__":
    best_ratios = train_all_project()
    print("Best SMOTE ratios for each project and model:", best_ratios)