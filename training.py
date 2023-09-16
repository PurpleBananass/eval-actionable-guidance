import json
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

def train_and_save_model(model, train, model_path, sample_strategy=None, class_weight=None):
    if sample_strategy:
        sm = SMOTE(random_state=SEED, sampling_strategy=sample_strategy)
        train.iloc[:, train.columns != "target"], train["target"] = sm.fit_resample(
            train.iloc[:, train.columns != "target"], train["target"]
        )

    if class_weight == "balanced":
        model.set_params(class_weight=class_weight)
    elif class_weight != None:
        model.set_params(class_weight={int(k): v for k, v in class_weight.items()})


    model.fit(
        train.iloc[:, train.columns != "target"].values, train["target"].values
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def evaluate_metrics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    y_pred = y_pred.astype(bool)
    
    return {
        'AUC-ROC': roc_auc_score(y, y_proba),
        'F1-score': f1_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'TP': sum(y_pred & y) / sum(y),
        '# of TP': sum(y_pred & y),
    }

def load_and_evaluate_model(model_path, train, test):
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    test_metrics = evaluate_metrics(loaded_model, test.iloc[:, test.columns != 'target'].values, test['target'].values)
    
    return test_metrics

def train_single_project(project, train, test, metrics={}):
    models_path = Path(f"{MODELS}/{project}") 
    models_path.mkdir(parents=True, exist_ok=True)

    with open("best_rf_params.json", "r") as f:
        best_params = json.load(f)
  

    for model, model_name in [
        (RandomForestClassifier(n_estimators=100, random_state=SEED), "RandomForest"),
        # (XGBClassifier(n_estimators=100, random_state=SEED), "XGBoost"),
        # (CatBoostClassifier(verbose=False, random_state=SEED), "CatBoost"),
    ]:
        model_path = models_path / f"{model_name}.pkl"

        if model_name == "RandomForest":
            smote_ratio = best_params.get(project, {}).get("smote_ratio")
            class_weight = best_params.get(project, {}).get("class_weight")

            
            # class_weight = {int(k): v for k, v in class_weight.items()} if class_weight else None
            
            n_majority = len(train[train['target'] == 0])

            if smote_ratio is None:
                sampling_strategy = None
            else:
                    
                n_minority_target = n_majority // smote_ratio  # Integer division
                sampling_strategy = {0: n_majority, 1: n_minority_target} 


        if not Path.exists(model_path):
            train_and_save_model(model, train, model_path, sampling_strategy, class_weight)

        # print(f"Working on {project} with {model_name}...")
        test_metrics = load_and_evaluate_model(model_path, train, test)
        metrics[model_name][project] = test_metrics

    return metrics


def train_all_project():
    projects = read_dataset()
    Path(MODEL_EVALUATION).mkdir(parents=True, exist_ok=True)
    metrics = {
        "RandomForest": {},
        "XGBoost": {},
        "CatBoost": {},
    }
    for project in projects:
        train, test = projects[project]
        metrics = train_single_project(project, train, test, metrics)

    # Save metrics per model as csv
    for model_name in metrics:
        model_metrics = metrics[model_name]
        df = pd.DataFrame(model_metrics)
        df.to_csv(f"{MODEL_EVALUATION}/{model_name}.csv")
        
if __name__ == "__main__":
    train_all_project()
