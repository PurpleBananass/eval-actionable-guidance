import joblib
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm

from hyparams import SEED, MODELS, MODEL_EVALUATION
from data_utils import read_dataset


def evaluate_metrics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = y_pred.astype(bool)

    return {
        "AUC-ROC": roc_auc_score(y, y_proba),
        "F1-score": f1_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "TP": sum(y_pred & y) / sum(y),
        "# of TP": sum(y_pred & y),
    }


def train_additional_models(project, train, test, metrics={}):
    models_path = Path(f"{MODELS}/{project}")
    models_path.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.iloc[:, train.columns != "target"].values)
    y_train = train["target"].values
    X_test = scaler.transform(test.iloc[:, test.columns != "target"].values)
    y_test = test["target"].values

    # Only train LightGBM and CatBoost
    for model, model_name in [
        (LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1), "LightGBM"),
        (CatBoostClassifier(n_estimators=100, random_state=SEED, verbose=False), "CatBoost"),
    ]:
        model.fit(X_train, y_train)
        model_path = models_path / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        tqdm.write(f"Working on {project} with {model_name}...")
        test_metrics = evaluate_metrics(model, X_test, y_test)
        metrics[model_name][project] = test_metrics

    return metrics


def main():
    projects = read_dataset()
    Path(MODEL_EVALUATION).mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "LightGBM": {},
        "CatBoost": {},
    }
    
    for project in tqdm(projects, desc="projects", leave=True, total=len(projects)):
        train, test = projects[project]
        metrics = train_additional_models(project, train, test, metrics)

    # Save metrics per model as csv
    for model_name in metrics:
        model_metrics = metrics[model_name]
        df = pd.DataFrame(model_metrics)
        df.to_csv(f"{MODEL_EVALUATION}/{model_name}.csv")


if __name__ == "__main__":
    main()