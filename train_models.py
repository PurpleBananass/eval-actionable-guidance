from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib


from tqdm import tqdm

from data_utils import read_dataset
from hyparams import SEED, MODELS, MODEL_EVALUATION

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


def train_single_project(project, train, test, metrics={}):
    models_path = Path(f"{MODELS}/{project}") 
    models_path.mkdir(parents=True, exist_ok=True)

    # Apply StandardScaler to both train and test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.iloc[:, train.columns != 'target'].values)
    y_train = train['target'].values
    X_test = scaler.transform(test.iloc[:, test.columns != 'target'].values)
    y_test = test['target'].values

    for model, model_name in [
        # (RandomForestClassifier(n_estimators=100, random_state=SEED), "RandomForest"),
        # (XGBClassifier(n_estimators=100, random_state=SEED), "XGBoost"),
        (SVC(probability=True, random_state=SEED), "SVM"),
        # (LogisticRegression(random_state=SEED), "LogisticRegression"),
    ]:
        model.fit(X_train, y_train)
        if model_name == "XGBoost":
            model_path = models_path / f"{model_name}.xgb"
            model.save_model(model)
        else:
            model_path = models_path / f"{model_name}.joblib"
            joblib.dump(model, model_path)        

        tqdm.write(f"Working on {project} with {model_name}...")
        test_metrics = evaluate_metrics(model, X_test, y_test)
        metrics[model_name][project] = test_metrics

    return metrics


def train_all_project():
    projects = read_dataset()
    Path(MODEL_EVALUATION).mkdir(parents=True, exist_ok=True)
    metrics = {
        "RandomForest": {},
        "XGBoost": {},
        "SVM": {},
        "LogisticRegression": {},
    }
    for project in tqdm(projects, desc="projects", leave=True, total=len(projects)):
        train, test = projects[project]
        metrics = train_single_project(project, train, test, metrics)

    # Save metrics per model as csv
    for model_name in metrics:
        model_metrics = metrics[model_name]
        df = pd.DataFrame(model_metrics)
        df.to_csv(f"{MODEL_EVALUATION}/{model_name}.csv")
        
if __name__ == "__main__":
    train_all_project()