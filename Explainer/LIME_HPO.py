from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler
from hyparams import SEED

def LIME_HPO(X_train, test_instance, training_labels, model, path):
    "hyper parameter optimized lime explainer"
    
    # Apply StandardScaler
    scaler = StandardScaler()
    # Ensure DataFrame format to preserve column names
    X_train_scaled = scaler.fit_transform(X_train.values)
    test_instance_scaled = scaler.transform(test_instance.values.reshape(1, -1))


    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=training_labels,
        feature_names=X_train.columns,  # Use original column names
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    def objective(params):
        num_samples = int(params[0])
        explanation = explainer.explain_instance(
            test_instance_scaled[0], model.predict_proba, num_samples=num_samples
        )
        local_model_predictions = explanation.local_pred

        # 모델 예측값 (넘파이 배열로 변환)
        model_predictions = model.predict_proba(test_instance_scaled)[0]

        residuals = model_predictions - local_model_predictions

        # 잔차 계산
        SS_res = np.sum(residuals ** 2)
        SS_tot = np.sum((model_predictions - np.mean(model_predictions)) ** 2)

        # SS_tot == 0인 경우 처리
        if SS_tot == 0:
            print(f"SS_tot is 0 for this instance, likely due to no variance in model predictions.")
            # 페널티를 부여하거나 무시할 수 있음
            return 100  # 큰 페널티 값 반환 (탐색을 이 영역에서 멀어지게 함)

        # 정상적인 경우 R^2 계산
        R2 = 1 - (SS_res / SS_tot)
        return -R2  # 최적화에서는 R^2을 최소화하려고 음수로 반환

    bounds = [(100, 10000)]
    result = differential_evolution(
        objective,
        bounds,
        strategy="currenttobest1bin",
        maxiter=10,
        popsize=10,
        mutation=0.8,
        recombination=0.5,
        seed=SEED,
    )
    num_samples = int(result.x[0])

    explanation = explainer.explain_instance(
        test_instance_scaled[0], model.predict_proba, num_samples=num_samples, num_features=len(X_train.columns)
    )
    
    top_features_rule = explanation.as_list()[:5]
    top_features = explanation.as_map()[1]
    top_features_index = [feature[0] for feature in top_features][:5]
    top_feature_names = X_train.columns[top_features_index]

    min_val = X_train.min()
    max_val = X_train.max()

    rules, importances = zip(*top_features_rule)
    
    total_rules, total_importances = zip(*explanation.as_list())
    total_importances = np.array(total_importances)
    abs_importances = np.abs(total_importances)
    importance_ratio = np.abs(np.array(importances)) / np.sum(abs_importances)

    rules_df = pd.DataFrame({
        'feature': top_feature_names,
        'value': test_instance_scaled[0][top_features_index], 
        'importance': importances,
        'min': min_val[top_features_index],
        'max': max_val[top_features_index],
        'rule': rules,
        'importance_ratio': importance_ratio
    })
    
    rules_df.to_csv(path, index=False)


def LIME_Planner(X_train, test_instance, training_labels, model, path):
    # Apply StandardScaler
    scaler = StandardScaler()
    # Ensure DataFrame format to preserve column names
    X_train_scaled = scaler.fit_transform(X_train.values)
    test_instance_scaled = scaler.transform(test_instance.values.reshape(1, -1))

    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=training_labels,
        feature_names=X_train.columns,  # Use original column names
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    explanation = explainer.explain_instance(
        test_instance_scaled[0], model.predict_proba, num_features=len(X_train.columns)
    )
    
    top_features_rule = explanation.as_list()
    top_features = explanation.as_map()[1]
    top_features_index = [feature[0] for feature in top_features]
    top_feature_names = X_train.columns[top_features_index]

    min_val = X_train.min()
    max_val = X_train.max()

    rules, importances = zip(*top_features_rule)

    rules_df = pd.DataFrame({
        'feature': top_feature_names,
        'value': test_instance_scaled[0][top_features_index],
        'importance': importances,
        'min': min_val[top_features_index],
        'max': max_val[top_features_index],
        'rule': rules
    })
    
    rules_df.to_csv(path, index=False)