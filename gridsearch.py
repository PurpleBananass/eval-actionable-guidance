# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE
from collections import defaultdict

from data_utils import read_dataset

# Define function to train a single project
def train_single_project(X_train, y_train, X_test, y_test, sampling_strategy, class_weight):
    if sampling_strategy is not None:
        smote = SMOTE(sampling_strategy=sampling_strategy)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    clf = RandomForestClassifier(class_weight=class_weight, random_state=1)
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    return precision

# Define function to train all projects
def train_all_projects(data, smote_ratios, class_weights):
    best_params = defaultdict(lambda: defaultdict(dict))
    
    for project in data:
        best_precision = 0
        best_smote = None
        best_class_weight = None
        
        train, test = data[project]
        X_train = train.iloc[:, train.columns != "target"].values
        y_train = train["target"].values
        X_test = test.iloc[:, test.columns != "target"].values
        y_test = test["target"].values
        
        for smote_ratio in smote_ratios:
            for class_weight in class_weights:
                n_majority = len(y_train[y_train == 0])
                n_minority = len(y_train[y_train == 1])
                if smote_ratio is None:
                    precision = train_single_project(X_train, y_train, X_test, y_test, None, class_weight)
                else:
                    n_minority_target = n_majority // smote_ratio  # Integer division
                    if n_minority_target >= n_minority:
                        sampling_strategy = {0: n_majority, 1: n_minority_target} 
                    else:
                        continue
                    precision = train_single_project(X_train, y_train, X_test, y_test, sampling_strategy, class_weight)
                    
                if precision > best_precision:
                    best_precision = precision
                    best_smote = smote_ratio
                    best_class_weight = class_weight
        
        
        best_params[project]['smote_ratio'] = best_smote
        best_params[project]['class_weight'] = best_class_weight
        best_params[project]['best_precision'] = best_precision

        print(f"Best parameters for {project}: {best_params[project]}")

    return best_params

# Simulate data (Replace this part with your actual data loading mechanism)
# Each project has training and testing data: X_train, y_train, X_test, y_test
data = read_dataset()

# Possible SMOTE ratios and class_weights
smote_ratios = [10, 8, 5, 3, 2, 1, None]
class_weights = ['balanced', None, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 8}, {0: 1, 1: 10}]

# Find the best parameters
best_params = train_all_projects(data, smote_ratios, class_weights)
best_params
print(best_params)