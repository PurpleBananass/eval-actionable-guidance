from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import gc
from TimeLIME import translate1, flip_interval


DEoptim_pkg = importr('DEoptim')
DEoptim_fn = DEoptim_pkg.DEoptim
DEoptim_control = DEoptim_pkg.DEoptim_control

SEED = 1


def LIME_HPO(X_train, test_instance, training_labels, model, path):
    "hyper parameter optimized lime explainer"
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        training_labels=training_labels,
        feature_names=X_train.columns,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    def objective(params):
        num_samples = int(params[0])
        explanation = explainer.explain_instance(
            test_instance, model.predict_proba, num_samples=num_samples
        )
        local_model_predictions = explanation.local_pred

        model_predictions = model.predict_proba(test_instance.reshape(1, -1))[0]

        residuals = model_predictions - local_model_predictions

        SS_res = np.sum(residuals**2)
        SS_tot = np.sum((model_predictions - np.mean(model_predictions)) ** 2)

        R2 = 1 - (SS_res / SS_tot)
        return -R2

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
        test_instance, model.predict_proba, num_samples=num_samples, num_features=len(X_train.columns)
    )
    
    top_features_rule = explanation.as_list()[:5]
    top_features = explanation.as_map()[1]
    top_features_index = [feature[0] for feature in top_features]
    top_feature_names = X_train.columns[top_features_index]
    top_feature_dtypes = X_train.dtypes[top_features_index]

    min_val = X_train.min()
    max_val = X_train.max()

    rules = []
    for i in range(len(top_features_rule)):
        original_l, original_r = translate1(top_features_rule[i][0], top_feature_names[i], top_feature_dtypes[i])
        l, r = flip_interval(original_l, original_r, min_val, max_val)
        rules.append([top_feature_names[i],test_instance[top_feature_names[i]], top_features_rule[i][1], l, r])

    rules_df = pd.DataFrame(rules, columns=['feature', 'value', 'importance', 'left', 'right'])
    rules_df.to_csv(path, index=False)




def LIME_HPO_R(X_train, test_instance, training_labels, model):
    "hyper parameter optimized lime explainer using DEoptim in R"
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        training_labels=training_labels,
        feature_names=X_train.columns,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    def objective_r(params):
        num_samples = int(params[0])
        explanation = explainer.explain_instance(
            test_instance, model.predict_proba, num_samples=num_samples
        )
        local_model_predictions = explanation.local_pred
        model_predictions = model.predict_proba(test_instance.reshape(1, -1))[0]
        residuals = model_predictions - local_model_predictions
        SS_res = np.sum(residuals**2)
        SS_tot = np.sum((model_predictions - np.mean(model_predictions)) ** 2)
        R2 = 1 - (SS_res / SS_tot)
        return ro.FloatVector([-R2])  # Note: R expects a vector, not a scalar

    # Define the R function for DEoptim
    gc.disable()

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        ro.r.assign("objective_r", objective_r)
        ro.r('''
        deoptim_func <- function(params) {
            return(objective_r(as.integer(params)))
        }
        ''')

        lower = ro.IntVector([100])
        upper = ro.IntVector([10000])

        control = DEoptim_control()
        control["NP"] = 10
        control["CR"] = 0.5
        control["F"] = 0.8
        control["itermax"] = 10

        
        
        result_r = DEoptim_fn(fn=ro.r.deoptim_func, lower=lower, upper=upper, control=control)

        num_samples_optimized = int(result_r.rx2("optim")[0])

        explanation = explainer.explain_instance(
            test_instance, model.predict_proba, num_samples=num_samples_optimized
        )
    gc.enable()

    return explanation, num_samples_optimized
