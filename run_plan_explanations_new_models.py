# run_plan_explanations_new_models.py
import subprocess
from itertools import product

model_types = ["LightGBM", "CatBoost"]
model_types = ["CatBoost"]
explainer_types = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]
sqa_strategies = ["confidence", "coverage", "lift"]

# Run for LIME variants and TimeLIME
for model, explainer in product(model_types, explainer_types):
    print(f"\nGenerating plans for {model} with {explainer}")
    subprocess.run([
        "python", "plan_explanations.py",
        "--model_type", model,
        "--explainer_type", explainer
    ])

# Run for SQAPlanner with different strategies
# for model, strategy in product(model_types, sqa_strategies):
#     print(f"\nGenerating plans for {model} with SQAPlanner ({strategy})")
#     subprocess.run([
#         "python", "plan_explanations.py",
#         "--model_type", model,
#         "--explainer_type", "SQAPlanner",
#         "--search_strategy", strategy
#     ])