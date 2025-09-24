# run_models_explainers.py
import subprocess
from itertools import product

model_types = ["LightGBM", "CatBoost"]  
explainer_types = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]

for model, explainer in product(model_types, explainer_types):
    print(f"\n{'='*50}")
    print(f"Running {explainer} on {model}")
    print(f"{'='*50}\n")
    
    subprocess.run([
        "python", "run_explainer.py",
        "--model_type", model,
        "--explainer_type", explainer,
        "--project", "all"
    ])