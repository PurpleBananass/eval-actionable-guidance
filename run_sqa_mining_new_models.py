# run_sqa_mining_new_models.py
import subprocess
from itertools import product

model_types = ["LightGBM", "CatBoost"]
model_types = ["CatBoost"]
search_strategies = ["confidence", "coverage", "lift"]
search_strategies = ["confidence"]

for model, strategy in product(model_types, search_strategies):
    print(f"\n{'='*50}")
    print(f"Mining rules for {model} with {strategy} strategy")
    print(f"{'='*50}\n")
    
    subprocess.run([
        "python", "mining_sqa_rules.py",
        "--model_type", model,
        "--search_strategy", strategy
    ])