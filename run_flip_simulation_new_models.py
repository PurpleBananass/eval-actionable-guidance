# run_flip_simulation_new_models.py
import subprocess
from itertools import product

model_types = ["LightGBM", "CatBoost"]
model_types = ["CatBoost"]
explainer_types = ["LIME", "LIME-HPO", "TimeLIME"]
sqa_strategies = ["confidence", "coverage", "lift"]

# Run simulations for LIME variants and TimeLIME
for model, explainer in product(model_types, explainer_types):
    print(f"\nRunning flip simulation for {model} with {explainer}")
    subprocess.run([
        "python", "flip_exp.py",
        "--model_type", model,
        "--explainer_type", explainer
    ])

# # Run simulations for SQAPlanner with different strategies
# for model, strategy in product(model_types, sqa_strategies):
#     print(f"\nRunning flip simulation for {model} with SQAPlanner ({strategy})")
#     subprocess.run([
#         "python", "flip_simulation.py",
#         "--model_type", model,
#         "--explainer_type", "SQAPlanner",
#         "--search_strategy", strategy
#     ])

# Get flip rates after all simulations are done
print("\n" + "="*50)
print("FLIP RATES SUMMARY")
print("="*50)

for model in model_types:
    for explainer in explainer_types:
        subprocess.run([
            "python", "flip_exp.py",
            "--model_type", model,
            "--explainer_type", explainer,
            "--get_flip_rate",
            "--verbose"
        ])
    
    # for strategy in sqa_strategies:
    #     subprocess.run([
    #         "python", "flip_simulation.py",
    #         "--model_type", model,
    #         "--explainer_type", "SQAPlanner",
    #         "--search_strategy", strategy,
    #         "--get_flip_rate"
    #     ])