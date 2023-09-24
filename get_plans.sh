#!/bin/bash

# Start of the script
echo "Starting the script..."

# # Generate plans with LIMEHPO
echo "Generating plans with LIMEHPO..."
poetry run python plan_explanations.py --explainer_type LIMEHPO
echo "Generating plans with LIMEHPO (only minimum)..."
poetry run python plan_explanations.py --explainer_type LIMEHPO --only_minimum

# Generate plans with TimeLIME
echo "Generating plans with TimeLIME..."
poetry run python plan_explanations.py --explainer_type TimeLIME
echo "Generating plans with TimeLIME (only minimum)..."
poetry run python plan_explanations.py --explainer_type TimeLIME --only_minimum

# Generate plans with SQAPlanner and search strategy: confidence
echo "Generating plans with SQAPlanner using confidence as search strategy..."
poetry run python plan_explanations.py --explainer_type SQAPlanner  --search_strategy confidence
echo "Generating plans with SQAPlanner using confidence as search strategy (only minimum)..."
poetry run python plan_explanations.py --explainer_type SQAPlanner --only_minimum --search_strategy confidence

# Generate plans with SQAPlanner and search strategy: coverage
echo "Generating plans with SQAPlanner using coverage as search strategy..."
poetry run python plan_explanations.py --explainer_type SQAPlanner  --search_strategy coverage
echo "Generating plans with SQAPlanner using coverage as search strategy (only minimum)..."
poetry run python plan_explanations.py --explainer_type SQAPlanner --only_minimum --search_strategy coverage

# Generate plans with SQAPlanner and search strategy: lift
echo "Generating plans with SQAPlanner using lift as search strategy..."
poetry run python plan_explanations.py --explainer_type SQAPlanner  --search_strategy lift
echo "Generating plans with SQAPlanner using lift as search strategy (only minimum)..."
poetry run python plan_explanations.py --explainer_type SQAPlanner --only_minimum --search_strategy lift

# End of the script
echo "Script execution completed."
