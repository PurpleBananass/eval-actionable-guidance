#!/bin/bash

# Start of the script
echo "Starting the script..."

# Running flip_exp with LIMEHPO
echo "Running flip_exp with LIMEHPO (only minimum)..."
poetry run python flip_exp.py --explainer_type LIMEHPO --only_minimum

# Running flip_exp with TimeLIME
echo "Running flip_exp with TimeLIME (only minimum)..."
poetry run python flip_exp.py --explainer_type TimeLIME --only_minimum

# Running flip_exp with SQAPlanner and search strategy: confidence
echo "Running flip_exp with SQAPlanner using confidence as search strategy (only minimum)..."
poetry run python flip_exp.py --explainer_type SQAPlanner --only_minimum --search_strategy confidence

# Running flip_exp with SQAPlanner and search strategy: coverage
echo "Running flip_exp with SQAPlanner using coverage as search strategy (only minimum)..."
poetry run python flip_exp.py --explainer_type SQAPlanner --only_minimum --search_strategy coverage

# Running flip_exp with SQAPlanner and search strategy: lift
echo "Running flip_exp with SQAPlanner using lift as search strategy (only minimum)..."
poetry run python flip_exp.py --explainer_type SQAPlanner --only_minimum --search_strategy lift

# Running deflip
echo "Running deflip..."
poetry run python deflip.py

# End of the script
echo "Script execution completed."
