#!/bin/bash

rye run python flip_exp.py --explainer LIME-HPO --model XGBoost --verbose --project $1