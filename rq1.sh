#!/usr/bin/env fish

poetry run python flip_exp.py --only_flip_rate --explainer_type TimeLIME
poetry run python flip_exp.py --only_flip_rate --explainer_type TimeLIME --only_minimum

poetry run python flip_exp.py --only_flip_rate --explainer_type LIMEHPO
poetry run python flip_exp.py --only_flip_rate --explainer_type LIMEHPO --only_minimum

poetry run python flip_exp.py --only_flip_rate --explainer_type SQAPlanner --search_strategy confidence
poetry run python flip_exp.py --only_flip_rate --explainer_type SQAPlanner --search_strategy confidence --only_minimum

poetry run python flip_exp.py --only_flip_rate --explainer_type SQAPlanner --search_strategy coverage
poetry run python flip_exp.py --only_flip_rate --explainer_type SQAPlanner --search_strategy coverage --only_minimum

poetry run python flip_exp.py --only_flip_rate --explainer_type SQAPlanner --search_strategy lift
poetry run python flip_exp.py --only_flip_rate --explainer_type SQAPlanner --search_strategy lift --only_minimum