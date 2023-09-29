# DeFlip Replication Package

This is the replication package for the paper "Evaluating the Effectiveness of Actionable Guidance for Defect Prediction from Developers' Perspective"

## ToC


## DeFlip

DeFlip is a tool that provides actionable guidance for defect prediction. It is implemented based on [DiCE](https://github.com/interpretml/DiCE/tree/main).

### Usage



## Installation
We recommned to use poetry to install the dependencies. To install poetry, please follow the instructions [here](https://python-poetry.org/docs/#installation).

After installing poetry, you can install the dependencies by running the following command:
```bash
poetry install
```

If you want to preprocess the dataset, you need to install the following dependencies:
```bash
poetry install -E preprocess
```

If you want to test with other models, you need to install the following dependencies:
```bash
poetry install -E all_models
```

## Replication Researh Questions

### Preprocess
#### Prerequisites
```
R 
AutoSpearman
rpy2
```

```bash
python preprocess.py
```
Our preprocessing steps include:
- Generate (k-1 , k) release (train, test) dataset from original dataset (Yatish et al.)
- Feature selection using AutoSpearman

or you can just use the preprocessed dataset in `Dataset/release_dataset`

### Run Explainers to generate proposed changes
for LIME-HPO and TimeLIME: 2 steps
```bash
python run_explainers.py --explainer "LIMEHPO" --project "activemq@2"
python plan_explanations.py --explainer "LIMEHPO" --project "activemq@2" # --only_minimum for MPC
```
The first step generates the explanation-based proposed changes, and the second step generates the all possible changes or the minimum changes.

for SQAPlanner: 3 steps
```bash
python run_explainers.py --explainer "SQAPlanner" --project "activemq@2"
cd Explainers/SQAPlanner
python mining_sqa_rules.py --project "activemq@2" --search_strategy confidence # Requires BigML API key
python plan_explanations.py --explainer "SQAPlanner" --search_strategy confidence --project "activemq@2" # --only_minimum for MPC
```
The first step generates random neighbor instances using training data, and the second step generates the association rules using BigML API. The third step generates the explanation-based proposed changes.

### Flip checking
```bash
python flip_exp.py --explainer "TimeLIME" # RQ1 (a) Flip check with possible flipping proposed changes 
python flip_exp.py --explainer "TimeLIME" --only_minimum # RQ1 (b) Flip check with minimum proposed changes
```
The flipped instances are stored in `flipped_instances` folder and flip rates are stored in `flip_rates` folder.

### Research Questions
```bash
python research_questions.py 
```

## Appendix


