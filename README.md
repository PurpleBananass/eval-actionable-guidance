# eval-actionable-guidance

This is the replication package for the paper "Why Should Practitioners Trust the Actionable Guidance from Explainable AI Techniques for Software Defect Prediction?"

## Abstract

Despite advances in high-performance Software Defect Prediction (SDP) models, practitioners hesitate to adopt them due to decision-making processes that are difficult to understand and a lack of actionable insights. Recent research has applied various explainable AI (XAI) techniques to provide explainable and actionable guidance for SDP results to address these limitations, but whether such guidance can be trusted by practitioners has not been sufficiently investigated. Practitioners may question the guidance's feasibility before implementation, and if modifications prove inaccurate or fail to resolve predicted defects after implementation, trust in the guidance may further diminish. In this study, We empirically evaluate the effectiveness of current XAI approaches for SDP across 32 releases of 9 large-scale projects, focusing on whether the guidance meets practitioners' expectations by leading to changes in prediction results when implemented. Our findings show that their actionable guidance (i) does not guarantee predicted defects to be resolved; (ii) fails to pinpoint modification required to resolve predicted defects; and (iii) diverges from common code changes by practitioners in their projects. These limitations indicate that the guidance is still not reliable enough for developers to invest their limited debugging resources. We suggest that future XAI research for SDP incorporate a feedback loop delivering clear rewards for practitioners' actions, and propose a potential alternative approach utilizing counterfactual explanations.


## Requirements

We tested the code on the following environment:
- os: MacOS
- python 3.11.8

```txt
"scikit-learn>=1.5.2",
"lime>=0.2.0.1",
"seaborn>=0.13.2",
"mlxtend>=0.23.1",
"scipy>=1.14.1",
"natsort>=8.4.0",
"dice-ml>=0.11",
"xgboost>=2.1.1",
"bigml>=9.7.1",
"ipykernel>=6.29.5",
"python-dotenv>=1.0.1",
"tabulate>=0.9.0",
"cliffs-delta>=1.0.0",
"rpy2>=3.5.16",
```

### Preprocessing

If you want to preprocess the dataset, you need to install the following dependencies:

```txt
R # for AutoSpearman
AutoSpearman # for feature selection
rpy2 
```
Our preprocessing steps include:
- Generate (k-1 , k) release (train, test) dataset from original dataset (Yatish et al.)
- Feature selection using AutoSpearman

or you can just use the preprocessed dataset in `Dataset/release_dataset`

## [Dataset](./Dataset/README.md)

## [Baselines](./Explainer/README.md)

## Training
```bash
python train_models.py
```
This script trains the models for each project and stores them in the `models` folder.

## Run Explainers to generate proposed changes
for LIME-HPO and TimeLIME: 2 steps
```bash
python run_explainers.py --explainer "LIMEHPO" --model "RandomForest" --project "activemq@2"
python plan_explanations.py --explainer "LIMEHPO" --model "RandomForest" --project "activemq@2"
```
The first step generates the explanation-based proposed changes, and the second step generates the all possible changes or the minimum changes.

for SQAPlanner: 3 steps
```bash
python run_explainers.py --explainer "SQAPlanner" --model "RandomForest" --project "activemq@2"
python mining_sqa_rules.py --project "activemq@2" --search_strategy confidence --model "RandomForest" 
python plan_explanations.py --explainer "SQAPlanner" --search_strategy confidence --model "RandomForest" --project "activemq@2"
```
The first step generates random neighbor instances using training data, and the second step generates the association rules using BigML API. The third step generates the explanation-based proposed changes. `.env` file is required to use BigML API key.

```.env
BIGML_USERNAME=YOUR_USERNAME
BIGML_API_KEY=YOUR_API_KEY
```

## Simulation and Evaluation

```bash
python flip_exp.py --explainer "TimeLIME" --model "RandomForest" 
```


## Research Questions Replication

- RQ1: Can the implementation of actionable guidance for SDP guarantee changes in predictions?
- RQ2: Does the guidance accurately suggest modifications that lead to prediction changes?
- RQ3: Can the techniques suggest changes that align with these typically made by practitioners in their projects?

```bash
python evaluate.py --rq1 --rq2 --rq3 --implications
python analysis.py --rq1 --rq2 --rq3 --implications # for statistical analysis and plots
```

## Appendix: A summary of the studied software metrics

| Category         | Metrics Level | Metrics                                                                                                                                                              | Count |
|------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| Code metrics     | File-level    | AvgCyclomaticModified, AvgCyclomaticStrict, AvgEssential, AvgLineBlank, AvgLineComment, CountDeclClass, CountDeclClassMethod, CountDeclClassVariable, CountDeclInstanceMethod, CountDeclInstanceVariable, CountDeclMethodDefault, CountDeclMethodPrivate, CountDeclMethodProtected, CountDeclMethodPublic, CountLineComment, RatioCommentToCode | 16    |
|                  | Class-level   | CountClassBase, CountClassCoupled, CountClassDerived, MaxInheritanceTree, PercentLackOfCohesion                                                                                                  | 5     |
|                  | Method-level  | CountInput_Mean, CountInput_Min, CountOutput_Mean, CountOutput_Min, CountPath_Min, MaxNesting_Mean, MaxNesting_Min                                                                             | 7     |
| Process metrics  |               | ADEV, ADDED_LINES, DEL_LINES                                                                                                                                                                      | 3     |
| Ownership metrics|               | MAJOR_COMMIT, MAJOR_LINE, MINOR_COMMIT, MINOR_LINE, OWN_COMMIT, OWN_LINE                                                                                                                          | 6     |