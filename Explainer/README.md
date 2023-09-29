# Explainer

We have reimplemented/modified each approach based on their implementations to fit our dataset and experimental setup.

## LIME-HPO
* original paper: [An Empirical Study of Model-Agnostic Techniques for Defect Prediction Models](https://ieeexplore.ieee.org/document/9044387/)

The original paper implemented it in R, but as there was no publicly released implementation, we implemented it in Python.


## TimeLIME

* original paper: [Defect Reduction Planning (Using TimeLIME)](https://ieeexplore.ieee.org/document/9371412/)

It was originally implemented based on k-1, k, k+1 and had some parts conflicting with the content of the paper, so we modified it to fit our experiment.

## SQAPlanner

* original paper: [SQAPlanner: Generating Data-Informed Software Quality Improvement Plans](10.1109/TSE.2021.3070559)

The original paper mentions the use of BigML, but there was no code related to automation, so we performed automation based on the BigML API in `mining_sqa_rules.py`.
