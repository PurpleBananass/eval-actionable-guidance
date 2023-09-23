## Studies Metrics
| Category  | Metrics                                                                                                                                                                                                                                                                                                                                         | Count |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| File      | AvgCyclomaticModified, AvgCyclomaticStrict, AvgEssential, AvgLineBlank, AvgLineComment, CountDeclClass, CountDeclClassMethod, CountDeclClassVariable, CountDeclInstanceMethod, CountDeclInstanceVariable, CountDeclMethodDefault, CountDeclMethodPrivate, CountDeclMethodProtected, CountDeclMethodPublic, CountLineComment, RatioCommentToCode | 16    |
| Method    | CountInput_Mean, CountInput_Min, CountOutput_Mean, CountOutput_Min, CountPath_Min, MaxNesting_Mean, MaxNesting_Min                                                                                                                                                                                                                              | 7     |
| Ownership | MAJOR_COMMIT, MAJOR_LINE, MINOR_COMMIT, MINOR_LINE, OWN_COMMIT, OWN_LINE                                                                                                                                                                                                                                                                        | 6     |
| Class     | CountClassBase, CountClassCoupled, CountClassDerived, MaxInheritanceTree, PercentLackOfCohesion                                                                                                                                                                                                                                                 | 5     |
| Process   | ADEV, Added_lines, Del_lines                                                                                                                                                                                                                                                                                                                    | 3     |
## Example of Proposed Changes by Explainers 
>java/engine/org/apache/derby/iapi/sql/execute/NoPutResultSet.java (derby 10.5.1.1)

| Explainer                 | Metric                  | Current Value | Mean Change | Proposed Change               |
|----------------------|-------------------------|---------------|-------------|------------------------------|
| TimeLIME             | OWN_LINE                | 0.780488      | 0.04        | 0.58, 0.53, ... 0.48 |
| TimeLIME             | MAJOR_LINE              | 1             | 1.32        | 2, 3, ... 10                 |
| TimeLIME             | CountDeclClass          | 1             | 1.36        | 2, 3, ... 13                 |
| TimeLIME             | CountClassCoupled       | 4             | 2.26        | 7, 8, ... 49                 |
| SQAPlanner_confidence | RatioCommentToCode      | 4.9           | 0.12        | 4.9, 4.85, ... 7.3 |
| SQAPlanner_confidence | CountLineComment        | 147           | 13.83       | 61, 60, ... 50               |
| LIMEHPO              | Added_lines             | 5             | 57.73       | 7, 8, ... 1159              |
| LIMEHPO              | CountDeclClassVariable  | 6             | 2.75        | 3, 2, ... 0                 |
| LIMEHPO              | CountLineComment        | 147           | 13.83       | 146, 145, ... 15             |
| LIMEHPO              | PercentLackOfCohesion   | 100           | 10.86       | 99, 98, ... 0               |
| LIMEHPO              | MaxNesting_Min          | 0             | 1.0         | 1, 2, ... 5                 |
| DeFlip               | MaxInheritanceTree      | 2             | 1.11        | 1.0                        |
| DeFlip               | MaxNesting_Mean         | 0.0           | 0.17        | 1.2                        |
