# Dataset

The original JIRA defect dataset by Yatish et al. can be found [here](https://github.com/awsm-research/Rnalytica).

We also provide the original dataset in `Dataset/original_dataset` for convenience.

The only difference between the original data and the original data included in this repository is a slight modification of the file (release name) for sorting by release.

## A statistical summary of the dataset

| Name    | Description                           | #Files        | #Defective files | Studied Releases            |
|---------|---------------------------------------|---------------|------------------|----------------------------|
| ActiveMQ| Messaging and Integration Patterns sever| 1,759-2,822 | 154-292        | 5.0.0, 5.1.0, 5.2.0, 5.3.0, 5.8.0|
| Camel   | Enterprise Integration Framework      | 1,492-8,243  | 189-285        | 1.4.0, 2.9.0, 2.10.0, 2.11.0|
| Derby   | Relational Database                   | 1,953-2,681  | 383-667        | 10.2.1.6, 10.3.1.4, 10.5.1.1|
| Groovy  | Java-syntax-compatible OOP for JAVA   | 749-868      | 26-73          | 1.5.7, 1.6.0_Beta1, 1.6.0_Beta2|
| HBase   | Distributed Scalable Data Store       | 1,059-1,793  | 218-463        | 0.94.0, 0.95.0, 0.95.2     |
| Hive    | Data Warehouse System for Hadoop      | 1,326-2,537  | 176-278        | 0.9.0, 0.10.0, 0.12.0      |
| JRuby   | Ruby Programming Lang for JVM         | 699-1,576    | 82-180         | 1.1, 1.4, 1.5, 1.7         |
| Lucene  | Text Search Engine Library            | 799-2,528    | 84-273         | 2.3.0, 2.9.0, 3.0.0, 3.1.0 |
| Wicket  | Web Application Framework             | 1,613-2,508  | 101-130        | 1.3.0.beta1, 1.3.0.beta2, 1.5.3|

## Dataset Structure

The dataset is structured as follows:

```
Dataset
├── original_dataset
├── project_dataset: just temporary files
├── release_dataset
    ├── activemq@k
        ├── train.csv, test.csv, mapping.csv (for k-th release)
├── historical_dataset
    ├── activemq@k.csv (each feature's historical mean changes for k-th release)
```