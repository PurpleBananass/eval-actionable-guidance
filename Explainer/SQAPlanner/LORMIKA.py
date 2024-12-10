import os
import numpy as np
import math
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class LORMIKA:
    def __init__(self, **kwargs):
        self.__train_set: pd.DataFrame = kwargs["train_set"]
        self.__train_class: pd.DataFrame = kwargs["train_class"]
        self.__cases: pd.DataFrame = kwargs["cases"]
        self.__model = kwargs["model"]
        self.__output_path = kwargs["output_path"]
        Path(self.__output_path).mkdir(parents=True, exist_ok=True)

    def instance_generation(self):
        target_train = self.__model.predict(self.__train_set.values)

        # Does feature scaling for continuous data
        scaler = StandardScaler()
        trainset_normalize = self.__train_set.copy()
        cases_normalize = self.__cases.copy()

        trainset_normalize = pd.DataFrame(
            scaler.fit_transform(trainset_normalize), index=trainset_normalize.index
        )
        cases_normalize = pd.DataFrame(
            scaler.transform(cases_normalize), index=cases_normalize.index
        )

        assert self.__train_set.index.equals(trainset_normalize.index)
        assert self.__cases.index.equals(cases_normalize.index)

        # make dataframe to store similarities of the trained instances from the explained instance
        dist_df = pd.DataFrame(index=trainset_normalize.index.copy())

        width = math.sqrt(len(self.__train_set.columns)) * 0.75

        for sample_number, test_instance in tqdm(
            cases_normalize.iterrows(),
            desc=f"{Path(self.__output_path).parent.name}",
            leave=False,
            total=len(cases_normalize),
        ):
            # We only consider the instances that are predicted as buggy
            pred = self.__model.predict(test_instance.values.reshape(1, -1))[0]
            if pred == 0:
                continue

            file_name = f"{self.__output_path}/{test_instance.name}.csv"

            # 파일이 이미 존재하는지 확인
            if os.path.exists(file_name):
                continue

            # Calculate the euclidian distance from the instance to be explained
            dist = np.linalg.norm(
                trainset_normalize.sub(np.array(test_instance)), axis=1
            )

            # Convert distance to a similarity score
            similarity = np.sqrt(np.exp(-(dist**2) / (width**2)))

            dist_df["dist"] = similarity
            dist_df["t_target"] = target_train

            # get the unique classes of the training set
            unique_classes = dist_df.t_target.unique()
            # Sort similarity scores in to decending order
            dist_df.sort_values(by=["dist"], ascending=False, inplace=True)

            # Make a dataframe with top 40 elements in each class
            top_fourty_df = pd.DataFrame([])
            for clz in unique_classes:
                top_fourty_df = pd.concat(
                    [top_fourty_df, dist_df[dist_df["t_target"] == clz].head(40)]
                )

            # get the minimum value of the top 40 elements and return the index
            cutoff_similarity = top_fourty_df.nsmallest(
                1, "dist", keep="last"
            ).index.values.astype(int)[0]

            # Get the location for the given index with the minimum similarity
            min_loc = dist_df.index.get_loc(cutoff_similarity)
            if isinstance(min_loc, np.ndarray):
                min_loc = min_loc[0]
            # whole neighbourhood without undersampling the majority class
            train_neigh_sampling_b = dist_df.iloc[0 : min_loc + 1]
            # get the size of neighbourhood for each class
            target_details = train_neigh_sampling_b.groupby(["t_target"]).size()

            target_details_df = pd.DataFrame(
                {
                    "target": target_details.index,
                    "target_count": target_details.values,
                }
            )

            # Get the majority class and undersample
            final_neighbours_similarity_df = pd.DataFrame([])
            for index, row in target_details_df.iterrows():
                if row["target_count"] > 200:
                    filterd_class_set = train_neigh_sampling_b.loc[
                        train_neigh_sampling_b["t_target"] == row["target"]
                    ].sample(n=200)
                    final_neighbours_similarity_df = pd.concat(
                        [final_neighbours_similarity_df, filterd_class_set]
                    )
                else:
                    filterd_class_set = train_neigh_sampling_b.loc[
                        train_neigh_sampling_b["t_target"] == row["target"]
                    ]
                    final_neighbours_similarity_df = pd.concat(
                        [final_neighbours_similarity_df, filterd_class_set]
                    )

            # Get the original training set instances which is equal to the index of the selected neighbours
            train_set_neigh = self.__train_set[
                self.__train_set.index.isin(final_neighbours_similarity_df.index)
            ]

            train_class_neigh = self.__train_class[
                self.__train_class.index.isin(final_neighbours_similarity_df.index)
            ]

            new_con_df = pd.DataFrame([])

            sample_classes_arr = []
            sample_indexes_list = []
            #######Generating 1000 instances using interpolation technique
            for num in range(0, 1000):
                rand_rows = train_set_neigh.sample(2, replace=True)
                sample_indexes_list = (
                    sample_indexes_list + rand_rows.index.values.tolist()
                )
                sample_classes = train_class_neigh[
                    train_class_neigh.index.isin(rand_rows.index)
                ]
                sample_classes = np.array(
                    sample_classes.to_records().view(type=np.matrix)
                )
                sample_classes_arr.append(sample_classes[0].tolist())

                alpha_n = np.random.uniform(low=0, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                new_ins = x + (y - x) * alpha_n
                new_ins = new_ins.to_frame().T

                new_ins.name = num
                new_con_df = pd.concat([new_con_df, new_ins], axis=0)

            #######Generating 1000 instances using cross-over technique
            for num in range(1000, 2000):
                rand_rows = train_set_neigh.sample(3, replace=True)
                sample_indexes_list = (
                    sample_indexes_list + rand_rows.index.values.tolist()
                )
                sample_classes = train_class_neigh[
                    train_class_neigh.index.isin(rand_rows.index)
                ]
                sample_classes = np.array(
                    sample_classes.to_records().view(type=np.matrix)
                )
                sample_classes_arr.append(sample_classes[0].tolist())

                mu_f = np.random.uniform(low=0.5, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                z = rand_rows.iloc[2]
                new_ins = x + (y - z) * mu_f
                new_ins = new_ins.to_frame().T

                new_ins.name = num
                new_con_df = pd.concat([new_con_df, new_ins], axis=0, ignore_index=True)

            # get the global model predictions of the generated instances and the instances in the neighbourhood
            predict_dataset = pd.concat(
                [train_set_neigh, new_con_df], ignore_index=True
            )

            # normalize predict_dataset
            predict_dataset_norm = pd.DataFrame(
                scaler.transform(predict_dataset), index=predict_dataset.index
            )

            target = self.__model.predict(predict_dataset_norm.values)
            target_df = pd.DataFrame(target)

            new_df_case = pd.concat([predict_dataset, target_df], axis=1)
            new_df_case = np.round(new_df_case, 2)
            new_df_case.rename(columns={0: self.__train_class.columns[0]}, inplace=True)
            new_df_case["target"] = new_df_case["target"].astype(int)
            new_df_case["target"] = new_df_case["target"].astype(str)
            new_df_case.to_csv(file_name, index=False)
