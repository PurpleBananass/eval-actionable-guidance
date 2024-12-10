from pathlib import Path
from bigml.api import BigML
from bigml.association import Association
import os


def get_or_create_dataset(api, file_name, project):
    dataset_name = Path(file_name).stem

    datasets = api.list_datasets()
    api.ok(datasets, wait_time=10)
    found_datasets = [
        d
        for d in datasets["objects"]
        if d["name"] == dataset_name and project in d["tags"]
    ]

    if found_datasets:
        return found_datasets[0]["resource"]

    else:
        source = api.create_source(file_name, {"name": dataset_name, "tags": [project]})
        api.ok(source, wait_time=10)
        dataset = api.create_dataset(source, {"name": dataset_name, "tags": [project]})
        api.ok(dataset, wait_time=10)
        return dataset["resource"]


def get_or_create_association(api, dataset_id, options, file_path):
    associations = api.list_associations()
    api.ok(associations, wait_time=10)
    found_associations = [
        a
        for a in associations["objects"]
        if a["name"] == options["name"] and a["tags"] == options["tags"]
    ]

    if found_associations:
        association_id = found_associations[0]["resource"]
    else:
        association = api.create_association(dataset_id, options)
        api.ok(association, wait_time=10)
        association_id = association["resource"]

    local_association = Association(association_id)

    local_association.rules_csv(file_path)


def delete_all_resources(list_function, delete_function):
    next_offset = 0
    max_batch_size = 100

    while True:
        query_string = "offset={}&limit={}".format(next_offset, max_batch_size)
        resources = list_function(query_string=query_string)["objects"]

        if len(resources) == 0:
            print("No more resources to delete")
            break

        for resource in resources:
            delete_function(resource["resource"])

        next_offset += max_batch_size  # 다음 페이지로 이동


if __name__ == "__main__":
    username = os.environ["BIGML_USERNAME"]
    api_key = os.environ["BIGML_API_KEY"]
    api = BigML(username, api_key)

    prompt = input("DELETE ALL BIGML RESOURCES? (y/n): ")
    if prompt == "y":
        # 모든 소스 삭제
        delete_all_resources(api.list_sources, api.delete_source)

        # 모든 데이터셋 삭제
        delete_all_resources(api.list_datasets, api.delete_dataset)

        # 모든 연관성 삭제
        delete_all_resources(api.list_associations, api.delete_association)
