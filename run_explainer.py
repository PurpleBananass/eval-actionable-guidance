
from pathlib import Path
from data_utils import read_dataset
from hyparams import *

def run_single_project(project, train, test, explainer_type):
    models_path = Path(f"{MODELS}/{project}") 
    output_path = Path(f"{OUTPUT}{project}/{explainer_type}")
    models_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)



def run_all_project():
    projects = read_dataset()
    for project in projects:
        train, test = projects[project]
        run_single_project(project, train, test)

        