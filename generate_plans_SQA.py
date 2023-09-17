#!/usr/bin/env python3
# coding: utf-8
import argparse
from pathlib import Path
import time
from bigml.api import BigML
import pandas as pd
import os
import re
from openpyxl import Workbook
from openpyxl import load_workbook
from pathlib import Path
from openpyxl.utils.dataframe import dataframe_to_rows
import pickle

from tqdm import tqdm
from bigml_mining import get_or_create_association, get_or_create_dataset

from data_utils import read_dataset
from hyparams import MODELS, OUTPUT, PLANS

def comparison(word):
	regexp =re.finditer(r'\w+=[0-9]+',word)
	regexp_minus_f = re.finditer(r'-[0-9]+.[0-9]+',word)
	regexp_minus = re.finditer(r'-[0-9]+(^.)',word)

	for m in regexp:
		matched_word=m.group()
		new_word="==".join(matched_word.split("="))
		word=word.replace(matched_word, new_word)
	return word

def generate_plans(project, mode):
	assert mode in ["coverage", "confidence", "lift"]
	username = os.environ['BIGML_USERNAME']
	api_key = os.environ['BIGML_API_KEY']
	api = BigML(username, api_key)

	projects = read_dataset()

	train, test = projects[project]
	model_path = Path(MODELS) / f"{project}/RandomForest.pkl"
	with open(model_path, "rb") as f:
		blackbox = pickle.load(f)

	generated_path = Path(OUTPUT) / f"{project}/generated"
	rules_path = Path(OUTPUT) / f"{project}/SQARules/{mode}"
	rules_path.mkdir(parents=True, exist_ok=True)
	plan_path = Path(PLANS) / f"{project}/SQAPlanner/{mode}"
	plan_path.mkdir(parents=True, exist_ok=True)

	len_csv = len(list(generated_path.glob("*.csv")))
	if len_csv == 0:
		return

	for csv in tqdm(generated_path.glob("*.csv"), desc=f"{project}", leave=False, total=len_csv):
		if Path(plan_path / f"{csv.stem}.csv").exists():
			continue

		case_data = test.loc[int(csv.stem), :]
		x_test = case_data.drop("target")
		real_target = blackbox.predict(x_test.values.reshape(1, -1))


		if case_data['target'] == 0 or real_target == 0:
			continue

		dataset_id = get_or_create_dataset(api, str(csv), project)
		options = {
			'name': csv.stem,
			'tags': [project, mode],
			'search_strategy': mode,
			'max_k': 10,
			"max_lhs": 5,
			'rhs_predicate': [{"field": "target", "operator": "=", "value": "0"}]
		}
		file = rules_path / f'{csv.stem}.csv'
		get_or_create_association(api, dataset_id, options, str(file))

		ff_df= pd.DataFrame([])

		real_target= 'target=='+str(real_target[0])+str('.000')

		rules_df = pd.read_csv(file, encoding='utf-8')

		x_test = x_test.to_frame().T
	
		for  index, row in rules_df.iterrows():
			rule = rules_df.iloc[index, 1]
			rule = comparison(rule)
			
			class_val = rules_df.iloc[index, 2]
			class_val = comparison(class_val)

			if real_target==class_val:
				print("correctly predicted")
	
			#Practices  to  follow  to  decrease  the  risk  of  having defects
			elif x_test.eval(rule).all()==False and real_target!=class_val:
				ff_df = pd.concat([ff_df,row.to_frame().T])

		# Sort by coverage and confidence
		if ff_df.empty:
			df = pd.DataFrame([], columns=['Antecedent', 'Antecedent Coverage %', 'Confidence', 'Lift'])
			df = df.reset_index(drop=True)
			df.to_csv(plan_path / f"{csv.stem}.csv", index=False)
		else:
			ff_df = ff_df[['Antecedent', 'Antecedent Coverage %', 'Confidence', 'Lift']]
			ff_df = ff_df.reset_index(drop=True)
			ff_df = ff_df.head(10)
		ff_df.to_csv(plan_path / f"{csv.stem}.csv", index=False)
				
	# Setting the file name (without extension) as the index name
def main(projects, mode):
	for proj in tqdm(projects, desc="Generating Plans ...", leave=True):
		generate_plans(proj, mode)

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--project', type=str)
	argparser.add_argument('--mode', type=str, default="coverage")
	args = argparser.parse_args()
	count = 0
	while True:
		try:
			print(f"Running... ({count})")
			if args.project:
				main(args.project.split(' '), args.mode)
			else:
				projects = read_dataset()
				projects = list(projects.keys())
				main(projects, args.mode)
				
			break
		except Exception as e:
			print(e)
			print("Error occurred. Restarting...")
			time.sleep(10)
			count += 1


