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

def comparison(word):
	regexp =re.finditer(r'\w+=[0-9]+',word)
	regexp_minus_f = re.finditer(r'-[0-9]+.[0-9]+',word)
	regexp_minus = re.finditer(r'-[0-9]+(^.)',word)

	for m in regexp:
		matched_word=m.group()
		new_word="==".join(matched_word.split("="))
		word=word.replace(matched_word, new_word)
	return word


def import_excel(fileName, sheetName, title, dataframe):
    # Create a new workbook and add a worksheet
    book = Workbook()
    book.create_sheet(sheetName)
    
    # Acquire the sheet by its name
    sheet = book[sheetName]
    
    # Writing the title
    sheet.cell(row=1, column=1).value = title
    
    # Writing the dataframe to sheet
    rows = dataframe_to_rows(dataframe, index=False, header=True)
    for r_idx, row in enumerate(rows, 2):  # start from row 2 to leave room for the title
        for c_idx, value in enumerate(row, 1):
            sheet.cell(row=r_idx, column=c_idx, value=value)
    
    # Save the workbook
    book.save(fileName + '.xlsx')


def print_df(df_name,row_pos,col_pos,sheet_name):
	rows = dataframe_to_rows(df_name)
	
	for r_idx, row in enumerate(rows, row_pos):
		for c_idx, value in enumerate(row, col_pos):
			 sheet_name.cell(row=r_idx, column=c_idx, value=value)
	

def checkSheet(file_handler,sheet_name):
	if sheet_name in file_handler.sheetnames:
		sheet_name = file_handler[sheet_name]
	else:
		file_handler.create_sheet(sheet_name)#don't give extension
		sheet_name = file_handler[sheet_name]
	return sheet_name


def checkFile(path,filename):
	my_file = Path(path+filename)
	if my_file.is_file():
		book = load_workbook(filename)
	else:
		book = Workbook()
	return book


def printExcel(path,file, case_df,ff_df):
	book=checkFile(path,str(file)+'.xlsx')

	ff_results =checkSheet(book,'ff_results')
	print_df(case_df,1,1,ff_results)
	print_df(ff_df,2,1,ff_results)

	book.save(path+'/'+str(file)+'.xlsx')

def generate_plans(project):
	username = os.environ['BIGML_USERNAME']
	api_key = os.environ['BIGML_API_KEY']
	api = BigML(username, api_key)

	projects = read_dataset()
	models_path = Path("./models")

	train, test, val = projects[project]
	model_path = models_path / f"{project}.pkl"
	with open(model_path, "rb") as f:
		blackbox = pickle.load(f)

	generated_path = "./output/generated/" + project
	rules_path = Path("./output/rules") / project
	rules_path.mkdir(parents=True, exist_ok=True)
	output_path = Path("./output/SQAPlanner") / project
	output_path.mkdir(parents=True, exist_ok=True)

	print(f"Working on {project}...")
	for csv in tqdm(Path(generated_path).glob("*.csv"), desc="csv", leave=True, total=len(list(Path(generated_path).glob("*.csv")))):
		if Path(output_path / f"{csv.stem}.csv").exists():
			continue
		if Path.exists(output_path / f"{csv.stem}.xlsx"):
			continue

		case_data = test.loc[int(csv.stem), :]
		x_test = case_data.drop("target")
		real_target = blackbox.predict(x_test.values.reshape(1, -1))

		if case_data['target'] == 0 or real_target == 0:
			continue

		dataset_id = get_or_create_dataset(api, str(csv), project)
		options = {
			'name': csv.stem,
			'tags': [project],
			'search_strategy': 'coverage',
			'max_k': 30,
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
		ff_df = ff_df.sort_values(by=['Antecedent Coverage %', 'Confidence'], ascending=False)
		ff_df = ff_df[['Antecedent', 'Antecedent Coverage %', 'Confidence']]
		ff_df = ff_df.reset_index(drop=True)
		ff_df = ff_df.head(10)
		ff_df.to_csv(output_path / f"{csv.stem}.csv", index=False)
				
	# Setting the file name (without extension) as the index name
def main(projects):
	# projects = [
	# 	"activemq@0",
	# 	"activemq@2",
	# 	"camel@0",
	# 	"camel@1",
	# 	"derby@0",
	# 	"groovy@0",
	# 	"hbase@0",
	# 	"hive@0",
	# 	"jruby@0",
	# 	"jruby@1",
	# 	"wicket@0"
	# ]
	for proj in projects:
		generate_plans(proj)

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--project', type=str)
	args = argparser.parse_args()
	count = 0
	while True:
		try:
			print(f"Running... ({count})")
			if args.project:
				main(args.project.split(' '))
			else:
				projects = read_dataset()
				projects = list(projects.keys())
				main(projects)
			break
		except Exception as e:
			print(e)
			print("Error occurred. Restarting...")
			time.sleep(10)
			count += 1


