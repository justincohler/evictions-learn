import json
import csv
import psycopg2
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#from src.db_init import db_connect
#from src.analysis.data_processing import make_countchart
#from src.analysis.data_processing import make_histogram

def get_var_by_year(var, year, cur, year2=None):
	if year2 is None:
		statement = "SELECT {} FROM evictions.blockgroup WHERE year={}".format(var, year)
	else:
		statement = "SELECT {} FROM evictions.blockgroup WHERE year={} or year={}".format(var, year, year2)
	cur.execute(statement)
	output = cur.fetchall()

	return output

def make_chart(output, title=None, categorical= True):
	output = pd.DataFrame.from_records(output)
	series = output[0].astype(float)

	if categorical:
		sns.countplot(series)
		if title is None:
			plt.title('Countplot')
		else: 
			plt.title(title)
		plt.show()
	else:
		sns.distplot(series[~series.isnull()])
		if title is None:
			plt.title('Histogram')
		else: 
			plt.title(title)
		plt.ylabel('Frequency')
		plt.show()