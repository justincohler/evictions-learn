import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

path = 'data/credit-data.csv'

#### LOAD DATA ####
def load_data(path, index_col = None):
	'''
	Load data into pandas from a csv

	Inputs:
	- path (str): Path to location of csv file
	- index_col (str): column to specify as index, defaults to None

	Returns pandas dataframe
	'''
	if os.path.exists(path):
	    df = pd.read_csv(path)
	else:
		raise Exception('The file does not exist at this location')

	return df


#### EXPLORE DATA ####
def make_histogram(df, var_of_interest, kde = True, rug = False):
	'''
	Make histograms of continuous variables using Seaborn "distplot" function

	Inputs:
	- df (DataFrame): Dataset of interest
	- var_of_interest (str): continuous variable to visualize
	- kde (bool): If true, show distribution trend line
	- rug (bool): If true, show exact numerical location of observations

	No return, shows a histogram
	'''
	# Note: NaN's are exluded
	plot_var = df[var_of_interest]
	sns.distplot(plot_var[~plot_var.isnull()], kde = kde, rug = rug)
	plt.title(var_of_interest + ' Histogram')
	plt.ylabel('Frequency')
	plt.show()


def make_countchart(df, var_of_interest):
	'''
	Make countchart of categorical variables using Seaborn "countplot" function

	Inputs:
	- df (DataFrame): Dataset of interest
	- var_of_interest (str): continuous variable to visualize

	No return, shows a countplot
	'''
	plot_var = df[var_of_interest]
	sns.countplot(plot_var, data=df)
	plt.title(var_of_interest + ' Countplot')
	plt.show()


def check_correlations(df):
	'''
	Check correlations between all variables in a dataframe

	Inputs:
	- df (DataFrame): Dataset of interest

	Returns a pandas datafarme
	'''
	return df.corr()


def find_high_corr(corr_matrix, threshold, predictor_var):
	'''
	Find all variables that are highly correlated with the predictor and thus 
	likely candidates to exclude

	Inputs
	- corr_matrix (DataFrame): Result of the "check_correlations" function
	- threshold (int): Value between 0 and 1
	- predictor_var (str): Predictor variable

	Returns list of variables highly correlated with the predictor_var
	''' 
	return corr_matrix[corr_matrix[predictor_var] > threshold].index

def plot_correlations(df, x, y, hue = None, fitreg = False):
	'''
	Make a scatter plot of using the Seaborn lmplot function

	Inputs:
	- df (DataFrame): Dataset of interest
	- x, y (strs): Variables to identify as x and y
	- hue (str): Optional third variable to color datapoints
	- fitreg (bool): Option to include a fitline

	No return, shows a scatter plot
	'''
	sns.lmplot(x, y, hue = hue, fit_reg = fitreg, data = df)
	plt.title(x + ' vs ' + y)
	plt.show()


#### DATA PRE-PROCESSING/CLEANING ####
def fill_nulls(df):
	'''
	Find values in a dataframe with null values and fill them with the median
	value of that variable

	Inputs:
	- df (DataFrame): Dataset of interest

	Returns the original dataframe with null values filled
	'''
	# Find columns with missing values
	isnull = df.isnull().any()
	isnull_cols = list(isnull[isnull == True].index)

	# Fill nulls with median
	for col in isnull_cols:
		col_mean = df[col].median()
		df[col].fillna(col_mean, inplace = True)

	return df


#### GENERATE FEATURES/PREDICTIONS ####
def x_y_generator(df, feature_cols, predictor_col):
	'''
	Build feature and predictor portions of the dataset

	Inputs:
	- df (DataFrame): Dataset of interest
	- feature_cols (list): Columns to keep as features
	- precictor_col (str): Column to specify as predictor

	Returns a dataframe of features and a dataframe of the predictor
	'''
	all_cols = df.columns
	drop_cols = set(all_cols) - set(feature_cols)

	X = df.drop(drop_cols, axis = 1)

	Y = df[predictor_col]

	return X, Y


def cat_to_dummy(df, var_of_interest):
	'''
	Turn a categorical/discrete variable into a dummy variable

	Inputs:
	- df (DataFrame): Dataset of interest
	- var_of_interest (str): variable to dummify

	Returns an updated dataframe
	'''
	return pd.get_dummies(df, columns = [var_of_interest])

def continuous_to_cat(df, var_of_interest, bins = 10, labels = False):
	'''
	Turn a continuous variable into a categorical variable

	Inputs:
	- df (DataFrame): Dataset of interest
	- var_of_interest (str): variable to categorize
	- bins (int): Number of bins to separate data into
	- labels (bool): Indications whether data should be shown as a range or 
	numerical value

	Returns an updated dataframe
	'''
	return pd.cut(df[var_of_interest], bins, labels = labels)


#### BUILD CLASSIFIER ####
def build_knn(num_neighbors, weights = None, metric = None, metric_params=None):
	'''
	Build a k nearest neighbor classifier using sklearn functionality

	Inputs:
	- num_neighbors (int): number of neighbors 
	- weights (str): How to weight the neighbors
	- metric (str): distance function to use to evaluate neighbors
	- metric_params (dict): specify additional metric parameters as needed

	Returns a classifier
	'''
	model = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights, 
		metric=metric, metric_params=metric_params)
	
	return model

def build_tree():
	'''
	To build later
	'''
	pass

def proba_wrap(model, x_data, predict = False, threshold = 0.5):
	'''
	To build later
	'''
	return model.predict_proba(x_data)


#### EVALUATE CLASSIFIER ####
def check_accuracy(y_actual, y_predict):
	'''
	Check accuracy of a prediction 

	Inputs: 
	- y_actual (Series): Actual predictor values from original dataset
	- y_predict (Series): Predicted values produced by classifier

	Returns float value between 0 and 1
	'''
	return metrics.accuracy_score(y_actual, y_predict)


def precision(y_actual, y_predict):
	'''
	Check precision of a prediction 

	Inputs: 
	- y_actual (Series): Actual predictor values from original dataset
	- y_predict (Series): Predicted values produced by classifier

	Returns float value between 0 and 1
	'''
	return metrics.precision_score(y_actual, y_predict)


def confusion_matrix(y_actual, y_predict):
	'''
	Build confusion matrix based on actual and predicted values

	Inputs: 
	- y_actual (Series): Actual predictor values from original dataset
	- y_predict (Series): Predicted values produced by classifier

	Returns a confusion matrix
	'''
	return metrics.confusion_matrix(y_actual, y_predict, labels=None, 
		sample_weight=None)


def knn_evaluation_matrix(k_range, x_train, y_train, x_test, y_test, 
	metrics = ['minkowski','euclidean', 'manhattan'], 
	weight_funcs = ['uniform', 'distance']):
	'''
	Evaluate models with a variety of different parameters

	Returns a dataframe
	'''
	df = pd.DataFrame(columns = ['num_neighbors', 'metric', 
		'weighting_function', 'training_acc', 'test_acc', 'train_confusion', 'test_confusion'])
	i = 0

	for k in k_range:
	    for metric in metrics:
	        for func in weight_funcs:
	            params = [k, metric, func]
	            model = build_knn(k, func, metric)
	            model.fit(x_train, y_train)
	            train_pred = model.predict(x_train)
	            test_pred = model.predict(x_test)
	            train_acc = check_accuracy(y_train, train_pred)
	            test_acc = check_accuracy(y_test, test_pred)
	            train_confusion = confusion_matrix(y_train, train_pred)
	            test_confusion = confusion_matrix(y_test, test_pred)
	            tup = [k, metric, func, train_acc, test_acc, train_confusion, test_confusion]
	            df.loc[i] = tup
	            i += 1

	return df


