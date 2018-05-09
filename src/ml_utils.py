"""Utilities for machine learning pipelines used in the evictions-learn project."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

def load_data(path, index_col = None):
	"""Load data into pandas DataFrame from a csv.

	Inputs:
		- path (str): Path to location of csv file
		- index_col (str): column to specify as index, defaults to None

	Returns:
		- pandas DataFrame
	"""
	if os.path.exists(path):
	    df = pd.read_csv(path)
	else:
		raise Exception('The file does not exist at this location')

	return df

def categorical_to_dummy(df, column):
	"""Convert a categorical/discrete variable into a dummy variable.

	Inputs:
		- df (DataFrame): Dataset of interest
		- column (str): variable to dummify

	Returns:
		- updated DataFrame
	"""
	return pd.get_dummies(df, columns = [column])

def continuous_to_categorical(df, column, bins = 10, labels = False):
	"""Convert a continuous variable into a categorical variable.

	Inputs:
	- df (DataFrame): Dataset of interest
	- column (str): variable to categorize
	- bins (int): Number of bins to separate data into
	- labels (bool): Indications whether data should be shown as a range or
	numerical value

	Returns:
		- updated DataFrame

	"""
	return pd.cut(df[column], bins, labels = labels)

def find_high_corr(corr_matrix, threshold, predictor):
	"""Find all variables that are highly correlated with the predictor and thus
	likely candidates to exclude.

	Inputs
		- corr_matrix (DataFrame): Result of the "check_correlations" function
		- threshold (int): Value between 0 and 1
		- predictor (str): Predictor variable

	Returns:
		- list of variables highly correlated with the predictor
	"""
	return corr_matrix[corr_matrix[predictor] > threshold].index

def dist_plot(df):
    """Plot a histogram of each variable to show the distribution.

    Input:
        df (DataFrame)
    Returns:
        grid of histogram plots for each variable in dataframe

    """
    plt.rcParams['figure.figsize'] = 16, 12
    df.hist()
    plt.show()

def discretize(df, field, bins=None, labels=None):
	"""Return a discretized Series of the given field.

	Inputs:
		- df (DataFrame): Data to discretize
		- field (str): Field name to discretize
		- bins (int): (Optional) number of bins to split
		- labels (list): (Optional) bin labels (must match # of bins if supplied)

	Returns:
		- A pandas series (to be named by the calling function)

	"""
	if not bins and not labels:
		series = pd.qcut(data[field], q=4)
	elif not labels and bins != None:
		series = pd.qcut(data[field], q=bins)
	elif not bins and labels != None:
		series = pd.qcut(data[field], q=len(labels), labels=labels)
	elif bins != len(labels):
		raise IndexError("Bin size and label length must be equal.")
	else:
		series = pd.qcut(data[field], q=bins, labels=labels)

	return series

def plot_by_class(df, y):
    """Produce plots for each variable in the dataframe showing distribution by
    each value of the dependent variable

    Inputs:
        - df (pandas dataframe)
        - y (str): name of dependent variable for analysis

    Returns:
        - layered histogram plots for each variable in the dataframe

    """
    plt.rcParams['figure.figsize'] = 5, 4
    grp = df.groupby(y)
    var = df.columns

    for v in var:
        getattr(grp, v).hist(alpha=0.4)
        plt.title(v)
        plt.legend([0,1])
        plt.show()

def fill_missing(df, type="mean"):
    """Fill all missing values with the mean value of the given column.

    Inputs:
        - df (DataFrame)
		- type (enum): {"mean", "median"} (Default "mean")
    Returns:
        df (DataFrame) with missing values imputed

	"""
	if type == "mean":
    	return df.fillna(df.mean())
	elif type == "median":
		return df.fillna(df.median())
	else:
		raise TypeError("Type parameter must be one of {'mean', 'median'}.")

def proba_wrap(model, x_data, predict=False, threshold=0.5):
	"""Return a probability predictor for a given threshold.""""
	# TODO - implement
	return model.predict_proba(x_data)

def create_avg_lag_n(origin_table, origin_column, destination_table, lag):
	# TODO
	return NotImplementedError

def create_pct_change_lag_n(origin_table, origin_column, destination_table, lag):
	# TODO
	return NotImplementedError
