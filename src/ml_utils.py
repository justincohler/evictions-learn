"""Utilities for machine learning pipelines used in the evictions-learn project."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
from db_init import DBClient
import logging

logger = logging.getLogger('evictionslog')

class Pipeline():

	def __init__(self):
		self.db = DBClient()


	def load_chunk(self, chunksize=1000):
		"""Return a cursor for the sql statement.

		Inputs:
			- chunksize (int)
		Returns:
			- pandas DataFrame
		"""
		l = []

		l = self.db.cur.fetchmany(chunksize)

		print(l[1])
		return pd.DataFrame(l, columns = [
			"_id", "state_code", "geo_id", "year", "name", "parent_location",
			"population", "poverty_rate", "pct_renter_occupied", "median_gross_rent",
			"median_household_income", "median_property_value", "rent_burden", "pct_white",
			"pct_af_am", "pct_hispanic", "pct_am_ind", "pct_asian", "pct_nh_pi", "pct_multiple",
			"pct_other", "renter_occupied_households", "eviction_filings", "evictions",
			"eviction_rate", "eviction_filing_rate", "imputed", "subbed", "state", "county", "tract",
			"geo_id (repeated)", "year (repeated)", "top20_evictions", "top20_eviction_rate"
		])

	def categorical_to_dummy(self, df, column):
		"""Convert a categorical/discrete variable into a dummy variable.

		Inputs:
			- df (DataFrame): Dataset of interest
			- column (str): variable to dummify

		Returns:
			- updated DataFrame
		"""
		return pd.get_dummies(df, columns = [column])

	def continuous_to_categorical(self, df, column, bins = 10, labels = False):
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

	def find_high_corr(self, corr_matrix, threshold, predictor):
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

	def dist_plot(self, df):
	    """Plot a histogram of each variable to show the distribution.

	    Input:
	        df (DataFrame)
	    Returns:
	        grid of histogram plots for each variable in dataframe

	    """
	    plt.rcParams['figure.figsize'] = 16, 12
	    df.hist()
	    plt.show()

	def discretize(self, df, field, bins=None, labels=None):
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

	def plot_by_class(self, df, y):
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

	def fill_missing(self, df, type="mean"):
		"""
		Fill all missing values with the mean value of the given column.

		Inputs:
			- df (DataFrame)
			- type (enum): {"mean", "median"} (Default "mean")

		Returns:
			- df (DataFrame) with missing values imputed

		"""
		if type == "mean":
			return df.fillna(df.mean())
		elif type == "median":
			return df.fillna(df.median())
		else:
			raise TypeError("Type parameter must be one of {'mean', 'median'}.")

	def proba_wrap(self, model, x_data, predict=False, threshold=0.5):
		"""Return a probability predictor for a given threshold."""
		# TODO - implement
		return model.predict_proba(x_data)


	def generate_outcome_table(self):
		"""Return a dataframe with the formatted columns required to present model outcomes.

		Based on Rayid Ghani's magicloops repository: https://github.com/rayidghani/magicloops.

		Inputs:
			- None
		Returns:
			- df (DataFrame) empty wit columns for each run's outcomes.
		"""
		return pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))

	def joint_sort_descending(self, l1, l2):
	    # l1 and l2 have to be numpy arrays
	    idx = np.argsort(l1)[::-1]
	    return l1[idx], l2[idx]

	def generate_binary_at_k(self, y_scores, k):
	    cutoff_index = int(len(y_scores) * (k / 100.0))
	    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
	    return test_predictions_binary

	def precision_at_k(self, y_true, y_scores, k):
	    y_scores, y_true = self.joint_sort_descending(np.array(y_scores), np.array(y_true))
	    preds_at_k = self.generate_binary_at_k(y_scores, k)
	    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
	    #precision = precision[1]  # only interested in precision for label 1
	    precision = precision_score(y_true, preds_at_k)
	    return precision

	def get_classifiers(self):

		classifiers = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
		'LR': LogisticRegression(penalty='l1', C=1e5),
		'NB': GaussianNB(),
		'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
		'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
		'DT': DecisionTreeClassifier(),
		'KNN': KNeighborsClassifier(n_neighbors=3)
		}

		classifier_params = {'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    		'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
			'NB' : {},
			'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
			'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
			'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
			'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}}
		return classifiers, classifier_params
	def classify(self, models_to_run, classifiers, params, X, y):
		"""Runs the loop using models_to_run, clfs, gridm and the data."""
		results_df =  self.generate_outcome_table()
		for n in range(1, 2):
			# create training and valdation sets
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
			for index, classifier in enumerate([classifiers[x] for x in models_to_run]):
				print("Running through model {}...".format(models_to_run[index]))
				parameter_values = params[models_to_run[index]]
				for p in ParameterGrid(parameter_values):
					try:
						classifier.set_params(**p)
						y_pred_probs = classifier.fit(X_train, y_train).predict_proba(X_test)[:,1]
						# you can also store the model, feature importances, and prediction scores
						# we're only storing the metrics for now
						y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
						results_df.loc[len(results_df)] = [models_to_run[index],classifier, p,
							roc_auc_score(y_test, y_pred_probs),
							self.precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
							self.precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
							self.precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
					except IndexError as e:
						print('Error:',e)
						continue
		return results_df

if __name__ == "__main__":
	pipeline = Pipeline()
	iter = pd.read_sql_query("select * from evictions.blockgroup", pipeline.db.conn, chunksize=100)
	print(iter.tail())
