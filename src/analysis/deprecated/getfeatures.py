import pandas as pd
import numpy as np


def cat_to_dummy(data_frame, var):
	'''
	Generates dummy variables for each category of a categorical variable
	Inputs:
		data_frame (pandas dataframe)
		var (str) name of categorical variable
	Returns:
		data_frame (pandas dataframe) updated to include dummy variables
	'''
	dummy = pd.get_dummies(data_frame[var], prefix=var)
	return pd.concat([data_frame, dummy], axis = 1)

def discretize(data_frame, var, num_bins=10, labels=False):
	'''
	Generates discretized variable with integer values for specified bins from 
	continuous variable
	Inputs:
		data_frame (pandas dataframe)
		var (str) name of categorical variable
		num_buns (int or list): integer specifying number of equal width bins or an array 
			with bin dividing points, default value is 10
		labels (boolean or list): list of bin labels for generated variable, default 
			value is False which labels each bin with an integer
	Returns:
		data_frame (pandas dataframe) updated to include discretized variable
		bins (list): list of bin dividing pointsx
	'''
	new_var = "{}_bins".format(var)
	data_frame[new_var], bins = pd.cut(data_frame[var], bins=num_bins, labels=labels, 
                            right=True, include_lowest=True, retbins = True)
	return data_frame, bins

