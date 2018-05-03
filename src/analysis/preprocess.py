import pandas as pd

def fill_missing_median(data_frame):
    '''
    Fills all missing values with the median value of the given column
    Inputs:
        data_frame (pandas dataframe)
    Returns:
        data_frame (pandas dataframe) with missing values imputed
    '''
    return data_frame.fillna(data_frame.median())

def fill_missing_mean(data_frame):
    '''
    Fills all missing values with the mean value of the given column
    Inputs:
        data_frame (pandas dataframe)
    Returns:
        data_frame (pandas dataframe) with missing values imputed
    '''
    return data_frame.fillna(data_frame.mean())

def drop_vars(data_frame, var_list_to_drop):
	'''
	Drops identified variables from dataframe
	Inputs:
		data_frame (pandas dataframe)
		var_list_to_drop (list): list of variables to drop
	Returns:
		data_frame (pandas dataframe)
	'''
	return data_frame.drop(var_list_to_drop, axis=1)