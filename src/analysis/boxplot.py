import pandas as pd
import matplotlib.pyplot as plt
import math

import sys
sys.path.insert(0, '/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/')
from db_init import db_connect

def boxplot(cur = db_connect()[0], var_list = 'all'):
    '''
    Produces box plot for all variables in the dataframe to inspect outliers
    Input:
        data_frame (pandas dataframe)
    Returns:
        grid of box plots for each variable in dataframe
    '''
    plt.rcParams['figure.figsize'] = 16, 12
    if vars != 'all':
        data_frame = data_frame[var_list].copy()
    data_frame.plot(kind='box', subplots=True, 
        layout=(min(4, len(data_frame.columns), math.ceil(len(data_frame.columns)/4)), 
        sharex=False, sharey=False)
    plt.show()