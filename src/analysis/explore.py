import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import math
import numpy as np

def read_data(file_path):
    '''
    Function to read in data from csv into pandas dataframe
    Input:
        file_path (str): file path for location of csv file
    Returns:
        pandas dataframe
        
    '''
    return pd.read_csv(file_path)

def dist_table(data_frame):
    '''
    Function to create a table showing the distribution of each variable in a 
    pandas data frame
    Input:
        data_frame (pandas dataframe)
    Returns:
        table showing the distribtion of each variable
    '''
    return data_frame.describe()

def dist_plot(data_frame):
    '''
    Function to plot a histogram of each variable to show the distribution
    Input:
        data_frame (pandas dataframe)
    Returns:
        grid of histogram plots for each variable in dataframe
    '''
    plt.rcParams['figure.figsize'] = 16, 12
    data_frame.hist()
    plt.show()

def outliers_plot(data_frame):
    '''
    Produces box plot for all variables in the dataframe to inspect outliers
    Input:
        data_frame (pandas dataframe)
    Returns:
        grid of box plots for each variable in dataframe
    '''
    plt.rcParams['figure.figsize'] = 16, 12
    data_frame.plot(kind='box', subplots=True, 
        layout=(4, math.ceil(len(data_frame.columns)/4)), 
        sharex=False, sharey=False)
    plt.show()

def corr_table(data_frame):
    '''
    Produces a coorelation table of the pearson correlations between each
    pair of two variables in the data frame.

    Input:
        data_frame (pandas dataframe)
    Returns:
        corr_table(pandas dataframe): dataframe of pairwise variable correlations 
    '''
    return data_frame.corr(method = 'pearson')

def plot_y_corr(corr_table, y):
    '''
    Produces a bar plot showing the correlation of each variable with the 
    dependent variable

    Inputs:
        corr_table(pandas dataframe): dataframe of variable correlations produced
        using corr_table function
        y (str): name of dependent variable for analysis
    Returns:
        bar plot showing correlation of each variable with the dependent variable
    '''
    corr_y = corr_table[y]
    title = "Correlation with {}".format(y)
    corr_y.plot.bar(title = title)
    plt.xlabel("Variable")
    plt.show()

def plot_corr_matrix(corr_table):
    '''
    Produces a plot showing the correlation between all sets of variables in
    the dataframe

    Input:
        corr_table(pandas dataframe): dataframe of variable correlations produced
        using corr_table function
    returns:
        plot showing correlations between all sets of variables in dataframe

    reference: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

    '''
    names = list(corr_table.index)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_table, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names,  rotation = 90)
    ax.set_yticklabels(names)
    
    plt.show()
    
   
def plot_by_class(data_frame, y):
    '''
    Produces plots for each variable in the dataframe showing distribution by
    each value of the dependent variable

    Inputs:
        data_frame (pandas dataframe)
        y (str): name of dependent variable for analysis
    Returns:
        layered histogram plots for each variable in the dataframe
    '''
    plt.rcParams['figure.figsize'] = 5, 4
    grp = data_frame.groupby(y)
    var = data_frame.columns

    for v in var:
        getattr(grp, v).hist(alpha=0.4)
        plt.title(v)
        plt.legend([0,1])
        plt.show()

