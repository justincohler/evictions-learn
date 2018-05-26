"""Utilities for machine learning pipelines used in the evictions-learn project."""
import os
import numpy as np
import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy import optimize
from db_init import DBClient
import logging
from validation import *
import graphviz
import warnings

logger = logging.getLogger('evictionslog')


class Pipeline():

    def __init__(self):
        self.db = DBClient()
        self.classifiers = self.generate_classifiers()

    def generate_classifiers(self):

        self.classifiers = {'RF': {
            "type": RandomForestClassifier(),
            "params": {'n_estimators': [10], 'max_depth': [5, 50], 'max_features': ['sqrt'], 'min_samples_split': [10]}
        },
            'LR': {
            "type": LogisticRegression(),
            "params": {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1]}
        },
            'NB': {
            "type": GaussianNB(),
            "params": {}
        },
            'SVM': {
            "type": svm.SVC(probability=True, random_state=0),
            "params": {'C': [1, 10], 'kernel': ['linear']}
        },
            'GB': {
            "type": GradientBoostingClassifier(),
            "params": {'n_estimators': [5, 10], 'learning_rate': [0.5], 'subsample': [0.5], 'max_depth': [1, 5]}
        },
            'BAG': {
            "type": BaggingClassifier(),
            "params": {'n_estimators': [10], 'max_samples': [5], 'max_features': [5, 20], 'bootstrap_features': [False, True]}
        },
            'DT': {
            "type": DecisionTreeClassifier(),
            "params": {'criterion': ['gini', 'entropy'], 'max_depth': [5, 50], 'min_samples_split': [2, 10]}
        },
            'KNN': {
            "type": KNeighborsClassifier(),
            "params": {'n_neighbors': [10, 20], 'weights': ['uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree']}
        }
        }

        return

    def load_county_data(self, county):
        "17031 is cook county"
        l = []
        with self.db.conn.cursor() as cur:
            cur.execute(
                "select * from blockgroup bg join outcome o on bg.geo_id=o.geo_id and bg.year=o.year where bg.county = '17031';")
            l = cur.fetchall()

        return pd.DataFrame(l, columns=[
            "_id", "state_code", "geo_id", "year", "name",
            "parent_location", "population", "poverty_rate", "pct_renter_occupied", "median_gross_rent",
            "median_household_income", "median_property_value", "rent_burden", "pct_white", "pct_af_am",
            "pct_hispanic", "pct_am_ind", "pct_asian", "pct_nh_pi", "pct_multiple",
            "pct_other", "renter_occupied_households", "eviction_filings", "evictions", "eviction_rate",
            "eviction_filing_rate", "imputed", "stubbed", "state", "county",
            "tract", "population_pct_change_5yr", "geo_id (repeated)", "year (repeated)", "top20_num",
            "top20_rate", "top20_num_01"
        ])

    def load_chunk(self, chunksize=1000):
        """Return a cursor for the sql statement.

        Inputs:
                - chunksize (int)
        Returns:
                - pandas DataFrame
        """
        l = []

        l = self.db.cur.fetchmany(chunksize)
        columns = [desc[0] for desc in self.db.cur.description]
        return pd.DataFrame(l, columns=columns)

    def categorical_to_dummy(self, df, column):
        """Convert a categorical/discrete variable into a dummy variable.

        Inputs:
                - df (DataFrame): Dataset of interest
                - column (str): variable to dummify

        Returns:
                - updated DataFrame
        """
        return pd.get_dummies(df, columns=[column])

    def continuous_to_categorical(self, df, column, bins=10, labels=False):
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
        return pd.cut(df[column], bins, labels=labels)

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
            plt.legend([0, 1])
            plt.show()

    def cols_with_nulls(self, df):
        '''
        '''
        isnull = df.isnull().any()
        isnull_cols = list(isnull[isnull == True].index)
        
        return isnull_cols

    def fill_nulls(self,df):
        '''
        Find values in a dataframe with null values and fill them with the median
        value of that variable

        Inputs:
        - df (DataFrame): Dataset of interest

        Returns the original dataframe with null values filled
        '''
        # Find columns with missing values
        isnull_cols = self.cols_with_nulls(df)

        # Fill nulls with median
        for col in isnull_cols:
            col_median = df[col].median()
            df[col].fillna(col_median, inplace = True)

        # Drop cols with all NA's
        #before = df.columns
        #df.dropna(axis = 1, how = "all", inplace = True)
        #after = df.columns

        #diff = [x for x in before if x not in after]
        #print(diff)

        return df

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
        return pd.DataFrame(columns=('model_type', 'clf', 'parameters', 'auc-roc', 'p_at_5', 'p_at_10', 'p_at_20'))

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
        # precision = precision[1]  # only interested in precision for label 1
        precision = precision_score(y_true, preds_at_k)
        return precision

    def recall_at_k(self, y_true, y_scores, k):
        y_scores_sorted, y_true_sorted = self.joint_sort_descending(
            np.array(y_scores), np.array(y_true))
        preds_at_k = self.generate_binary_at_k(y_scores_sorted, k)
        recall = recall_score(y_true_sorted, preds_at_k)
        return recall

    def f1_at_k(self, y_true, y_scores, k):
        y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
        preds_at_k = generate_binary_at_k(y_scores, k)

        f1 = f1_score(y_true, preds_at_k)

        return f1

    def plot_precision_recall_n(self, y_true, y_prob, model_name, output_type):
        y_score = y_prob
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in pr_thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, 'b')
        ax1.set_xlabel('percent of population')
        ax1.set_ylabel('precision', color='b')
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, 'r')
        ax2.set_ylabel('recall', color='r')
        ax1.set_ylim([0, 1])
        ax1.set_ylim([0, 1])
        ax2.set_xlim([0, 1])

        name = model_name
        plt.title(name)
        if (output_type == 'save'):
            plt.savefig(name, close=True)
        elif (output_type == 'show'):
            plt.show()
            plt.close()
        else:
            plt.show()
            plt.close()

    def temporal_train_test_sets(self, df, train_start, train_end, test_start, test_end, feature_cols, predictor_col):
        """Return X and y train/test dataframes based on the appropriate timeframes, features, and predictors."""
        train_df = df[(df['year'] >= train_start) & (df['year'] <= train_end)]
        test_df = df[(df['year'] >= test_start) & (df['year'] <= test_end)]

        X_train = train_df[feature_cols]
        y_train = train_df[predictor_col]

        X_test = test_df[feature_cols]
        y_test = test_df[predictor_col]

        return X_train, y_train, X_test, y_test

    def generate_feature_subsets(self, feature_set_list):
        all_predictors=[]

        predictor_subsets = get_subsets(feature_set_list)

        for p in predictor_subsets:
            merged = list(itertools.chain.from_iterable(p))
            all_predictors.append(merged)

        return all_predictors

    def visualize_tree(self, fit, X_train, show=True):
        viz = tree.export_graphviz(fit, out_file="tree.dot", feature_names=X_train.columns,
                           class_names=['High Risk', 'Low Risk'],
                           rounded=True, filled=True)
        if show:
            f = open("tree.dot")
            dot_graph = f.read()
            graph = graphviz.Source(dot_graph)
            return graph

    # Update feature_set from "" once defined
    def populate_outcome_table(self, train_dates, test_dates, model_key, classifier, outcome, params, y_test, y_pred_probs):
        y_pred_probs_sorted, y_test_sorted = zip(
            *sorted(zip(y_pred_probs, y_test), reverse=True))

        return (train_dates, test_dates, model_key, classifier, params, "feature_set", outcome,
                roc_auc_score(y_test, y_pred_probs),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 1.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 2.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 5.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 10.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 20.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 30.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 50.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 1.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 2.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 5.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 10.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 20.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 30.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 50.0),
                self.f1_at_k(
                    y_test_sorted, y_pred_probs_sorted, 1.0),
                self.f1_at_k(
                    y_test_sorted, y_pred_probs_sorted, 2.0),
                self.f1_at_k(
                    y_test_sorted, y_pred_probs_sorted, 5.0),
                self.f1_at_k(
                    y_test_sorted, y_pred_probs_sorted, 10.0),
                self.f1_at_k(
                    y_test_sorted, y_pred_probs_sorted, 20.0),
                self.f1_at_k(
                    y_test_sorted, y_pred_probs_sorted, 30.0),
                self.f1_at_k(
                    y_test_sorted, y_pred_probs_sorted, 50.0),
                )

    def run_temporal(self, df, start, end, prediction_windows, feature_set_list, predictor_col_list, models_to_run, baselines_to_run=None):
        results = []
        for prediction_window in prediction_windows:
            train_start = start
            train_end = train_start + \
                relativedelta(months=+prediction_window) - relativedelta(days=+1)
            while train_end + relativedelta(months=+prediction_window) <= end:
                test_start = train_end + relativedelta(days=+1)
                test_end = test_start + \
                    relativedelta(months=+prediction_window) - relativedelta(days=+1)

                logger.info("\nTemporally validating on:\nTrain: {} - {}\nTest: {} - {}\nPrediction window: {} months\n"
                            .format(train_start, train_end, test_start, test_end, prediction_window))
                # Loop over feature set and precitors
                for feature_cols in feature_set_list:
                    for predictor_col in predictor_col_list:

                        # Build training and testing sets
                        X_train, y_train, X_test, y_test = self.temporal_train_test_sets(
                            df, train_start, train_end, test_start, test_end, feature_cols, predictor_col)
                        #before_fill = (X_train, X_test)
                        # Fill nulls here to avoid data leakage
                        X_train = self.fill_nulls(X_train)
                        X_test = self.fill_nulls(X_test)
                        #after_fill = (X_train, X_test)

                        #return before_fill, after_fill
                        # Build classifiers
                        result = self.classify(models_to_run, X_train, X_test, y_train, y_test,
                                               (train_start, train_end), (test_start, test_end), predictor_col, baselines_to_run)
                        # Increment time
                        train_end += relativedelta(months=+prediction_window)
                        results.extend(result)
#
        results_df = pd.DataFrame(results, columns=('training_dates', 'testing_dates', 'model_key', 'classifier',
                                                    'parameters', 'feature_set', 'outcome', 'auc-roc', 
                                                    'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30','p_at_50', 
                                                    'r_at_1', 'r_at_2','r_at_5', 'r_at_10', 'r_at_20', 'r_at_30','r_at_50',
                                                    'f1_at_1', 'f1_at_2','f1_at_5', 'f1_at_10', 'f1_at_20', 'f1_at_30','f1_at_50'))

        return results_df

    def classify(self, models_to_run, X_train, X_test, y_train, y_test, train_dates, test_dates, outcome_label, baselines_to_run=None):

        self.generate_classifiers()
        results = []
        for model_key in models_to_run:
            count = 1
            logger.info("Running {}...".format(model_key))
            classifier = self. classifiers[model_key]["type"]
            grid = ParameterGrid(self.classifiers[model_key]["params"])
            for params in grid:
                logger.info("Running with params {}".format(params))
                try:
                    classifier.set_params(**params)
                    fit = classifier.fit(X_train, y_train)
                    y_pred_probs = fit.predict_proba(X_test)[:, 1]
                    
                    # Printing graph section, pull into function
                    if model_key == 'DT':
                        graph = self.visualize_tree(fit, X_train)
                        graph

                    results.append(self.populate_outcome_table(
                        train_dates, test_dates, model_key, classifier, params, outcome_label, y_test, y_pred_probs))

                    self.plot_precision_recall_n(
                        y_test, y_pred_probs, model_key+str(count), 'save')
                    count = count + 1

                except IndexError as e:
                    print('Error:', e)
                    continue
            logger.info("{} finished.".format(model_key))

        if baselines_to_run != None:
            for baseline in baselines_to_run:
                if baseline == "RAND":
                    pct_negative = len(y_train[y_train == 0])/len(y_train)
                    y_pred_probs = np.random.rand(len(y_test))
                    y_pred_probs = [1 if row > pct_negative else 0 for row in y_pred_probs]
                    results.append(self.populate_outcome_table(
                        baseline, baseline, {}, y_test, y_pred_probs))
        return results


def main():
    pipeline = Pipeline()
    logger.info("Loading chunk....")
    df = pipeline.load_chunk()
    logger.info("Chunk loaded.")
    columnNumbers = [x for x in range(df.shape[1])]  # list of columns' integer indices

    columnNumbers.remove(2)  # removing the year column
    df = df.iloc[:, columnNumbers]

    df['year'] = pd.to_datetime(df['year'].apply(str), format='%Y')

    # Set time period
    start = parser.parse("2006-01-01")
    end = parser.parse("2016-01-01")
    prediction_windows = [12]
    
    # Define feature sets
    features = ['population', 'poverty_rate', 
    'pct_renter_occupied', 'median_gross_rent', 'median_household_income', 'median_property_value', 
    'rent_burden', 'pct_white', 'pct_af_am', 'pct_hispanic', 'pct_am_ind', 'pct_asian', 'pct_nh_pi', 
    'pct_multiple', 'pct_other', 'renter_occupied_households', 'eviction_filings', 
    'eviction_filing_rate', 'imputed', 'subbed', 'population_pct_change_5yr', 
    'poverty_rate_pct_change_5yr', 'pct_renter_occupied_pct_change_5yr', 'median_gross_rent_pct_change_5yr', 
    'median_household_income_pct_change_5yr', 'median_property_value_pct_change_5yr', 'rent_burden_pct_change_5yr', 
    'pct_white_pct_change_5yr', 'pct_af_am_pct_change_5yr', 'pct_hispanic_pct_change_5yr', 'pct_am_ind_pct_change_5yr', 
    'pct_asian_pct_change_5yr', 'pct_nh_pi_pct_change_5yr', 'pct_multiple_pct_change_5yr', 'pct_other_pct_change_5yr', 
    'renter_occupied_households_pct_change_5yr', 'eviction_filings_pct_change_5yr', 
    'eviction_filing_rate_pct_change_5yr', 'renter_occupied_households_pct_change_1yr']
    feature_set_list = [features]

    excluded = ['top20_rate','state_code', 'geo_id', 'year', 'name', 'parent_location','evictions_inc_10pct_5yr', 'evictions_dec_10pct_5yr', 
    'evictions_inc_20pct_5yr', 'evictions_dec_20pct_5yr', 'top20_num', 'top20_num_01', 'top20_rate_01', 
    'top10_num', 'top10_rate', 'top10_num_01', 'avg_hh_size', 'top10_rate_01', 'testcol' 'state', 'county', 'tract', 'pct_renter_occupied_pct_change_1yr',
    'evictions_pct_change_5yr', 'eviction_rate_pct_change_5yr','conversion_rate', 'evictions', 'eviction_rate'  ]

    # check pct renter occupied pct change 1 year
    
    predictor_col_list = ['top20_rate']
    models_to_run = ['RF', 'DT']
    results_df = pipeline.run_temporal(
        df, start, end, prediction_windows, feature_set_list, predictor_col_list, models_to_run)

    #results_df.to_csv('test_results.csv')
    return results_df

if __name__ == "__main__":
    main()
