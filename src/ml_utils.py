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
from scipy import optimize
from db_init import DBClient
import logging
from validation import *

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
        print(columns)
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

    def temporal_train_test_sets(self, df, train_start, train_end, test_start, test_end, feature_cols, predictor_col):
        """Return X and y train/test dataframes based on the appropriate timeframes, features, and predictors."""
        print(train_start)
        train_df = df[(df['year'] >= train_start) & (df['year'] <= train_end)]
        test_df = df[(df['year'] >= test_start) & (df['year'] <= test_end)]

        X_train = train_df[feature_cols]
        y_train = train_df[predictor_col]

        X_test = test_df[feature_cols]
        y_test = test_df[predictor_col]

        return X_train, y_train, X_test, y_test

    def run_temporal(self, df, start, end, prediction_windows, feature_cols, predictor_col, models_to_run, baselines_to_run=None):
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
                # Build training and testing sets
                X_train, y_train, X_test, y_test = self.temporal_train_test_sets(
                    df, train_start, train_end, test_start, test_end, feature_cols, predictor_col)
                # Fill nulls here to avoid data leakage
                X_train = self.fill_missing(X_train, "median")
                X_test = self.fill_missing(X_test, "median")
                # Build classifiers
                result = self.classify(models_to_run, X_train, X_test, y_train, y_test,
                                       (train_start, train_end), (test_start, test_end), baselines_to_run)
                # Increment time
                train_end += relativedelta(months=+prediction_window)
                results.extend(result)

        results_df = pd.DataFrame(results, columns=('training_dates', 'testing_dates', 'model_type', 'clf',
                                                    'parameters', 'baseline', 'auc-roc', 'a_at_5', 'a_at_20', 'a_at_50', 'f1_at_5', 'f1_at_20', 'f1_at_50', 'p_at_1', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_50', 'r_at_1', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_50'))

        return results_df

    def classify(self, models_to_run, X_train, X_test, y_train, y_test, train_dates, test_dates, baselines_to_run=None):

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
                    results.append(self.populate_outcome_table(
                        model_key, classifier, params, y_test, y_pred_probs))

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


if __name__ == "__main__":
    pipeline = Pipeline()
    df = pipeline.load_chunk()
    columnNumbers = [x for x in range(df.shape[1])]  # list of columns' integer indices
    columnNumbers.remove(3)  # removing column integer index 0
    df = df.iloc[:, columnNumbers]
    df['year'] = pd.to_datetime(df['year'].apply(str), format='%Y')
    print(df['year'].tail())
    start = parser.parse("2001-01-01")
    end = parser.parse("2016-01-01")
    prediction_windows = [12]
    feature_cols = df.drop(['top20_rate',  'year', 'name', 'parent_location',
                            'state_code', 'geo_id'], axis=1).columns
    predictor_col = 'top20_rate'
    models_to_run = ['RF']
    results_df = pipeline.run_temporal(
        df, start, end, prediction_windows, feature_cols, predictor_col, models_to_run)
