"""Utilities for machine learning pipelines used in the evictions-learn project."""
import os
from io import StringIO
import numpy as np
import pandas as pd
import json
from dateutil import parser
import matplotlib
matplotlib.use('agg')
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
from model_result import BAG, DT, GB, LR, RF, SVM
import itertools

logger = logging.getLogger('evictionslog')
GRID_1 = "small grid"
GRID_2 = "large grid"
GRID_3 = "final grid"


class Pipeline():

    def __init__(self):
        self.db = DBClient()
        self.df = pd.DataFrame()
        self.feature_sets = {}
        self.run_number = 1
        self.prediction_windows = 0
        self.temporal_lags = 0
        self.feature_combos = 0
        self.predictors = 0
        self.models = 0
        self.gridsize = 0

        self.classifiers = {'RF': {
            "type": RandomForestClassifier(),
            GRID_1: {'n_estimators': [10], 'max_depth': [5], 'max_features': ['sqrt'], 'min_samples_split': [10]},
            GRID_2: {'n_estimators': [5, 10], 'max_depth': [
                5, 10, 20], 'max_features': ['sqrt']},
            GRID_3: {'n_estimators': [100], 'max_depth': [10, 20], 'max_features': [
                'sqrt', 'log2'], 'min_samples_split': [2, 10]}
        },
            'LR': {
            "type": LogisticRegression(),
            GRID_1: {'penalty': ['l2'], 'C': [0.01]},
            GRID_2: {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10]},
            GRID_3: {}
        },
            'NB': {
            "type": GaussianNB(),
            GRID_1: {},
            GRID_2: {},
            GRID_3: {}
        },
            'GB': {
            "type": GradientBoostingClassifier(),
            GRID_1: {'n_estimators': [5], 'learning_rate': [0.5], 'subsample': [0.5], 'max_depth': [5]},
            GRID_2: {'n_estimators': [5, 10], 'learning_rate': [
                0.001, 0.01, 0.5], 'subsample': [0.1, 0.5], 'max_depth': [5, 20]},
            GRID_3: {}
        },
            'BAG': {
            "type": BaggingClassifier(),
            GRID_1: {'n_estimators': [5], 'max_samples': [5], 'max_features': [3], 'bootstrap_features': [True]},
            GRID_2: {'n_estimators': [10, 100], 'max_samples': [
                5, 10], 'max_features': [5, 10], 'bootstrap_features': [True]},
            GRID_3: {}
        },
            'DT': {
            "type": DecisionTreeClassifier(),
            GRID_1: {'criterion': ['gini'], 'max_depth': [20], 'min_samples_split': [10]},
            GRID_2: {'criterion': ['gini'], 'max_depth': [
                20], 'min_samples_split': [10]},
            GRID_3: {}
        },
            'BASELINE_DT': {
            "type": DecisionTreeClassifier(),
            GRID_1: {'criterion': ['gini'], 'max_depth': [3]},
            GRID_2: {'criterion': ['gini', 'entropy'], 'max_depth': [3]},
            GRID_3: {}
        },
            'KNN': {
            "type": KNeighborsClassifier(),
            GRID_1: {'n_neighbors': [10], 'weights': ['distance'], 'algorithm': ['kd_tree']},
            GRID_2: {'n_neighbors': [5, 10, 25], 'weights': [
                'distance', 'uniform'], 'algorithm': ['auto', 'kd_tree']},
            GRID_3: {}
        }}

    ##### Loading Functions #####
    def load_county_data(self, county):
        """Load data from Cook County for local testing."""
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

    def load_data(self, chunksize=1000, max_chunks=None):
        """Return a list of data from the project db cursor.

        Inputs:
                - chunksize (int)
                - nmax_chunks (int)
        Returns:
                - pandas DataFrame
        """
        data = chunk = self.db.cur.fetchmany(chunksize)
        if not max_chunks:
            while chunk != []:
                chunk = self.db.cur.fetchmany(chunksize)
                data.extend(chunk)
        else:
            while chunk != [] and max_chunks > 0:
                max_chunks = max_chunks - 1
                logger.info("Loading chunk....")
                chunk = self.db.cur.fetchmany(chunksize)
                data.extend(chunk)
                logger.info("{} chunks left to load.".format(max_chunks))
        return data

    ##### Processing Fumctions #####
    def discretize_cols(self, dataframe, feature, num_bins=4, labels=False):
        """Discretize features of interest
        Inputs:
            - dataframe (pandas dataframe)
            - feature (string): label of column to discretize
            - num_bins (int): number of bins to discretize into
            - labels

        Returns a pandas dataframe
        """
        series, _ = pd.cut(dataframe[feature], bins=num_bins, labels=[
            'low', 'med-low', 'med-high', 'high'], right=True, include_lowest=True, retbins=True)
        dataframe[feature] = series
        return dataframe

    def cols_with_nulls(self, df):
        """
        Finds list of all columns in a dataframe with null values
        """
        isnull = df.isnull().any()
        isnull_cols = list(isnull[isnull == True].index)

        return isnull_cols

    def fill_nulls(self, df):
        """
        Find values in a dataframe with null values and fill them with the median
        value of that variable

        Inputs:
        - df (DataFrame): Dataset of interest

        Returns the original dataframe with null values filled
        """
        # Find columns with missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        isnull_cols = self.cols_with_nulls(df)

        # Fill nulls with median
        for col in isnull_cols:
            col_median = df[col].median()
            df[col] = df[col].fillna(df[col].median())

        return df

    def get_subsets(self):
        """Returns all possible subsets of defined feature sets"""
        subsets = []
        for i in range(1, len(self.feature_sets.keys()) + 1):

            for combo in itertools.combinations(self.feature_sets.values(), i):
                combo = list(combo)
                combo = [item for sublist in combo for item in sublist]

                combolabels = []
                for key, val in self.feature_sets.items():
                    if self.feature_sets[key][0] in combo:
                        combolabels.append(key)
                subsets.append(
                    {"feature_set_labels": combolabels, "features": combo})
        return subsets

    def joint_sort_descending(self, l1, l2):
        """Jointly sort two numpy arrays."""
        # l1 and l2 have to be numpy arrays
        idx = np.argsort(l1)[::-1]
        return l1[idx], l2[idx]

    def generate_binary_at_k(self, y_scores, k):
        """Turn probabilistic outcomes to binary variable based on threshold k."""
        cutoff_index = int(len(y_scores) * (k / 100.0))
        test_predictions_binary = [
            1 if x < cutoff_index else 0 for x in range(len(y_scores))]
        return test_predictions_binary

    def precision_at_k(self, y_true, y_scores, k):
        """Calculate precision of predictions at a threshold k."""
        y_scores, y_true = self.joint_sort_descending(
            np.array(y_scores), np.array(y_true))
        preds_at_k = self.generate_binary_at_k(y_scores, k)
        precision = precision_score(y_true, preds_at_k)
        return precision

    def recall_at_k(self, y_true, y_scores, k):
        """Calculate recall of predictions at a threshold k."""
        y_scores_sorted, y_true_sorted = self.joint_sort_descending(
            np.array(y_scores), np.array(y_true))
        preds_at_k = self.generate_binary_at_k(y_scores_sorted, k)
        recall = recall_score(y_true_sorted, preds_at_k)
        return recall

    def f1_at_k(self, y_true, y_scores, k):
        """Calculate F1 score of predictions at a threshold k."""
        y_scores, y_true = joint_sort_descending(
            np.array(y_scores), np.array(y_true))
        preds_at_k = generate_binary_at_k(y_scores, k)

        f1 = f1_score(y_true, preds_at_k)

        return f1

    ##### Writeout/Visualization Functions #####
    def plot_precision_recall_n(self, y_true, y_prob, model_name, output_type):
        """Plot and output a precision recall graph for a given model run"""
        y_score = y_prob
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_true, y_score)
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

    def visualize_tree(self, fit, X_train, show=True):
        """Use graphviz package to render a tree and return the file location"""
        num = self.run_number

        viz = tree.export_graphviz(fit, out_file=None, feature_names=X_train.columns,
                                   class_names=['High Risk', 'Low Risk'],
                                   rounded=True, filled=True)

        viz_source = graphviz.Source(viz)
        viz_source.format = 'png'
        viz_source.render('results/images/tree_viz' + str(num), view=False)

        return 'results/images/tree_viz' + str(num) + '.png'

    def generate_feature_importance(self, feature_set_labels, feature_values, match_type, model_key):
        """Generate model-specific feature importance, write out to csv and return file location"""
        outpath = 'results/' + model_key + str(self.run_number) + '.csv'

        logger.debug(feature_set_labels)

        labels = []
        for fset in feature_set_labels:
            labels.extend(self.feature_sets[fset])

        fi = feature_values

        fi_names = pd.DataFrame({match_type: fi, 'feature': labels})
        fi_names.sort_values(by=[match_type], ascending=False, inplace=True)

        fi_names.to_csv(outpath)

        return outpath

    def analyze_bias_and_fairness(self, X_test, y_test, y_pred_probs, bias_features, outcome_label, model_key):
        """Analyze bias of resulting predictions"""
        bias_df = X_test[bias_features]

        for feature in bias_df.columns:
            logger.debug(feature)
            self.discretize_cols(bias_df, feature)
            bias_df[feature] = bias_df[feature].astype(str)

        bias_df['label_value'] = y_test
        bias_df['label_value'] = bias_df['label_value'].astype(str)
        bias_df['score'] = y_pred_probs

        outpath_analysis = 'results/' + model_key + \
            str(self.run_number) + '_feature_analysis.csv'
        analysis_df = X_test.copy()
        analysis_df['label_value'] = y_test
        analysis_df['label_value'] = analysis_df['label_value'].astype(str)
        analysis_df['score'] = y_pred_probs
        analysis_df.to_csv(outpath_analysis)

        outpath_bias = 'results/' + model_key + \
            str(self.run_number) + '_bias.csv'
        bias_df.to_csv(outpath_bias)

    ##### Generate Output Dataframe #####
    def populate_outcome_table(self, train_dates, test_dates, model_key, classifier, params, feature_set_labels, outcome, model_result, y_test, y_pred_probs):
        """Generate tuple of desired model results to include final results dataframe"""
        y_pred_probs_sorted, y_test_sorted = zip(
            *sorted(zip(y_pred_probs, y_test), reverse=True))

        return (train_dates, test_dates, model_key, classifier, params, feature_set_labels, outcome, model_result,
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
                    y_test_sorted, y_pred_probs_sorted, 50.0)
                )

    def get_model_result(self, model_key, fit, X_train, feature_set_labels):
        """Based on model type, generate augmented model result outputs as needed"""
        if model_key == 'DT' or model_key == 'BASELINE_DT':
            graph = self.visualize_tree(fit, X_train, show=False)
            model_result = DT(graph)
        elif model_key == 'SVM':
            model_result = SVM(self.generate_feature_importance(
                feature_set_labels, fit.coef_[0], "coef", model_key))
        elif model_key == 'RF':
            model_result = RF(self.generate_feature_importance(
                feature_set_labels, fit.feature_importances_, "feature_importances", model_key))
        elif model_key == 'LR':
            model_result = LR(self.generate_feature_importance(
                feature_set_labels, fit.coef_[0], "coef", model_key), fit.intercept_,)
        elif model_key == 'GB':
            model_result = GB(self.generate_feature_importance(
                feature_set_labels, fit.feature_importances_, "feature_importances", model_key))
        elif model_key == 'BAG':
            model_result = BAG(fit.base_estimator_, fit.estimators_features_)
        else:
            model_result = None

        return model_result

    ##### Temporal Train/Test Split Generation #####
    def temporal_train_test_sets(self, df, train_start, train_end, test_start, test_end, feature_cols, predictor_col):
        """Return X and y train/test dataframes based on the appropriate timeframes, features, and predictors."""
        # Make train and test dfs based on time of interest, exclude null values in the predictor column
        train_df = df[(df['year'] >= train_start) & (
            df['year'] <= train_end) & (~df[predictor_col].isnull())]
        test_df = df[(df['year'] >= test_start) & (
            df['year'] <= test_end) & (~df[predictor_col].isnull())]

        # Make train and test dataframes based on column inputs
        X_train = train_df[feature_cols]
        y_train = train_df[predictor_col]

        X_test = test_df[feature_cols]
        y_test = test_df[predictor_col]

        return X_train, y_train, X_test, y_test

    ##### Primary Control Structure Functions #####
    def run_temporal(self, df, start, end, prediction_windows, feature_set_list, predictor_col_list, models_to_run,
                     bias_features, grid=GRID_1):
    """Temporal control structure. Builds incremental train/test splits based on prediction windo, trains models
    and generates final output dataframe."""
        self.run_number = 0
        results = []
        self.prediction_windows = len(prediction_windows)
        # Time incrementation
        for prediction_window in prediction_windows:
            train_start = start
            print("Latest")
            print(latest)
            if not latest:
                train_end = train_start + \
                    relativedelta(months=+prediction_window) - \
                    relativedelta(days=+1)
            else:
                train_end = end - \
                    relativedelta(months=+prediction_window) - \
                    relativedelta(days=+1)

            self.temporal_lags = ((end.year - train_end.year)
                                  * 12 + end.month - end.month) / prediction_window
            while train_end + relativedelta(months=+prediction_window) <= end:
                test_start = train_end + relativedelta(days=+1)
                test_end = test_start + \
                    relativedelta(months=+prediction_window) - \
                    relativedelta(days=+1)

                logger.info("\nTemporally validating on:\nTrain: {} - {}\nTest: {} - {}\nPrediction window: {} months\n"
                            .format(train_start, train_end, test_start, test_end, prediction_window))
                # Loop over feature set and precitors
                self.feature_combos = len(feature_set_list)
                for feature_cols in feature_set_list:
                    self.predictors = len(predictor_col_list)
                    for predictor_col in predictor_col_list:
                        # Build training and testing sets
                        X_train, y_train, X_test, y_test = self.temporal_train_test_sets(
                            df, train_start, train_end, test_start, test_end, feature_cols["features"], predictor_col)

                        # Fill nulls here to avoid data leakage
                        X_train = self.fill_nulls(X_train)
                        X_test = self.fill_nulls(X_test)

                        # Build classifiers
                        result = self.classify(models_to_run, X_train, X_test, y_train, y_test,
                                               (train_start, train_end), (test_start,
                                                                          test_end), feature_cols["feature_set_labels"],
                                               predictor_col, bias_features, grid)

                        # Add new result to existing results list
                        results.extend(result)

                # Increment time
                train_end = train_end + \
                    relativedelta(months=+prediction_window)

        # Build results dataframe
        results_df = pd.DataFrame(results, columns=('training_dates', 'testing_dates', 'model_key', 'classifier',
                                                    'parameters', 'feature_sets', 'outcome', 'model_result', 'auc-roc',
                                                    'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50',
                                                    'r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_30', 'r_at_50',
                                                    'f1_at_1', 'f1_at_2', 'f1_at_5', 'f1_at_10', 'f1_at_20', 'f1_at_30', 'f1_at_50'))

        return results_df

    def classify(self, models_to_run, X_train, X_test, y_train, y_test, train_dates, test_dates, feature_set_labels, outcome_label,
                 bias_features, grid):
        """With given train and test splits, feature sets, and outcome, train all models of interest over all parameter combinations"""
        results = []
        self.models = len(models_to_run)
        for model_key in models_to_run:
            # Baseline contingencies
            if model_key == 'BASELINE_RAND':
                pct_negative = len(y_train[y_train == 0]) / len(y_train)
                y_pred_probs = np.random.rand(len(y_test))
                y_pred_probs = [1 if row >
                                pct_negative else 0 for row in y_pred_probs]
                results.append(self.populate_outcome_table(
                    train_dates, test_dates, model_key, model_key, {}, feature_set_labels, outcome_label, None, y_test, y_pred_probs))
            elif model_key == 'BASELINE_PRIOR':
                X_test = X_test[outcome_label + '_lag']
                results.append(self.populate_outcome_table(
                    train_dates, test_dates, model_key, model_key, {}, feature_set_labels, outcome_label, None, y_test, X_test))
            # Main classification loop
            else:
                classifier = self. classifiers[model_key]["type"]
                param_grid = ParameterGrid(
                    self.classifiers[model_key][grid])
                self.gridsize = len(param_grid)
                for params in param_grid:
                    total_runs = self.prediction_windows * self.temporal_lags * \
                        self.feature_combos * self.predictors * self.models * self.gridsize
                    total_runs = int(total_runs)

                    logger.info("Running run # {}/{} with model {}, params {}, features {}, outcome {}".format(
                        self.run_number, total_runs, model_key, params, feature_set_labels, outcome_label))
                    try:
                        classifier.set_params(**params)
                        fit = classifier.fit(X_train, y_train)
                        y_pred_probs = fit.predict_proba(X_test)[:, 1]

                        model_result = None
                        if grid != GRID_1:
                            model_result = self.get_model_result(
                                model_key, fit, X_train, feature_set_labels)

                            self.plot_precision_recall_n(
                                y_test, y_pred_probs, 'images/'+model_key+str(self.run_number), 'save')

                        if len(feature_set_labels) == 4:
                            self.analyze_bias_and_fairness(
                                X_test, y_test, y_pred_probs, bias_features, outcome_label, model_key)

                        results.append(self.populate_outcome_table(
                            train_dates, test_dates, model_key, classifier, params, feature_set_labels, outcome_label, model_result, y_test, y_pred_probs))

                        self.run_number = self.run_number + 1
                    except IndexError as e:
                        logger.error('Error:', e)
                        continue
            logger.debug("{} finished.".format(model_key))

        return results


##### Main Runner #####
def main():
    pipeline = Pipeline()
    data = pipeline.load_data(chunksize=10000, max_chunks=1)
    columns = [desc[0] for desc in pipeline.db.cur.description]
    pipeline.df = pd.DataFrame(data, columns=columns)

    # Remove duplicate year column returned from the db background cursor
    columnNumbers = [x for x in range(pipeline.df.shape[1])]
    columnNumbers.remove(2)  # removing the year column
    pipeline.df = pipeline.df.iloc[:, columnNumbers]
    pipeline.df['year'] = pd.to_datetime(
        pipeline.df['year'].apply(str), format='%Y')

    # Set window of analysis
    start = parser.parse("2006-01-01")
    end = parser.parse("2017-01-01")
    prediction_windows = [12]

    # Load dictionary of feature subsets
    with open("feature_sets.json") as f:
        pipeline.feature_sets = json.load(f)
    # Generate combinations of feature subsets
    all_features = pipeline.get_subsets()

    bias_features = []
    with open("bias_sets.json") as f:
        bias_sets = json.load(f)
        bias_features = bias_sets["bias_features"]

    # Define models and predictors to run
    predictors = ['top20_num', 'e_num_inc_20pct']
    phase1 = {"name": "phase1", "models": [
        'RF', 'LR', 'NB', 'GB', 'BAG', 'KNN', 'DT', 'BASELINE_DT'], "grid": GRID_1, "latest_split": False}
    phase2 = {"name": "phase2", "models": [
        'RF', 'GB', 'BASELINE_DT'], "grid": GRID_2, "latest_split": False}
    phase3 = {"name": "phase3", "models": [
        'RF', 'BASELINE_DT'], "grid": GRID_3, "latest_split": True}

    # Run through 3-phase approach to increase feature and model selectivity
    for phase in [phase3]:
        logger.info("Running {}...".format(phase['name']))
        # Run models over all temporal splits, model parameters, feature
        print(phase['name'])
        print(phase['grid'])
        results_df1 = pipeline.run_temporal(
            pipeline.df, start, end, prediction_windows, all_features,
            predictors, phase['models'], bias_features, grid=phase['grid'], latest=phase['latest_split'])
        logger.info('Finished non-baseline modeling.')

        # Run random and prior year baselines
        prior_features = [{"feature_set_labels": "prior_year",
                           "features": ["top20_num_lag", "e_num_inc_20pct_lag"]}]
        results_df2 = pipeline.run_temporal(pipeline.df, start, end, prediction_windows, prior_features, predictors, [
                                            'BASELINE_RAND', 'BASELINE_PRIOR'], bias_features, grid=phase['grid'], latest=phase['latest_split'])
        logger.info('Finished baseline modeling.')

        # Generate final results dataframe and write to csv
        results_df = results_df1.append(results_df2)
        results_df.to_csv("results/{}_results.csv".format(phase['name']))


if __name__ == "__main__":
    main()
