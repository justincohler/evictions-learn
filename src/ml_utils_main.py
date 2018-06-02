"""Utilities for machine learning pipelines used in the evictions-learn project."""
import os
from io import StringIO
import numpy as np
import pandas as pd
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
from aequitas.bias import Bias
from aequitas.group import Group
from aequitas.fairness import Fairness

logger = logging.getLogger('evictionslog')
#comment to push again

class Pipeline():

    def __init__(self):
        self.db = DBClient()
        self.df = pd.DataFrame()
        self.classifiers = self.generate_classifiers()
        self.feature_dict = {}
        self.run_number = 1
        self.prediction_windows = 0
        self.temporal_lags = 0
        self.feature_combos = 0
        self.predictors = 0
        self.models = 0
        self.gridsize = 0

    def generate_classifiers(self):
        self.classifiers = {'RF': {
            "type": RandomForestClassifier(),
            "params": {'n_estimators': [10], 'max_depth': [5, 10], 'max_features': ['sqrt']}
        },
        'GB': {
            "type": GradientBoostingClassifier(),
            "params": {'n_estimators': [5], 'learning_rate': [0.5], 'subsample': [0.5], 'max_depth': [5]}
        }
        }

        return

    def load_chunk(self, chunksize=1000):
        """Return a cursor for the sql statement.

        Inputs:
                - chunksize (int)
        Returns:
                - pandas DataFrame
        """
        l = []

        l = self.db.cur.fetchmany(chunksize)
        return l

    def discretize_cols(self, data_frame, var, labels=False, num_bins=4):
        new_var= "{}_bins".format(var)
        data_frame[new_var], _ = pd.cut(data_frame[var], bins=num_bins, labels=['low', 'med-low', 'med-high', 'high'], right=True, include_lowest=True, retbins = True)
        return data_frame

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
        df = df.replace([np.inf, -np.inf], np.nan)
        isnull_cols = self.cols_with_nulls(df)

        # Fill nulls with median
        for col in isnull_cols:
            col_median = df[col].median()
            df[col] = df[col].fillna(df[col].median())

        return df

    def proba_wrap(self, model, x_data, predict=False, threshold=0.5):
        """Return a probability predictor for a given threshold."""
        # TODO - implement
        return model.predict_proba(x_data)

    def get_subsets(self):
        subsets = []
        for i in range(1, len(self.feature_dict.keys())+1):

                for combo in itertools.combinations(self.feature_dict.values(), i):
                    combo = list(combo)
                    combo = [item for sublist in combo for item in sublist]

                    combolabels = []
                    for key, val in self.feature_dict.items():
                        if self.feature_dict[key][0] in combo:
                            combolabels.append(key)
                    subsets.append({"feature_set_labels": combolabels, "features": combo})
        return subsets

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
        train_df = df[(df['year'] >= train_start) & (df['year'] <= train_end) & (~df[predictor_col].isnull())]
        test_df = df[(df['year'] >= test_start) & (df['year'] <= test_end) & (~df[predictor_col].isnull())]

        X_train = train_df[feature_cols]
        y_train = train_df[predictor_col]

        X_test = test_df[feature_cols]
        y_test = test_df[predictor_col]

        return X_train, y_train, X_test, y_test

    def visualize_tree(self, fit, X_train, show=True):
        num = self.run_number

        viz = tree.export_graphviz(fit, out_file=None, feature_names=X_train.columns,
                           class_names=['High Risk', 'Low Risk'],
                           rounded=True, filled=True)

        viz_source = graphviz.Source(viz)
        viz_source.format = 'png'
        viz_source.render('tree_viz'+str(num), view=False)

        return 'images/tree_viz'+str(num)+'.png'

    def populate_outcome_table(self, train_dates, test_dates, model_key, classifier, params, feature_set_labels, outcome, model_result, y_test, y_pred_probs):
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

    def run_temporal(self, df, start, end, prediction_windows, feature_set_list, predictor_col_list, models_to_run, disc_cols, bias_cols):
        self.run_number = 0
        results = []
        self.prediction_windows = len(prediction_windows)
        for prediction_window in prediction_windows:
            train_start = start
            # Set explicitly to end of 2015 to train on 2016
            train_end = parser.parse("2015-12-31")
            self.temporal_lags =  ((end.year - train_end.year)*12 + end.month - end.month)/prediction_window
            while train_end + relativedelta(months=+prediction_window) <= end:
                test_start = train_end + relativedelta(days=+1)
                test_end = test_start + \
                    relativedelta(months=+prediction_window) - relativedelta(days=+1)

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
                                               (train_start, train_end), (test_start, test_end), feature_cols["feature_set_labels"],
                                               predictor_col, disc_cols, bias_cols)

                        results.extend(result)

                # Increment time
                train_end = train_end + relativedelta(months=+prediction_window)

        # Build results dataframe
        results_df = pd.DataFrame(results, columns=('training_dates', 'testing_dates', 'model_key', 'classifier',
                                                    'parameters', 'feature_sets', 'outcome', 'model_result', 'auc-roc',
                                                    'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30','p_at_50',
                                                    'r_at_1', 'r_at_2','r_at_5', 'r_at_10', 'r_at_20', 'r_at_30','r_at_50',
                                                    'f1_at_1', 'f1_at_2','f1_at_5', 'f1_at_10', 'f1_at_20', 'f1_at_30','f1_at_50'))

        return results_df

    def match_label_array(self, feature_set_labels, feature_values, match_type, model_key):
        run = self.run_number
        outpath = 'results/'+model_key+str(run)+'.csv'

        labels = []
        for fset in feature_set_labels:
            labels.extend(self.feature_dict[fset])

        fi = feature_values

        fi_names = pd.DataFrame({match_type: fi, 'feature': labels})
        fi_names.sort_values(by=[match_type], ascending=False, inplace=True)

        fi_names.to_csv(outpath)

        return outpath

    def classify(self, models_to_run, X_train, X_test, y_train, y_test, train_dates, test_dates, feature_set_labels, outcome_label, disc_cols, bias_cols):

        self.generate_classifiers()
        results = []
        self.models = len(models_to_run)
        for model_key in models_to_run:
            if model_key == 'BASELINE_RAND':
                pct_negative = len(y_train[y_train == 0])/len(y_train)
                y_pred_probs = np.random.rand(len(y_test))
                y_pred_probs = [1 if row > pct_negative else 0 for row in y_pred_probs]
                results.append(self.populate_outcome_table(
                    train_dates, test_dates, model_key,model_key, {}, feature_set_labels, outcome_label, None, y_test, y_pred_probs))
            elif model_key == 'BASELINE_PRIOR':
                X_test = X_test[outcome_label+'_lag']
                results.append(self.populate_outcome_table(
                    train_dates, test_dates, model_key, model_key, {}, feature_set_labels, outcome_label, None, y_test, X_test))
            else:
                classifier = self. classifiers[model_key]["type"]
                grid = ParameterGrid(self.classifiers[model_key]["params"])
                self.gridsize = len(grid)
                for params in grid:
                    total_runs = self.prediction_windows * self.temporal_lags * self.feature_combos * self.predictors  * self.models  * self.gridsize
                    total_runs = int(total_runs)

                    logger.info("Running run # {}/{} with model {}, params {}, features {}, outcome {}".format(self.run_number, total_runs, model_key, params, feature_set_labels, outcome_label))
                    try:
                        classifier.set_params(**params)
                        fit = classifier.fit(X_train, y_train)
                        y_pred_probs = fit.predict_proba(X_test)[:, 1]

                        # Printing graph section, pull into function
                        if (model_key[0:2] == 'RF') or (model_key[0:2] == 'GB'):
                            model_result = RF(self.match_label_array(feature_set_labels, fit.feature_importances_, "feature_importances", model_key))
                        else:
                            model_result = None

                        # Save precicsion recall graph
                        self.plot_precision_recall_n(
                            y_test, y_pred_probs, 'images/'+model_key+str(self.run_number), 'save')

                        self.analyze_bias_and_fairness(X_test, y_test, y_pred_probs, disc_cols, bias_cols, outcome_label, model_key)

                        results.append(self.populate_outcome_table(
                            train_dates, test_dates, model_key, classifier, params, feature_set_labels, outcome_label, model_result, y_test, y_pred_probs))

                        self.run_number = self.run_number + 1
                    except IndexError as e:
                        print('Error:', e)
                        continue
            logger.debug("{} finished.".format(model_key))

        return results

    def analyze_bias_and_fairness(self, X_test, y_test, y_pred_probs, disc_cols, bias_cols, outcome_label, model_key):
        run = self.run_number
        bias_df = X_test.copy()
        for var in disc_cols:
            print(var)
            self.discretize_cols(bias_df, var)

        bias_df = bias_df[bias_cols]
        for col in bias_df.columns:
            bias_df[col] = bias_df[col].astype(str)


        bias_df['label_value'] = y_test
        bias_df['label_value'] = bias_df['label_value'].astype(str)
        bias_df['score'] = y_pred_probs

        g = Group()
        xtab, _ = g.get_crosstabs(bias_df, score_thresholds={'rank_pct': [0.5]})

        b = Bias()
        bdf = b.get_disparity_predefined_groups(xtab, {'pct_renter_occupied_bins':'low',
                                         'pct_white_bins':'low',
                                         'pct_af_am_bins': 'high',
                                         'pct_hispanic_bins' : 'high',
                                         'pct_am_ind_bins' : 'high',
                                         'pct_asian_bins' : 'high',
                                         'pct_nh_pi_bins' : 'high',
                                         'pct_multiple_bins' : 'high',
                                         'pct_other_bins': 'high',
                                         'renter_occupied_households_bins':'high',
                                         'median_household_income_bins':'low',
                                         'median_property_value_bins': 'low',
                                         'urban': '1'})

        f = Fairness()
        fdf = f.get_group_value_fairness(bdf)

        gaf = f.get_group_attribute_fairness(fdf)


        outpath_xtab = 'results/'+model_key+str(run)+'_xtab.csv'
        outpath_bdf = 'results/'+model_key+str(run)+'_bdf.csv'
        outpath_fdf = 'results/'+model_key+str(run)+'_fdf.csv'
        outpath_gaf = 'results/'+model_key+str(run)+'_gaf.csv'

        xtab.to_csv(outpath_xtab)
        bdf.to_csv(outpath_bdf)
        fdf.to_csv(outpath_fdf)
        gaf.to_csv(outpath_gaf)


def main():
    pipeline = Pipeline()
    logger.info("Loading chunk 1....")
    chunk = pipeline.load_chunk(chunksize=20000)
    data = chunk
    #max_chunks = 1
    chunk_ct = 2
    while chunk != []: #and max_chunks > 0:
        #max_chunks = max_chunks - 1
        logger.info("Loading chunk {}....".format(chunk_ct))
        chunk_ct = chunk_ct + 1
        chunk = pipeline.load_chunk(chunksize=200000)
        data.extend(chunk)
        #logger.info("{} chunks left to load.".format(max_chunks))
    logger.info("Finished loading chunks")
    columns = [desc[0] for desc in pipeline.db.cur.description]
    pipeline.df = pd.DataFrame(data, columns=columns)

    columnNumbers = [x for x in range(pipeline.df.shape[1])]  # list of columns' integer indices

    columnNumbers.remove(2)  # removing the year column
    pipeline.df = pipeline.df.iloc[:, columnNumbers]

    pipeline.df['year'] = pd.to_datetime(pipeline.df['year'].apply(str), format='%Y')

    # Set time period
    start = parser.parse("2006- 01-01")
    end = parser.parse("2017-01-01")
    prediction_windows = [12]

    # Define feature sets
    demographic = ['population', 'poverty_rate',
    'pct_renter_occupied', 'pct_white', 'pct_af_am', 'pct_hispanic', 'pct_am_ind', 'pct_asian', 'pct_nh_pi',
    'pct_multiple', 'pct_other', 'renter_occupied_households', 'avg_hh_size',
    'rent_burden','median_gross_rent', 'median_household_income', 'median_property_value',
    'population_avg_5yr','poverty_rate_avg_5yr','median_gross_rent_avg_5yr',
    'median_household_income_avg_5yr','median_property_value_avg_5yr','rent_burden_avg_5yr','pct_white_avg_5yr',
    'pct_af_am_avg_5yr','pct_hispanic_avg_5yr','pct_am_ind_avg_5yr','pct_asian_avg_5yr','pct_nh_pi_avg_5yr',
    'pct_multiple_avg_5yr','pct_other_avg_5yr','renter_occupied_households_avg_5yr','pct_renter_occupied_avg_5yr',
    'population_pct_change_5yr','poverty_rate_pct_change_5yr','median_gross_rent_pct_change_5yr',
    'median_household_income_pct_change_5yr','median_property_value_pct_change_5yr','rent_burden_pct_change_5yr',
    'pct_white_pct_change_5yr','pct_af_am_pct_change_5yr','pct_hispanic_pct_change_5yr','pct_am_ind_pct_change_5yr',
    'pct_asian_pct_change_5yr','pct_nh_pi_pct_change_5yr','pct_multiple_pct_change_5yr','pct_other_pct_change_5yr',
    'renter_occupied_households_pct_change_5yr','pct_renter_occupied_pct_change_5yr']

    economic = ['total_bldg', 'total_units', 'total_value', 'total_bldg_avg_3yr', 'total_units_avg_3yr', 'total_value_avg_3yr',
    'total_bldg_avg_5yr', 'total_units_avg_5yr', 'total_value_avg_5yr', 'total_bldg_pct_change_1yr',
    'total_units_pct_change_1yr', 'total_value_pct_change_1yr', 'total_bldg_pct_change_3yr', 'total_units_pct_change_3yr',
    'total_value_pct_change_3yr', 'total_bldg_pct_change_5yr', 'total_units_pct_change_5yr', 'total_value_pct_change_5yr',
    'div_sa', 'div_enc', 'urban']

    eviction = ['eviction_filings_lag','evictions_lag','eviction_rate_lag','eviction_filing_rate_lag','conversion_rate_lag',
    'eviction_filings_avg_3yr_lag', 'evictions_avg_3yr_lag', 'eviction_rate_avg_3yr_lag', 'eviction_filing_rate_avg_3yr_lag',
    'eviction_filings_avg_5yr_lag', 'evictions_avg_5yr_lag', 'eviction_rate_avg_5yr_lag', 'eviction_filing_rate_avg_5yr_lag',
    'conversion_rate_avg_5yr_lag', 'eviction_filings_pct_change_1yr_lag','evictions_pct_change_1yr_lag','eviction_rate_pct_change_1yr_lag',
    'eviction_filing_rate_pct_change_1yr_lag','conversion_rate_pct_change_1yr_lag','eviction_filings_pct_change_3yr_lag',
    'evictions_pct_change_3yr_lag','eviction_rate_pct_change_3yr_lag','eviction_filing_rate_pct_change_3yr_lag',
    'conversion_rate_pct_change_3yr_lag','eviction_filings_pct_change_5yr_lag','evictions_pct_change_5yr_lag',
    'eviction_rate_pct_change_5yr_lag','eviction_filing_rate_pct_change_5yr_lag','conversion_rate_pct_change_5yr_lag']

    tract = ['population_avg_5yr_tr','poverty_rate_avg_5yr_tr','median_gross_rent_avg_5yr_tr',
    'median_household_income_avg_5yr_tr','median_property_value_avg_5yr_tr','rent_burden_avg_5yr_tr','pct_white_avg_5yr_tr',
    'pct_af_am_avg_5yr_tr','pct_hispanic_avg_5yr_tr','pct_am_ind_avg_5yr_tr','pct_asian_avg_5yr_tr','pct_nh_pi_avg_5yr_tr',
    'pct_multiple_avg_5yr_tr','pct_other_avg_5yr_tr','renter_occupied_households_avg_5yr_tr','pct_renter_occupied_avg_5yr_tr',
    'population_pct_change_5yr_tr','poverty_rate_pct_change_5yr_tr',
    'median_gross_rent_pct_change_5yr_tr','median_household_income_pct_change_5yr_tr','median_property_value_pct_change_5yr_tr',
    'rent_burden_pct_change_5yr_tr','pct_white_pct_change_5yr_tr','pct_af_am_pct_change_5yr_tr','pct_hispanic_pct_change_5yr_tr',
    'pct_am_ind_pct_change_5yr_tr','pct_asian_pct_change_5yr_tr','pct_nh_pi_pct_change_5yr_tr','pct_multiple_pct_change_5yr_tr',
    'pct_other_pct_change_5yr_tr','renter_occupied_households_pct_change_5yr_tr','pct_renter_occupied_pct_change_5yr_tr',
    'eviction_filings_lag_tr','evictions_lag_tr','eviction_rate_lag_tr','eviction_filing_rate_lag_tr','conversion_rate_lag_tr',
    'eviction_filings_avg_3yr_lag_tr', 'evictions_avg_3yr_lag_tr', 'eviction_rate_avg_3yr_lag_tr', 'eviction_filing_rate_avg_3yr_lag_tr',
    'eviction_filings_avg_5yr_lag_tr', 'evictions_avg_5yr_lag_tr', 'eviction_rate_avg_5yr_lag_tr', 'eviction_filing_rate_avg_5yr_lag_tr',
    'conversion_rate_avg_5yr_lag_tr', 'eviction_filings_pct_change_1yr_lag_tr','evictions_pct_change_1yr_lag_tr','eviction_rate_pct_change_1yr_lag_tr',
    'eviction_filing_rate_pct_change_1yr_lag_tr','conversion_rate_pct_change_1yr_lag_tr','eviction_filings_pct_change_3yr_lag_tr',
    'evictions_pct_change_3yr_lag_tr','eviction_rate_pct_change_3yr_lag_tr','eviction_filing_rate_pct_change_3yr_lag_tr',
    'conversion_rate_pct_change_3yr_lag_tr','eviction_filings_pct_change_5yr_lag_tr','evictions_pct_change_5yr_lag_tr',
    'eviction_rate_pct_change_5yr_lag_tr','eviction_filing_rate_pct_change_5yr_lag_tr','conversion_rate_pct_change_5yr_lag_tr']

    disc_cols = ['pct_renter_occupied', 'pct_white', 'pct_af_am', 'pct_hispanic', 'pct_am_ind', 'pct_asian', 'pct_nh_pi',
    'pct_multiple', 'pct_other', 'renter_occupied_households', 'median_household_income', 'median_property_value']

    bias_cols = ['pct_renter_occupied_bins', 'pct_white_bins', 'pct_af_am_bins', 'pct_hispanic_bins', 'pct_am_ind_bins', 'pct_asian_bins', 'pct_nh_pi_bins',
    'pct_multiple_bins', 'pct_other_bins', 'renter_occupied_households_bins', 'median_household_income_bins', 'median_property_value_bins', 'urban']



    #pipeline.feature_dict = {"demographic": demographic,
    #                    "economic": economic,
    #                "eviction": eviction,
    #                "tract": tract
    #                }
    af = demographic + economic + eviction + tract

    pipeline.feature_dict = {"all": af}
    # Generate all feature subsets
    all_features = [{"feature_set_labels": ["all"], "features" : af}]

    # Define models and predictors to run
    models_to_run = ['RF', 'GB']
    predictor_col_list = ['top20_num', 'e_num_inc_20pct']

    # Run models over all temporal splits, model parameters, feature sets
    results_df1 = pipeline.run_temporal(
        pipeline.df, start, end, prediction_windows, all_features, predictor_col_list, models_to_run, disc_cols, bias_cols)
    print('done standard')

    # Run random and prior year baselines
    prior_features = [{"feature_set_labels": "prior_year", "features": ["top20_num_lag", "e_num_inc_20pct_lag"]}]
    results_df2 = pipeline.run_temporal(pipeline.df, start, end, prediction_windows, prior_features, predictor_col_list, ['BASELINE_RAND', 'BASELINE_PRIOR'], disc_cols, bias_cols)
    print('done baseline')

    # Generate final results dataframe and write to csv
    results_df = results_df1.append(results_df2)
    results_df.to_csv('final_run.csv')

    return results_df, pipeline

if __name__ == "__main__":
    main()
