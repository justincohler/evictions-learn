"""
Pipeline is a generic ML pipeline containing basic functions for cleaning, processing, and evaluating a model.

Current Implementations:
    - K-Nearest-Neighbors

@author: Justin Cohler
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas_profiling

class Pipeline(ABC):
    """
    Pipeline is a generic ML pipeline containing basic functions for cleaning, processing, and evaluating a model.

    Current Implementations:
        - K-Nearest-Neighbors
    """

    def ingest(self, source):
        """Return a pandas dataframe of the data from a given source string."""
        return pd.read_csv(source)

    def distribution(self, data):
        """Return the distribution in the dataframe."""
        return pandas_profiling.ProfileReport(data)

    def correlation(self, *fields):
        """Return the correlation matrix between the given fields."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, data):
        """
        Return an updated df, filling missing values for all fields.

        (Uses mean to fill in missing values)
        """
        raise NotImplementedError

    @abstractmethod
    def discretize(self, data, field, bins=None, labels=None):
        """Return a discretized Series of the given field."""
        raise NotImplementedError

    def dummify(self, data, categorical):
        """Return an updated dataframe with binary/dummy fields from the given categorical field."""
        return data.join(pd.get_dummies(data[categorical]))

    def model_and_split(self, data, features, target, test_size=None):
        """Return train and test sets of the feature matrix and target series."""
        X = np.array(data[features])
        y= np.array(data[target])

        if not test_size:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=37)

        return (X_train, X_test, y_train, y_test)

    @abstractmethod
    def classify(self, data, features, target, **kwargs):
        """Return a trained classifier and test data specific to the implementation.

        (e.g. Logistic Regression, Decision Trees)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, classifier, test_features):
        """Return a prediction for the given classifier and test data."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_classifier(self, prediction, test_target):
        """Return evaluation (float) for the implemented classifier."""
        raise NotImplementedError

    @abstractmethod
    def classify_cross_validate(self, data, features, target, **kwargs):
        """Return a cross-validated KNN classifier, as well as remaining test data."""
        raise NotImplementedError
