"""
Neighbors is a basic ML pipeline implementation using a K-Nearest-Neighbors model.

@author: Justin Cohler
"""
from week2.hw2.src.pipeline import Pipeline
from abc import ABCMeta
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class Neighbors(Pipeline):
    """Implement ML pipeline using a K-Nearest-Neighbors model."""

    def __init__(self):
        """Set up k-nearest-neighbor specific globals."""
        super().__init__()

    def preprocess(self, data):
        """
        Return an updated df, filling missing values for all fields.

        (Uses mean to fill in missing values)
        """
        return data.fillna(data.mean())

    # TODO - GRAB THIS
    def discretize(self, data, field, bins=None, labels=None):
        """Return a discretized Series of the given field."""
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

    def classify(self, data, features, target, **kwargs):
        """Return a KNN classifier as well as remaining test data."""
        if kwargs is not None and 'n_neighbors' in kwargs:
            knn = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'])
        else:
            knn = KNeighborsClassifier() # defaults to 5 Neighbors

        X_train, X_test, y_train, y_test = self.model_and_split(data, features, target)
        knn.fit(X_train, y_train)

        return (knn, X_test, y_test)

    def predict(self, classifier, test_features):
        """Return a prediction for the given classifier and test data."""
        return classifier.predict(test_features)

    def evaluate_classifier(self, prediction, test_target):
        """Return evaluation (float) for the implemented classifier."""
        return accuracy_score(test_target, prediction)

    def classify_cross_validate(self, data, features, target, **kwargs):
        """Return a cross-validated KNN classifier, as well as remaining test data."""
        cv_scores = []

        if kwargs is not None and 'k_range' in kwargs:
            for k in kwargs['k_range']:
                knn = KNeighborsClassifier(n_neighbors=k)
                X_train, X_text, y_train, y_test = self.model_and_split(data, features, target)
                if 'folds' not in kwargs:
                    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
                else:
                    scores = cross_val_score(knn, X_train, y_train, cv=kwargs['folds'], scoring='accuracy')
                cv_scores.append(scores.mean())
        else:
            return self.classify(data, features, target)

        print(cv_scores)
