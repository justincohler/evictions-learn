from abc import ABC

class ModelResult(ABC):


class DT(ModelResult):

    def __init__(self, tree_viz):
        self.type="DT"
        self.tree_viz = None

    def __repr__():
        return self.type + ": {tree vizualization: " + self.tree_viz + "}"

class SVM(ModelResult):

    def __init__(self, coef):
        self.type="SVM"
        self.coef = coef

    def __repr__():
        return self.type + ": {coefficients: " + self.coef + "}"

class RF(ModelResult):

    def __init__(self, feature_importances):
        self.type="RF"
        self.feature_importances = feature_importances

    def __repr__():
        return self.type + ": {feature importances: " + self.feature_importances + "}"

class LR(ModelResult):

    def __init__(self, coef, intercept):
        self.type="LR"
        self.coef = coef
        self.intercept = intercept

    def __repr__():
        return self.type + ": {coefficients: " + self.coef + ", intercept: " + self.intercept +"}"

class GB(ModelResult):

    def __init__(self, feature_importances):
        self.type="GB"
        self.feature_importances = feature_importances

    def __repr__():
        return self.type + ": {feature importances: " + self.feature_importances + "}"

class BAG(ModelResult):

    def __init__(self, base_estimator, estimators_features):
        self.type="BAG"
        self.base_estimator = base_estimator
        self.estimators_features = estimators_features

    def __repr__():
        return self.type + ": {base estimator: " + self.base_estimator + ", estimator features: " + self.estimators_features + "}"        
