from abc import ABC

class ModelResult(ABC):
    def __init__(self):
        self.type="GenericModelResult"

class DT(ModelResult):

    def __init__(self, tree_viz):
        self.type="DT"
        self.tree_viz = tree_viz

    def __repr__(self):
        return "{}: tree_viz: {}".format(self.type, self.tree_viz)

class SVM(ModelResult):

    def __init__(self, coef):
        self.type="SVM"
        self.coef = coef

    def __repr__(self):
        return "{}: coefficients: {}".format(self.type, self.coef)

class RF(ModelResult):

    def __init__(self, feature_importances):
        self.type="RF"
        self.feature_importances = feature_importances

    def __repr__(self):
        return "{}: feature_importances: {}".format(self.type, self.feature_importances)

class LR(ModelResult):

    def __init__(self, coef, intercept):
        self.type="LR"
        self.coef = coef
        self.intercept = intercept

    def __repr__(self):
        return "{}: coefficients: {}, intercept: {}".format(self.type, self.coef, self.intercept)

class GB(ModelResult):

    def __init__(self, feature_importances):
        self.type="GB"
        self.feature_importances = feature_importances

    def __repr__(self):
        return "{}: feature_importances: {}".format(self.type, self.feature_importances)

class BAG(ModelResult):

    def __init__(self, base_estimator, estimators_features):
        self.type="BAG"
        self.base_estimator = base_estimator
        self.estimators_features = estimators_features

    def __repr__(self):
        return "{}: base_estimator: {}, estimator_features: {}".format(self.type, self.base_estimator, self.estimators_features)
