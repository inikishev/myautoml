import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
import copy
import cleanlab.classification, cleanlab.regression
from .utility import ToPandas

class _BaseCleanLearning(BaseEstimator):
    C = cleanlab.classification.CleanLearning
    R = cleanlab.regression.learn.CleanLearning
    def __init__(self, clean_learning: cleanlab.classification.CleanLearning | cleanlab.regression.learn.CleanLearning):
        self.clean_learning = clean_learning

    def _transform(self, X):
        if isinstance(X, np.ndarray): return X
        return self.topandas_.transform(X)

    def fit(self, X, y, sample_weight=None):
        _, y = validate_data(self, X=X, y=y, ensure_all_finite=False)

        self.topandas_ = ToPandas().fit(X)
        X = self._transform(X)

        self.clean_learning_ = copy.deepcopy(self.clean_learning)
        self.clean_learning_.fit(X, y, sample_weight=sample_weight)
        self.clean_learning_.save_space()
        return self

    def predict(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, ensure_all_finite=False, reset=False)

        X = self._transform(X)
        return self.clean_learning_.predict(X) # type:ignore

    def predict_proba(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, ensure_all_finite=False, reset=False)

        X = self._transform(X)
        return self.clean_learning_.predict_proba(X) # type:ignore


class CleanLearningClassifier(ClassifierMixin, _BaseCleanLearning):
    def __init__(self, clean_learning: cleanlab.classification.CleanLearning):
        super().__init__(clean_learning)

class CleanLearningRegressor(RegressorMixin, _BaseCleanLearning):
    def __init__(self, clean_learning: cleanlab.regression.learn.CleanLearning):
        super().__init__(clean_learning)