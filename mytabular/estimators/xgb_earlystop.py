from typing import Literal, Any

import numpy as np
import copy
from scipy.special import expit, softmax # pylint:disable=no-name-in-module
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from xgboost import XGBClassifier, XGBRegressor, XGBRanker, XGBRFClassifier, XGBRFRegressor

class _BaseXGBEarlyStoppingCV(BaseEstimator):
    """pass ``early_stopping_rounds`` and ``eval_metric`` to xgb!"""
    is_classification: bool

    def __init__(
        self,
        xgb: XGBClassifier | XGBRegressor | XGBRanker | XGBRFClassifier | XGBRFRegressor,
        cv,
        shuffle: bool,
        random_state,
        verbose,
    ):
        self.xgb = xgb
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

    def __sklearn_tags__(self):
        return self.xgb.__sklearn_tags__()

    def fit(self, X, y, sample_weight=None):
        _, y = validate_data(self, X=X, y=y)
        if self.is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        cv = self.cv

        if isinstance(cv, int):
            if self.is_classification: cv = StratifiedKFold(cv, shuffle=self.shuffle, random_state=self.random_state)
            else: cv = KFold(cv, shuffle=self.shuffle, random_state=self.random_state)

        fold_indexes = list(cv.split(X, y))

        self.estimators_ = []
        for fold, (train_index, test_index) in enumerate(fold_indexes):
            eval_set=[(X[test_index], y[test_index])]
            xgb = copy.deepcopy(self.xgb).fit(
                X[train_index],
                y[train_index],
                eval_set = eval_set,
                sample_weight = sample_weight,
                verbose = self.verbose
            )
            self.estimators_.append(xgb)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        proba = None
        for est in self.estimators_:
            if proba is None: proba = est.predict_proba(X)
            else: proba += est.predict_proba(X)

        assert proba is not None
        return proba / len(self.estimators_)

    def predict(self, X):
        if self.is_classification:
            return self.classes_[np.argmax(self.predict_proba(X), -1)]

        preds = None
        for est in self.estimators_:
            if preds is None: preds = est.predict(X)
            else: preds += est.predict(X)

        assert preds is not None
        return preds / len(self.estimators_)

    def apply(self, X, model_idx=0):
        return self.estimators_[model_idx].apply(X)



class XBGEarlyStoppingClassifierCV(ClassifierMixin, _BaseXGBEarlyStoppingCV):
    """pass ``early_stopping_rounds`` and ``eval_metric`` to xgb!"""
    is_classification = True

    def __init__(
        self,
        xgb: XGBClassifier | XGBRFClassifier,
        cv: Any = 5,
        shuffle: bool = True,
        random_state = None,
        verbose = False,
    ):
        super().__init__(xgb=xgb,cv=cv,shuffle=shuffle,random_state=random_state,verbose=verbose)

    def __sklearn_tags__(self):
        return self.xgb.__sklearn_tags__()

class XBGEarlyStoppingRegressorCV(RegressorMixin, _BaseXGBEarlyStoppingCV):
    """pass ``early_stopping_rounds`` and ``eval_metric`` to xgb!"""
    is_classification = True

    def __init__(
        self,
        xgb: XGBRegressor | XGBRFRegressor,
        cv: Any = 5,
        shuffle: bool = True,
        random_state = None,
        verbose = False,
    ):
        super().__init__(xgb=xgb,cv=cv,shuffle=shuffle,random_state=random_state,verbose=verbose)

    def __sklearn_tags__(self):
        return self.xgb.__sklearn_tags__()