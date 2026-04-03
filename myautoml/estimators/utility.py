import copy

import numpy as np
import polars as pl
import scipy.special
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..utils.polars_utils import to_dataframe
from ..utils.torch_utils import to_numpy


class ToDtype(TransformerMixin, BaseEstimator):
    """Converts input to ``np.ndarray`` with specified dtype, None to just convert to ``np.ndarray``."""
    def __init__(self, dtype: np.typing.DTypeLike | None):
        self.dtype = dtype

    def fit(self, X, y=None):
        validate_data(self, X=X, y=y, ensure_all_finite=False)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, ensure_all_finite=False)
        if self.dtype is None: return np.asarray(X)
        return np.asarray(X, dtype=self.dtype)


class NanToNum(TransformerMixin, BaseEstimator):
    def __init__(self, nan=0, posinf=None, neginf=None):
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def fit(self, X, y=None):
        validate_data(self, X=X, y=y, ensure_all_finite=False)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, ensure_all_finite=False)
        return np.nan_to_num(X, nan=self.nan, posinf=self.posinf, neginf=self.neginf)


class ToPandas(TransformerMixin, BaseEstimator):
    """Converts input to pandas dataframe."""

    def fit(self, X, y=None):
        validate_data(self, X=X, y=y, ensure_all_finite=False)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, reset=False, ensure_all_finite=False)
        import pandas as pd
        if isinstance(X, pd.DataFrame): return X
        if isinstance(X, pl.DataFrame): return X.to_pandas()
        return pd.DataFrame(to_numpy(X))

class ToList(TransformerMixin, BaseEstimator):
    """For text feature extractors. if COlumn is None, then there should be 1 column in X"""
    def __init__(self, column: str | None = None):
        self.column = column

    def fit(self, X, y=None):
        self.fitted_ = True

        if self.column is None: # X is str
            validate_data(self, X=X, y=y, ensure_all_finite=False, dtype=str)

        else: # X may have other cols
            validate_data(self, X=X, y=y, ensure_all_finite=False)

        return self

    def transform(self, X) -> list[str]:
        check_is_fitted(self)
        self.fitted_ = True

        if self.column is None:
            X = validate_data(self, X=X, ensure_all_finite=False, reset=False, dtype=str)
            if X.ndim == 2: X = np.squeeze(X, -1)
            assert X.ndim == 1
            return X.tolist()

        validate_data(self, X=X, ensure_all_finite=False)
        return to_dataframe(X)[self.column].cast(pl.String).to_list()


class ClassifierWithLabelEncoder(ClassifierMixin, BaseEstimator):
    """For estimators that don't support labels other than integers"""
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_

        y_enc = self.encoder_.transform(y)
        self.fitted_classifier_ = copy.deepcopy(self.classifier).fit(X, y_enc)

    def predict(self, X):
        check_is_fitted(self)
        y_enc = self.fitted_classifier_.predict(X)
        return self.encoder_.inverse_transform(y_enc)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.fitted_classifier_.predict_proba(X)


class RegressorAsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, regressor, softmax: bool = False):
        self.regressor = regressor
        self.softmax = softmax

    def fit(self, X, y):
        _, y = validate_data(self, X=X, y=y, ensure_all_finite=False, dtype=str)
        self.classes_, y = np.unique(y, return_inverse=True)

        if len(self.classes_) == 2:
            self.fitted_regressor_ = copy.deepcopy(self.regressor).fit(X, y)
            self.oh_ = None

        else:
            self.oh_ = OneHotEncoder().fit(y)
            y_oh = self.oh_.transform(y)
            self.fitted_regressor_ = copy.deepcopy(self.regressor).fit(X, y_oh)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return self.classes_[np.argmax(y_proba, axis=1)]

    def predict_proba(self, X):
        y_raw = self.decision_function(X)
        if y_raw.shape[-1] == 1: y_raw = np.squeeze(y_raw, -1)
        if y_raw.ndim == 1: y_raw = np.stack([1-y_raw, y_raw])

        if self.softmax:
            y_proba = scipy.special.softmax(y_raw, -1)
        else:
            y_proba = y_raw / y_raw.sum(-1, keepdims=True)

        return y_proba

    def decision_function(self, X):
        check_is_fitted(self)
        return self.fitted_regressor_.predict(X)
