import copy
from abc import ABC, abstractmethod
from typing import Any

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.frozen import FrozenEstimator
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..utils.polars_utils import to_lazyframe, to_dataframe


def _maybe_collect(x):
    if isinstance(x, pl.LazyFrame): x = x.collect()
    return x

def _get_feature_names(x):
    if isinstance(x, pl.DataFrame): return x.schema.names()
    if isinstance(x, pl.LazyFrame): return x.collect_schema().names()
    return to_lazyframe(x).collect_schema().names()

class _PolarsTransformWrapper(TransformerMixin, BaseEstimator):
    feature_names_in_: list[str]

    def __init__(self, transformer: "PolarsTransformer", collect: bool = False):
        self.transformer = transformer
        self.collect = collect

    def set_fitted(self):
        """Marks this estimator as fitted, in case transformer was fitted"""
        self.fitted_transformer_ = self.transformer
        return self

    def fit(self, X, y=None, **fit_params):
        validate_data(self, X=X, y=y, skip_check_array=True)

        self.fitted_transformer_ = copy.copy(self.transformer).fit(X)
        return self

    def fit_transform(self, X, y=None, **fit_params): # pyright:ignore[reportIncompatibleMethodOverride]
        validate_data(self, X=X, y=y, skip_check_array=True)

        self.fitted_transformer_ = copy.copy(self.transformer)
        ret = self.fitted_transformer_.fit_transform(X)

        if self.collect: ret = _maybe_collect(ret)
        self.features_names_out_ = _get_feature_names(ret)
        return ret

    def transform(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, skip_check_array=True, reset=False)

        ret = self.fitted_transformer_.transform(X)

        if self.collect: ret = _maybe_collect(ret)
        self.features_names_out_ = _get_feature_names(ret)
        return ret

    def inverse_transform(self, X):
        check_is_fitted(self)

        if hasattr(self.fitted_transformer_, "inverse_transform"):
            return _maybe_collect(getattr(self.fitted_transformer_, "inverse_transform")(X))

        raise NotImplementedError()

    def validate_data(self, X):
        validate_data(self, X=X, skip_check_array=True, reset=False)

    def get_feature_names_out(self, input_features=None):
        return self.features_names_out_

class PolarsTransformer:
    """Base class for a transformer"""

    feature_names_in_: list[str]

    @abstractmethod
    def fit(self, df):
        return self

    @abstractmethod
    def transform(self, df) -> Any:
        """transform"""

    def fit_transform(self, df) -> Any:
        self.fit(df)
        return self.transform(df)

    # def inverse_transform(self, df) -> Any:
    #     raise NotImplementedError(f"{self.__class__.__name__} doesn't implement `inverse_transform`")

    def to_sklearn(self, collect: bool = True):
        return _PolarsTransformWrapper(self, collect=collect)

    def to_frozen(self, collect: bool = True):
        est = self.to_sklearn(collect=collect).set_fitted()
        est.feature_names_in_ = self.feature_names_in_.copy()
        return FrozenEstimator(est)
