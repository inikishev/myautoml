import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import polars as pl
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..utils.polars_utils import (
    PolarsColumnSelector,
    include_exclude_cols,
    maybe_stack,
    to_dataframe,
    to_lazyframe,
)
from .bases import PolarsTransformer, _PolarsTransformWrapper
from .chain import Chain
from .impute import SimpleImputer
from .one_hot import OneHotEncoder
from .ordinal import BinaryToBool, MapEncoder, OrdinalEncoder
from .scale import MinMaxScaler, StandardScaler
from .select import Cast, DropConstant

class _ToNumpyWrapper(_PolarsTransformWrapper):
    transformer: "ToNumpy"
    feature_names_in_: list[str]

    def __init__(self, transformer: "ToNumpy"):
        super().__init__(transformer)

    def fit(self, X, y=None, df_unlabeled=None, **fit_params):
        if self.transformer.label is not None:
            raise RuntimeError("fit can only be called on transformer whose label=None. Otherwise use set_fitted.")

        validate_data(self, X=X, y=y, skip_check_array=True)
        self.fitted_transformer_ = copy.copy(self.transformer).fit(X, df_unlabeled)
        return self

    def fit_transform(self, X, y=None, df_unlabeled=None, **fit_params):
        if self.transformer.label is not None:
            raise RuntimeError("fit can only be called on transformer whose label=None. Otherwise use set_fitted.")

        validate_data(self, X=X, y=y, skip_check_array=True)
        self.fitted_transformer_ = copy.copy(self.transformer).fit(X, df_unlabeled)

        ret = self.fitted_transformer_.transform_X(X)
        self.n_features_out_ = ret.shape[1]
        return ret

    def set_fitted(self):
        """Marks this estimator as fitted, in case ToNumpy was fitted"""
        if not hasattr(self.transformer, "X_chain_"):
            raise RuntimeError("set_fitted can only be called when ToNumpy was fitted.")

        self.fitted_transformer_ = self.transformer
        return self

    def transform(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, skip_check_array=True, reset=False)

        ret = self.fitted_transformer_.transform_X(X)
        self.n_features_out_ = ret.shape[1]
        return ret

    def inverse_transform(self, X):
        check_is_fitted(self)
        return self.fitted_transformer_.inverse_transform_X(X).collect()

    def validate_data(self, X):
        validate_data(self, X=X, skip_check_array=True, reset=False)

    def transform_y(self, y):
        check_is_fitted(self)
        return self.fitted_transformer_.transform_y(y)

    def inverse_transform_y(self, y):
        return self.fitted_transformer_.inverse_transform_y(y).collect()

    def get_feature_names_out(self, input_features=None):
        return np.arange(self.n_features_out_).astype(np.str_)

class ToNumpy(PolarsTransformer):
    """Encodes the dataframe to ``(X, y)`` numpy arrays.

    Args:
        label: name of label column, if exists, otherwise None.
        target_encoder: encoder to apply to target. Defaults to None.
        scale: whether to scale numeric features. Defaults to False.
        impute: whether to apply SimpleImputer. Defaults to False.
        missing_indicators: whether to add indicators for missing columns. Defaults to False.
        drop_constant: whether to drop constant columns (with one unique value). Defaults to True.
        map: Dictionary with column names as keys, and replacement mappings as values:
            ``{column_name: {value: replace_value}}``.
        drop_first: Whether to drop first category per feature to prevent multi-colinearity.
            - ``"all"`` drops first column in all one-hot encoded columns.
            - ``"binary"`` drops first column in binary features only, making them ordinal (default).
            - ``"none"`` - doesn't drop first columns.

            Note that inverse transform reconstructs dropped columns by assuming a zero-vector maps to the dropped category,
            that means ``inverse_transform`` should not be applied to logits/probabilities unless ``drop_first="none"``.
        min_frequency: features with less than this frequency will be bundled into one.
            If float, relative to total number of samples. Defaults to None.
        max_categories: if specified, one-hot encodes top ``max_categories``
            most common categories and the rest are bundled into one. Defaults to None.
        extra_categorical_cols: cols to treat as categorical in addition to string/categorical data types. Defaults to None.
        binarize_binary: whether to convert features with two unique values to 0 and 1. Defaults to True.
        include: all columns except those ones will be dropped. None means no columns are dropped. Defaults to None.
        exclude: columns to drop after applying ``include``. None means no columns are dropped. Defaults to None.
    """
    def __init__(
        self,
        label: str | None,
        target_encoder: Literal["standard", "minmax", "ordinal", "none"] | None = None,
        scale: bool = False,
        impute: bool = False,
        missing_indicators: bool = False,
        drop_constant: bool = True,
        replace: Mapping[str, Mapping[Any, Any]] | None = None,
        drop_first: Literal["all", "binary", "none"] = "binary",
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
        extra_categorical_cols: PolarsColumnSelector | None = None,
        binarize_binary: bool = True,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
    ):
        self.label = label

        self.scale = scale
        self.impute = impute
        self.target_encoder: Literal["standard", "minmax", "ordinal", "none"] | None = target_encoder
        self.drop_constant = drop_constant
        self.replace = replace
        self.missing_indicators = missing_indicators

        self.drop_first: Literal["all", "binary", "none"] = drop_first
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.extra_categorical_cols = extra_categorical_cols

        self.include = include
        self.exclude = exclude
        self.binarize_binary = binarize_binary

    def fit(self, df, df_unlabeled=None):

        X_stages = []
        y_stages = []

        # Convert to categorical
        if self.extra_categorical_cols is not None:
            X_stages.append(Cast(self.extra_categorical_cols, pl.String))

        # Drop constant
        if self.drop_constant:
            X_stages.append(DropConstant(exclude=self.label))

        # Replace
        if self.replace is not None:
            X_stages.append(MapEncoder(self.replace, unknown_strategy='passthrough'))

        # Impute
        if self.impute:
            X_stages.append(SimpleImputer(exclude=self.label, add_indicator=self.missing_indicators))

        # Scale
        if self.scale:
            X_stages.append(StandardScaler(exclude=self.label))

        # Binary to 0/1
        if self.binarize_binary:
            X_stages.append(BinaryToBool(exclude=self.label, allow_unknown=True))

        # One-hot
        X_stages.append(OneHotEncoder(
            include=pl.selectors.string(include_categorical=True),
            exclude=self.label,
            drop_first=self.drop_first,
            min_frequency=self.min_frequency,
            max_categories=self.max_categories,

        ))

        # Encode target
        if (self.target_encoder is not None) and (self.target_encoder != "none"):
            if self.label is None: raise RuntimeError(f"target_encoder=`{self.target_encoder}`, but label=None.")
            if self.target_encoder == 'minmax': y_stages.append(MinMaxScaler(include=self.label))
            elif self.target_encoder == 'standard': y_stages.append(StandardScaler(include=self.label))
            elif self.target_encoder == 'ordinal': y_stages.append(OrdinalEncoder(include=self.label))

        # Fit
        df = to_lazyframe(df)
        if df_unlabeled is not None: df_unlabeled = to_lazyframe(df_unlabeled)

        self.feature_names_in_ = df.collect_schema().names()

        if self.label is None:
            X = maybe_stack(df, df_unlabeled)

            y = None

        else:
            X = maybe_stack(df.drop(self.label), df_unlabeled)

            y = df.select(self.label)
            self.y_chain_: Chain[pl.LazyFrame] = Chain(y_stages).fit(y)
            y_tfm = self.y_chain_.transform(y).collect()
            self.y_schema_ = y_tfm.collect_schema().copy()


        assert X is not None
        X = include_exclude_cols(X, include=self.include, exclude=self.exclude)
        self.X_chain_: Chain[pl.LazyFrame] = Chain(X_stages).fit(X)
        X_tfm = self.X_chain_.transform(X)
        self.X_schema_ = X_tfm.collect_schema().copy()

        return self

    def transform_X(self, df) -> np.ndarray:
        x = to_lazyframe(df)
        if self.label is not None: x = x.drop(self.label, strict=False)
        x = include_exclude_cols(x, include=self.include, exclude=self.exclude)
        x = self.X_chain_.transform(x.fill_nan(None))
        return x.collect().to_numpy()

    def transform_y(self, df) -> np.ndarray | None:
        if self.label is None: raise RuntimeError("can't use `transform_y` when label=None")
        df = to_lazyframe(df)

        y = df.select(self.label)
        y = self.y_chain_.transform(y)
        return y.collect().to_numpy()

    def transform(self, df) -> tuple[np.ndarray, np.ndarray]:
        df = to_lazyframe(df)
        X = self.transform_X(df)
        if self.label is None: y = None
        else: y = self.transform_y(df)
        return X, cast(np.ndarray, y) # y can be None. but usually label is not None so its just annoying for typing.

    def inverse_transform_X(self, X: np.ndarray | Any) -> pl.LazyFrame:
        X = pl.from_numpy(np.asarray(X), schema=self.X_schema_)
        return self.X_chain_.inverse_transform(X)

    def inverse_transform_y(self, y: np.ndarray | Any) -> pl.LazyFrame:
        y = np.asarray(y)
        if y.ndim == 1: y = np.expand_dims(y, 1)
        y = pl.from_numpy(y, schema=self.y_schema_)
        return self.y_chain_.inverse_transform(y)

    def to_sklearn(self): # pyright:ignore[reportIncompatibleMethodOverride]
        return _ToNumpyWrapper(self)

    def to_frozen(self): # pyright:ignore[reportIncompatibleMethodOverride]
        est = self.to_sklearn().set_fitted()
        feature_names_in = self.feature_names_in_.copy()
        if self.label is not None and self.label in feature_names_in: feature_names_in.remove(self.label)
        est.feature_names_in_ = feature_names_in
        return FrozenEstimator(est)
