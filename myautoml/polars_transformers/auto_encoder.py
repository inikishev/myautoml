import copy
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import polars as pl
from sklearn.frozen import FrozenEstimator
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..utils.polars_utils import maybe_stack, to_dataframe, to_lazyframe, to_series
from .bases import PolarsTransformer, _PolarsTransformWrapper
from .chain import Chain
from .impute import MissingStatistics, SimpleImputer
from .infrequent import MergeInfrequent
from .one_hot import OneHotEncoder
from .ordinal import BinaryToBool, MapEncoder, OrdinalEncoder
from .scale import MinMaxScaler, SampleNormalizer, StandardScaler
from .select import Cast, Collect, DropCols, DropConstant, RemoveDuplicates, SelectCols
from .tonumpy import ToNumpy

if TYPE_CHECKING:
    from ..core._fitter_utils import ProblemType

def _infer_problem_type(y: pl.Series) -> "tuple[ProblemType, str]":

    if y.n_unique() == 2:
        return 'binary', "Target column has two unique values"

    if type(y.dtype) in (pl.String, pl.Categorical):
        return 'multiclass', f"Target column type is is {y.dtype}"

    if y.dtype.is_integer():
        if y.n_unique() < len(y): return 'multiclass', "Target column is integer and has repeating values"
        return 'regression', "Target column is integer but has no repeating values"

    if y.dtype.is_float():
        return 'regression', "Target column is floating point and has more than 2 unique values"

    raise RuntimeError(f"Unable to infer problem type from column {y.name}, dtype={y.dtype}. "
                       "Please pass `problem_type` to AutoMLEncoder manually.")

class _AutoEncoderWrapper(_PolarsTransformWrapper):
    transformer: "AutoEncoder"
    feature_names_in_: list[str]

    def __init__(self, transformer: "AutoEncoder"):
        super().__init__(transformer) # pyright:ignore[reportArgumentType]

    def fit(self, X, y=None, X_unlabeled=None, problem_type: "ProblemType | None" = None, **fit_params):
        validate_data(self, X=X, y=y, skip_check_array=True)
        self.fitted_transformer_ = copy.copy(self.transformer).fit(X, y, X_unlabeled, problem_type)
        self.problem_type_ = self.transformer.problem_type_
        return self

    def fit_transform(self, X, y=None, X_unlabeled=None, problem_type: "ProblemType | None" = None, **fit_params):
        validate_data(self, X=X, y=y, skip_check_array=True)
        self.fitted_transformer_ = copy.copy(self.transformer).fit(X, y, X_unlabeled, problem_type)
        ret = self.fitted_transformer_.transform_X(X)
        self.problem_type_ = self.transformer.problem_type_
        return ret

    def set_fitted(self):
        """Marks this estimator as fitted, in case AutoEncoder was fitted"""
        if not hasattr(self.transformer, "X_chain_"):
            raise RuntimeError("set_fitted can only be called when AutoEncoder was fitted.")

        self.fitted_transformer_ = self.transformer
        self.problem_type_ = self.transformer.problem_type_
        return self

    def transform(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, skip_check_array=True, reset=False)
        ret = self.fitted_transformer_.transform_X(X)
        return ret

    def validate_data(self, X):
        validate_data(self, X=X, skip_check_array=True, reset=False)

    def transform_X(self, X):
        return self.transform(X)

    def transform_X_y(self, X, y=None):
        check_is_fitted(self)
        return self.fitted_transformer_.transform(X, y)

    def transform_y(self, df):
        check_is_fitted(self)
        return self.fitted_transformer_.transform_y(df)

    def inverse_transform_y(self, y):
        return self.fitted_transformer_.inverse_transform_y(y)

class AutoEncoder:
    """Encodes input polars dataframe to a common format:
    - All categorical columns are explicitly converted to categorical datatype.
    - Constant columns are dropped.
    - Columns with 2 unique values are converted to boolean format.
    - The target is encoded ordinally for classification, or casted to Float64 for regression.

    Also infers problem type if it is not provided, splits to X and y, provides inverse y transform.
    """

    logger = logging.getLogger("myautoml_AutoMLEncoder")

    def __init__(
        self,
        convert_categorical: bool = True,
        drop_constant: bool = True,
        binary_to_bool: bool = True,
        encode_target: bool = True,
        drop_cols: str | Sequence[str] | None = None,
    ):
        self.convert_categorical = convert_categorical
        self.drop_constant = drop_constant
        self.binary_to_bool = binary_to_bool
        self.encode_target = encode_target
        self.drop_cols = drop_cols

    @pl.StringCache()
    def fit(
        self,
        X: pl.DataFrame | np.ndarray | Any,
        y: str | pl.Series | pl.DataFrame | np.ndarray | Any | None,
        X_unlabeled: pl.DataFrame | np.ndarray | Any | None = None,
        problem_type: "ProblemType | None" = None,
    ):
        drop_cols = self.drop_cols
        if drop_cols is None: drop_cols = ()
        if isinstance(drop_cols, str): drop_cols = (drop_cols, )

        # Convert stuff to polars
        label = None

        if isinstance(y, str):
            label = y
            y = X[y]
            X = to_dataframe(X).drop(label)

        else:
            X = to_dataframe(X)

        X_all = X
        if X_unlabeled is not None:
            X_unlabeled = to_dataframe(X_unlabeled)
            X_all = pl.concat([X, X_unlabeled])

        if y is not None:
            y = to_series(y)
            if label is None:
                label = y.name

        self.label_ = label
        self.feature_names_in_ = X.schema.names()

        self.logger.info("Loaded X dataframe with %i rows and %i columns", X.shape[0], X.shape[1])
        if X_unlabeled is not None:
            self.logger.info(
                "Loaded unlabeled dataframe with %i rows and %i columns", X_unlabeled.shape[0], X_unlabeled.shape[1])

        # Process
        stages = []
        if len(drop_cols) > 0: stages.append(DropCols(*drop_cols))
        if self.binary_to_bool: stages.append(BinaryToBool(exclude=self.label_))


        # warn on categorical cols with only unique values
        for col in X_all.select(pl.selectors.string(include_categorical=True)):
            if (col.name == self.label_) or (col.name in drop_cols): continue
            if col.n_unique() == len(col):
                self.logger.info("Column %s is string/categorical and has no repeating values.", col.name)
                # drop_cols.append(col.name)

        # Drop constant cols
        if self.drop_constant:
            for col in X_all:
                if (col.name == self.label_) or (col.name in drop_cols): continue
                if col.n_unique() == 1:
                    self.logger.warning("Column %s is constant, it will be dropped.", col.name)

            stages.append(DropConstant(exclude=self.label_))

        # this should be at the end, after text cols has been dropped
        stages.append(Cast(pl.selectors.string(), pl.Categorical, exclude=self.label_))

        self.X_chain_ = Chain(stages)
        self.X_chain_.fit(X_all)

        # encode target
        if (problem_type is None) and (y is not None):
            problem_type, msg = _infer_problem_type(y)
            self.logger.warning("Problem type inferred as %r, reason: %s. If this is incorrect, "
                                "pass correct problem_type to TabularPredictor", problem_type, msg)

        self.problem_type_ = problem_type

        self.y_encoder_ = None
        self.y_schema_ = None
        if self.encode_target:
            if problem_type in ('binary', 'multiclass'):
                self.y_encoder_ = OrdinalEncoder(self.label_).fit(y)

            elif problem_type == "regression":
                self.y_encoder_ = Chain(Cast(self.label_, pl.Float64)).fit(y)

        if self.y_encoder_ is not None:
            self.y_schema_ = self.y_encoder_.transform(y).collect_schema().copy()

        return self

    @pl.StringCache()
    def transform(self, X, y=None) -> tuple[pl.DataFrame, pl.Series]:
        X = to_dataframe(self.X_chain_.transform(to_dataframe(X)))

        if y is None or y == self.label_:
            if (self.label_ is not None) and (self.label_ in X.columns):
                y = X[self.label_]
                X = X.drop(self.label_)

        if y is not None:
            y = self.transform_y(y)

        return X, cast(pl.Series | Any, y)

    def fit_transform(
        self,
        X: pl.DataFrame | np.ndarray | Any,
        y: str | pl.Series | pl.DataFrame | np.ndarray | Any | None,
        X_unlabeled: pl.DataFrame | np.ndarray | Any | None = None,
        problem_type: "ProblemType | None" = None,
    ) -> tuple[pl.DataFrame, pl.Series]:
        self.fit(X=X, y=y, X_unlabeled=X_unlabeled, problem_type=problem_type)
        if isinstance(y, str): y = None
        return self.transform(X, y)

    @pl.StringCache()
    def transform_X(self, X) -> pl.DataFrame:
        X = to_dataframe(self.X_chain_.transform(to_dataframe(X)))
        if (self.label_ is not None) and (self.label_ in X.columns): X = X.drop(self.label_)
        return X

    def transform_y(self, y) -> pl.Series:
        if self.y_encoder_ is not None: y = to_series(self.y_encoder_.transform(to_series(y)))
        if self.label_ is not None: y = y.alias(self.label_)
        return y

    def inverse_transform_y(self, y) -> pl.Series:

        if y.__class__.__name__.lower() not in ("dataframe", "series"):
            y = np.asarray(y)
            if y.ndim == 1: y = np.expand_dims(y, 1)
            y = pl.from_numpy(y, schema=self.y_schema_)

        else:
            y = to_series(y)
            if self.label_ is not None: y = y.alias(self.label_)

        if self.y_encoder_ is not None and hasattr(self.y_encoder_, "inverse_transform"):
            y = getattr(self.y_encoder_, "inverse_transform")(y)

        return to_series(y)

    def to_sklearn(self):
        return _AutoEncoderWrapper(self)

    def to_frozen(self) -> _AutoEncoderWrapper:
        est = self.to_sklearn().set_fitted()
        feature_names_in = self.feature_names_in_.copy()
        if self.label_ is not None and self.label_ in feature_names_in: feature_names_in.remove(self.label_)
        est.feature_names_in_ = feature_names_in
        return cast(_AutoEncoderWrapper, FrozenEstimator(est))

