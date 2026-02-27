import polars as pl
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)


from ..utils.torch_utils import to_numpy
from ..utils.polars_utils import to_dataframe


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
