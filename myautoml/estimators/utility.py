import polars as pl
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)


from ..utils.torch_utils import to_numpy

class ToDtype(TransformerMixin, BaseEstimator):
    """Converts input to ``np.ndarray`` with specified dtype, None (default) to just convert to ``np.ndarray``."""
    def __init__(self, dtype: np.typing.DTypeLike | None = None):
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
