from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..utils.torch_utils import TORCH_INSTALLED


class UnsupervisedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self.estimator.fit(X, y)

    def transform(self, X):
        return self.estimator.predict(X).reshape(-1, 1)
