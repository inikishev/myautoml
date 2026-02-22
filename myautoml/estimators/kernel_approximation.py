import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)


class LaplaceRFF(TransformerMixin, BaseEstimator):
    """Random fourier features laplace kernel estimator.

    Args:
        gamma: Gamma parameter, should be a positive float, defaults to ``1/n_features``.
        n_components: Number of features to construct. Defaults to 1000.
        dtype: dtype (float32 with larger n components is usually better)
        random_state: seed. Defaults to None.
    """
    def __init__(self, gamma: float | None = None, n_components=1000, dtype=np.float32, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.dtype = dtype
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)
        X = validate_data(self, X=X)

        n_features = X.shape[1]
        gamma = self.gamma
        if gamma is None: gamma = 1/n_features
        self.gamma_ = gamma

        self.W_ = rng.standard_cauchy((n_features, self.n_components)).astype(self.dtype) * self.gamma_
        self.b_ = rng.uniform(0, 2 * np.pi, self.n_components).astype(self.dtype)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        X = X.astype(self.dtype)
        projection = X @ self.W_ + self.b_
        return np.cos(projection).astype(self.dtype) * np.sqrt(2.0 / self.n_components)