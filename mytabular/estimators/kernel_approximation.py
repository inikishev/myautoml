import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)


def _ortho_qr(X: np.ndarray, random_state):
    m,n = X.shape
    transpose = m < n
    if transpose: X = X.T
    X = np.linalg.qr(X).Q
    if transpose: X = X.T
    return X

class Project(TransformerMixin, BaseEstimator):
    """Multiply by ``(n_features, k)`` orthonormal matrix and add random bias.

    Args:
        n_components: number of output features.
        bias: whether to add bias (length-k vector) to output.
        random_state: seed. Defaults to None.
    """
    def __init__(self, n_components: int, bias=True, random_state=None):
        self.n_components = n_components
        self.bias = bias
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)
        X = validate_data(self, X=X)

        n_features = X.shape[1]

        self.W_ = _ortho_qr(rng.standard_normal((n_features, self.n_components)), rng)
        if self.bias:
            self.b_ = rng.standard_normal(self.n_components)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        X = X @ self.W_
        if self.bias: X = X = self.b_
        return X


class LaplaceRFF(TransformerMixin, BaseEstimator):
    """Random fourier features laplace kernel estimator.

    Args:
        gamma: Gamma parameter, should be a positive float, defaults to ``1/n_features``.
        n_components: Number of features to construct. Defaults to 1000.
        random_state: seed. Defaults to None.
    """
    def __init__(self, gamma: float | None = None, n_components=1000, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)
        X = validate_data(self, X=X)

        n_features = X.shape[1]
        gamma = self.gamma
        if gamma is None: gamma = 1/n_features
        self.gamma_ = gamma

        self.W_ = rng.standard_cauchy((n_features, self.n_components)).astype(X.dtype) * self.gamma_
        self.b_ = rng.uniform(0, 2 * np.pi, self.n_components).astype(X.dtype)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        return np.cos(X @ self.W_ + self.b_) * np.sqrt(2.0 / self.n_components)