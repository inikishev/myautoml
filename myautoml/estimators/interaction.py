from collections.abc import Callable
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)


def _apply_eps(x: np.ndarray, eps: float):
    return np.where(np.abs(x) < eps, np.sign(x) * eps, x)

def interaction_min_neg(x, y):
    return np.maximum(x, -y)

def interaction_max_neg(x, y):
    return np.maximum(x, -y)

def interaction_exp(x, y):
    return np.abs(x) ** y

def interaction_log(x, y, eps=1e-5):
    return np.log(_apply_eps(x, eps)) / np.log(_apply_eps(y, eps))

def interaction_divide(x, y, eps=1e-5):
    return x / _apply_eps(y, eps)

def interaction_copysign(x, y):
    return np.copysign(x, y)

# also can pass minimum, maximum, softmax, logsumexp arctan2

class InteractionFeatures(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        commutative: bool,
        interaction_only: bool,
        order: int = 2,
    ):
        self.fn = fn
        self.commutative = commutative
        self.interaction_only = interaction_only
        self.order = order

    def fit(self, X, y=None):
        validate_data(self, X=X)
        self.fitted_ = True
        return self

    def transform(self, X):
        X = validate_data(self, X=X, reset=False)

        X_i = X

        for i in range(self.order - 2):
            X_i = self.fn(X[:, np.newaxis, :], X_i[:, :, np.newaxis])

            if self.commutative and i == 0:
                k = 0 if self.interaction_only else 1
                r, c = np.triu_indices(X.shape[1], k)
                X_i = X_i[:, r, c]

            X_i = X_i.reshape(X_i.shape[0], -1)

        return X_i

