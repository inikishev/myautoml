import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

def _get_hoc_features(x):
    """
    x: (B, L) tensor
    returns: (B, F) tensor of Higher-Order Correlation features
    """
    B, L = x.shape
    dtype = x.dtype
    x = x.astype(np.float64)

    X = np.fft.rfft(x, n=L) # (B, L//2 + 1)
    X_len = X.shape[1]

    def get_X(k):
        # k is from 0 to L-1
        k = k % L
        if k < X_len:
            return X[:, k]
        else:
            return np.conj(X[:, L - k])

    # First order: Mean
    mean = np.expand_dims(X[:, 0].real, 1) / L

    # Second order: Power Spectrum
    power_spectrum = np.abs(X)**2 # (B, L//2 + 1)

    # Third order: Bispectrum
    bispectrum_list = []

    for k1 in range(X_len):
        for k2 in range(k1 + 1):
            # B(k1, k2) = X(k1) * X(k2) * conj(X(k1+k2))
            k3 = (k1 + k2) % L
            b = X[:, k1] * X[:, k2] * np.conj(get_X(k3))

            bispectrum_list.append(np.expand_dims(b.real, 1))
            bispectrum_list.append(np.expand_dims(b.imag, 1))

    features = np.concatenate([mean, power_spectrum, *bispectrum_list], axis=1)
    return features.astype(dtype)

class HOCFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        validate_data(self, X=X, y=y)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        return _get_hoc_features(X)