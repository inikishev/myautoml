"""Basic datasets mainly for tests"""
import polars as pl
import numpy as np
from ..utils.rng import RNG

def _add_missing_values(A: np.ndarray, p: float, rng):
    rng = RNG(rng)
    A = A.copy()
    if p<=0: return A
    mask = np.zeros(A.size, dtype=bool)
    mask[:int(A.size * p)] = True
    rng.numpy.shuffle(mask)
    A_flat = A.ravel()
    A_flat[mask] = np.nan
    return A_flat.reshape(A.shape)

def _make_targets(X: np.ndarray, n_targets: int, n_classes: int | None, act, rng: RNG):
    _, n_features = X.shape
    if n_classes is None:
        W = rng.numpy.standard_normal((n_features, n_targets))
        y = X @ W
        if act is not None: y = act(y)
        return y

    else:
        W = rng.numpy.standard_normal((n_targets, n_features, n_classes))
        y_logits = X @ W
        if act is not None: y_logits = act(y_logits)
        return y_logits.argmax(-1).T

def _make_colinear(X: np.ndarray, rng: RNG):
    _, n_features = X.shape
    corrs = rng.numpy.standard_normal(n_features-1)
    biases = rng.numpy.standard_normal(n_features-1)

    X_i = X[:, 0]
    Xs = [X_i]
    for i,(corr,bias) in enumerate(zip(corrs, biases)):
        X_i = X_i + X[:, i+1] * corr + bias
        Xs.append(X_i)

    return np.stack(Xs, 1)

def get_linear(
    n_samples=1000,
    n_features=10,
    n_targets=1,
    n_classes=None,
    act=None,
    colinear: bool = False,
    missing_p: float = 0,
    noise: float = 0,
    seed: int | None = 0,
):
    """Linear dataset with normally distributed or colinear samples.

    Args:
        n_samples: n samples. Defaults to 1000.
        n_features: n features. Defaults to 10.
        n_targets: n targets. Defaults to 1.
        n_classes: if None, uses regression. Defaults to None.
        act: nonlinearity to apply to targets. Defaults to None.
        colinear: whether to make dataset highly multicolinear. Defaults to False.
        missing_p: probability that a feature is missing. Defaults to 0.
        noise: sigma of noise. Defaults to 0.
        seed: random seed. Defaults to 0.
    """
    rng = RNG(seed)
    X = rng.numpy.standard_normal((n_samples, n_features))
    y = _make_targets(X, n_targets=n_targets, n_classes=n_classes, act=act, rng=rng)
    if colinear: X = _make_colinear(X, rng)
    X = X + rng.numpy.normal(0, scale=noise, size=(n_samples, n_features))
    X = _add_missing_values(X, missing_p, rng)
    return X, y


def get_1cat(
    n_samples=1000,
    n_categories=10,
    n_features=10,
    n_targets=1,
    n_classes=None,
    act=None,
    colinear: bool = False,
    missing_p: float = 0,
    noise: float = 0,
    seed=0,
):
    """A dataset with one categorical feature which determines the linear model.

    Args:
        n_samples: n samples. Defaults to 1000.
        n_categories: number of categories in the categorical variable. Defaults to 10.
        n_features: n features. Defaults to 10.
        n_targets: n targets. Defaults to 1.
        n_classes: if None, uses regression. Defaults to None.
        act: nonlinearity to apply to targets. Defaults to None.
        colinear: whether to make dataset highly multicolinear. Defaults to False.
        missing_p: probability that a feature is missing. Defaults to 0.
        noise: sigma of noise. Defaults to 0.
        seed: random seed. Defaults to 0.
    """
    rng = RNG(seed)
    X_cat = np.random.randint(0, n_categories, size=n_samples)

    n_features = n_features - 1
    X_num = np.empty((n_samples, n_features))
    y = None
    for i in range(n_categories):
        X_i = rng.numpy.standard_normal((n_samples, n_features))
        y_i = _make_targets(X_i, n_targets=n_targets, n_classes=n_classes, act=act, rng=rng)
        if colinear: X_i = _make_colinear(X_i, rng)

        mask = X_cat == i
        X_num[mask] = X_i[mask]

        if y is None: y = y_i
        else: y[mask] = y_i[mask]

    assert y is not None

    X_num = X_num + rng.numpy.standard_normal((n_samples, n_features)) * noise
    X_cat = X_cat[:, np.newaxis]

    X_num = _add_missing_values(X_num, missing_p, rng)
    X_cat = _add_missing_values(X_cat, missing_p, rng)

    return pl.concat([
        pl.from_numpy(X_cat, schema=[f"Xcat_{i}" for i in range(X_cat.shape[1])]).cast(pl.String).cast(pl.Categorical),
        pl.from_numpy(X_num, schema=[f"Xnum_{i}" for i in range(X_num.shape[1])]),
    ], how='horizontal'), pl.from_numpy(y, schema=[f"y_{i}" for i in range(y.shape[1])])
