import time
import logging, warnings
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from sklearn.utils import check_random_state

from ..metrics.scoring import get_scorer
from ..utils.polars_utils import to_dataframe, to_series
from ..utils.numpy_utils import one_hot

from .weighted_ensemble import _get_individual_preds, _make_int

class _BaseHillClimbingEnsemble(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        scoring,
        search_iters: int,
        bracket,
        bag_iters: int,
        min_magnitude: float,
        eps: float,
        verbose: int,
    ):
        self.scoring = scoring

        self.search_iters = search_iters
        self.bracket = bracket
        self.bag_iters = bag_iters
        self.eps = eps
        self.min_magnitude = min_magnitude
        self.verbose = verbose

    def fit(self, X, y):
        from scipy.optimize import minimize_scalar
        validate_data(self, X=X, y=y, ensure_all_finite=False)

        if isinstance(y, (pl.Series, pl.DataFrame)):
            y = y.to_numpy()
        elif hasattr(y, "values"):
            y = getattr(y, "values")
        else:
            y = np.asarray(y)

        if self.is_classification: self.n_classes_ = len(set(y))
        else: self.n_classes_ = None

        preds_dict, col_indexes = _get_individual_preds(X, self.n_classes_)
        if len(preds_dict) <= 1:
            raise RuntimeError(f"At least two models are required for greedy weighted ensemble, got {list(preds_dict.keys())}")

        scorer = get_scorer(self.scoring)
        if self.is_classification:
            init_errors = np.array(
                [scorer.error(y, preds=np.argmax(y_hat, -1), proba=y_hat) for name, y_hat in preds_dict.items()])
        else:
            init_errors = np.array(
                [scorer.error(y, preds=np.squeeze(y_hat), proba=None) for name, y_hat in preds_dict.items()])

        sorted_names = [str(n) for n in np.array(list(preds_dict.keys()))[np.argsort(init_errors)]]

        weights = defaultdict(lambda: 0.0)
        weights[sorted_names[0]] = 1

        weights_sum: float = 1.0
        bag_sum = preds_dict[sorted_names[0]].copy()

        def compute_error(preds: np.ndarray):
            if self.is_classification:
                return scorer.error(targets=y, preds=preds.argmax(-1), proba=preds)
            return scorer.error(targets=y, preds=np.squeeze(preds), proba=None)

        lowest_error = compute_error(bag_sum)
        if self.verbose >= 1: print(f"Initialized to {sorted_names[0]}, {lowest_error=:.8f}")

        new_preds = np.zeros_like(bag_sum) # pre-allocate

        for bag_iter in range(self.bag_iters):
            for estimator in sorted_names[1:]:
                estimator_preds = preds_dict[estimator]

                def barrier_fn(x: float):
                    if x < weights_sum - self.eps: # make sure weights_sum is strictly positive
                        diff = (weights_sum - self.eps) - x
                        x = weights_sum - (self.eps / diff)
                    return x

                def objective(x: float):
                    x = barrier_fn(x)
                    new_preds[:] = (bag_sum + estimator_preds * x) / (weights_sum + x)
                    if self.is_classification and new_preds.min() < 0: error = max(-x, 1) * 1e5
                    else: error = compute_error(new_preds)
                    if self.verbose >= 2: print(f"{bag_iter} {estimator} {x}: {error=:.8f}, {lowest_error=:.8f}")
                    return error

                res: Any = minimize_scalar(objective, self.bracket, options={"maxiter": self.search_iters})

                x = barrier_fn(res.x)
                if res.fun < lowest_error and abs(x) > self.min_magnitude:
                    lowest_error = res.fun
                    weights[estimator] += x
                    bag_sum = bag_sum + estimator_preds * x
                    weights_sum += x

                if self.verbose >= 1: print(f"{bag_iter} {estimator} {x}: error={res.fun:.8f}, {lowest_error=:.8f}")


        # Normalize and store weights
        if self.verbose >= 1:
            print(f"final weights: {weights}")
            print(f'{sum(weights.values()) = }, {weights_sum = }')

        self.weights_ = {model: w/weights_sum for model, w in weights.items()}
        self.required_cols_ = set(f"{model}-{i}" for model in self.weights_.keys() for i in col_indexes[model])
        return self

    def __myautoml_used_estimators__(self):
        return list(self.weights_.keys())

    def transform(self, X):
        check_is_fitted(self)

        X = to_dataframe(X)

        if not set(X.columns).issuperset(self.required_cols_):
            missing = self.required_cols_.difference(X.columns)
            raise RuntimeError(f"X is missing the following columns: {missing}")

        # X will only have estimators from ``__myautoml_used_estimators__``
        return X

    def _predict_raw(self, X):
        check_is_fitted(self)

        # can't use validate_data(self, X=X, reset=False, ensure_all_finite=False)
        # X might not have some columns seen during fit as they have a weight of 0
        X = to_dataframe(X)
        if not set(X.columns).issuperset(self.required_cols_):
            missing = self.required_cols_.difference(X.columns)
            raise RuntimeError(f"X is missing the following columns: {missing}")


        preds, _ = _get_individual_preds(X, self.n_classes_)

        ensemble_preds = np.zeros_like(next(iter(preds.values())))
        for k, w in self.weights_.items():
            ensemble_preds += preds[k] * w

        return ensemble_preds


class HillClimbingEnsembleClassifier(ClassifierMixin, _BaseHillClimbingEnsemble):
    is_classification = True
    def __init__(
        self,
        scoring,
        search_iters: int = 32,
        bracket = (-0.3, 1.5),
        bag_iters: int = 1,
        min_magnitude: float = 0.001,
        eps: float = 0.1,
        verbose: int = 0,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict_proba(self, X):
        return self._predict_raw(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(-1)


class HillClimbingEnsembleRegressor(RegressorMixin, _BaseHillClimbingEnsemble):
    is_classification = False

    def __init__(
        self,
        scoring,
        search_iters: int = 16,
        bracket = (-0.3, 1.5),
        bag_iters: int = 1,
        min_magnitude: float = 0.001,
        eps: float = 0.1,
        verbose: int = 0,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict(self, X):
        return self._predict_raw(X)

class HillClimbingEnsembleSelector(TransformerMixin, _BaseHillClimbingEnsemble):
    is_classification = False

    def __init__(
        self,
        scoring,
        search_iters: int = 16,
        bracket = (-0.3, 1.5),
        bag_iters: int = 1,
        min_magnitude: float = 0.001,
        eps: float = 0.1,
        verbose: int = 0,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    # inherits transform
