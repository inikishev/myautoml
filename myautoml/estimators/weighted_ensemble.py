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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from sklearn.utils import check_random_state

from ..metrics.scoring import get_scorer
from ..utils.polars_utils import to_dataframe, to_series
from ..utils.numpy_utils import one_hot


def _get_individual_preds(X, n_classes:int | None) -> tuple[dict[str, np.ndarray], defaultdict[str, list[int]]]:
    X = to_dataframe(X)

    models = set()
    indexes = defaultdict(list)

    # get all models
    for col in sorted(set(X.columns)):
        if "-" not in col: raise RuntimeError(f"Column {col} has incorrect name, should be f'{{model}}_{{index}}'")
        model, index = col.rsplit("-", 1)
        models.add(model)
        indexes[model].append(int(index))

    # split into preds
    preds = {}
    for model in sorted(models):
        model_cols = [f"{model}-{col}" for col in indexes[model]]

        if n_classes is None:
            preds[model] = X.select(model_cols).to_numpy()

        else:
            # TabularFitter will use predict if predict_proba is None
            arr = X.select(model_cols).to_numpy()
            assert arr.ndim == 2
            if arr.shape[1] != n_classes:
                assert arr.shape[1] == 1, arr.shape
                arr = arr[:, 0]
                if n_classes == 2: # TabularFitter only keeps positive probability
                    arr = np.stack([1-arr, arr], -1)
                else:
                    assert np.allclose((arr % 1).sum(), 0)
                    arr = one_hot(arr.astype(np.uint16), n_classes)
            preds[model] = arr

    return preds, indexes

def _make_int(i: int | float, l: int):
    if i == 0: return 0
    if isinstance(i, int):
        assert i >= 1
        return i

    if isinstance(i, float):
        assert 0 < i <= 1
        return math.ceil(i * l)

    raise TypeError(type(i))

class _BaseGreedyWeightedEnsemble(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        scoring,
        n_bags: int = 20,
        p: int | float = 0.5,
        n_init: int | float | None = 5,
        max_iter: int = 1_000,
        max_no_improvement: int = 3,
        subsample: int | float | None = 1_000_000,
        max_sec: float | None = None,
        random_state=0,
        verbose: int = 0
    ):
        self.scoring = scoring
        self.n_bags = n_bags
        self.p = p
        self.n_init = n_init
        self.random_state = random_state

        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.subsample = subsample
        self.max_sec = max_sec
        self.verbose = verbose

    def fit(self, X, y):
        validate_data(self, X=X, y=y, ensure_all_finite=False)
        random_state = check_random_state(self.random_state)

        if isinstance(y, (pl.Series, pl.DataFrame)):
            y = y.to_numpy()
        elif hasattr(y, "values"):
            y = getattr(y, "values")
        else:
            y = np.asarray(y)

        if self.is_classification: self.n_classes_ = len(set(y))
        else: self.n_classes_ = None

        preds_dict, indexes = _get_individual_preds(X, self.n_classes_)
        if len(preds_dict) <= 1:
            raise RuntimeError(f"At least two models are required for greedy weighted ensemble, got {list(preds_dict.keys())}")

        preds_np = np.stack(list(preds_dict.values()), 0) # (n_models, n_rows, 1) or (n_models, n_rows, n_classes)
        names = np.asarray(list(preds_dict.keys()), dtype=np.str_)
        del preds_dict


        if self.is_classification:
            if preds_np.shape[-1] == 1:
                # Handle binary classification case, where only positive label probabilities are provided
                if np.min(preds_np) < 0 or np.max(preds_np) > 1:
                    raise RuntimeError("test_pred must contain probabilities, but "
                                       f"{np.min(preds_np) = }, {np.max(preds_np) = }.")
                preds_np = np.concatenate([1-preds_np, preds_np], -1)


        scorer = get_scorer(self.scoring)

        if self.is_classification:
            init_errors = np.asarray([scorer.error(y, preds=np.argmax(y_hat, -1), proba=y_hat) for y_hat in preds_np])
        else:
            init_errors = np.asarray([scorer.error(y, preds=y_hat, proba=None) for y_hat in preds_np])
        assert init_errors.ndim == 1

        weights = np.zeros(len(names), dtype=int)

        for bag_i in range(self.n_bags):

            # Subsample rows
            y_rows = y
            preds_rows = preds_np
            if self.subsample is not None:
                subsample = _make_int(self.subsample, len(y))
                if subsample < len(y):
                    indices = random_state.choice(len(y), size=subsample, replace=False)
                    y_rows = y[indices]
                    preds_rows = preds_np[:, indices]

            # Subsample models
            p = _make_int(self.p, len(names))
            sub_idx = random_state.choice(len(names), size=p, replace=False)
            sub_preds = preds_rows[sub_idx] # subsampled rows and models
            sub_errors = init_errors[sub_idx]

            # Initialize with best-performing models
            bag_weights = np.zeros_like(sub_idx)
            bag_sum = np.zeros_like(sub_preds[0])
            n_models = 0

            if self.n_init is not None:
                n_init = _make_int(self.n_init, len(names))
                if n_init > 0:
                    bag_idx = [i for i,error in sorted(enumerate(sub_errors), key=lambda x: x[1])][:n_init]
                    bag_sum = sub_preds[bag_idx].sum(0)
                    bag_weights[bag_idx] = 1
                    n_models = n_init

            # Hillclimbing
            trial_preds = np.zeros_like(sub_preds[0]) # pre-allocate
            best_weights = bag_weights.copy()
            best_weights_error = float("inf")
            num_no_improvement = 0
            start_time = time.time()

            for iteration in range(self.max_iter):

                # Try each model and pick one that improves ensemble the most
                lowest_error = float("inf")
                lowest_error_index = None

                for i, model_preds in enumerate(sub_preds):

                    trial_preds[:] = (bag_sum + model_preds) / (n_models + 1)

                    if self.is_classification:
                        trial_error = scorer.error(y_rows, preds=np.argmax(trial_preds, -1), proba=trial_preds)
                    else:
                        trial_error = scorer.error(y_rows, preds=trial_preds, proba=None)

                    if trial_error < lowest_error:
                        lowest_error = trial_error
                        lowest_error_index = i

                    if self.verbose >= 2:
                        print(f"{i} {names[sub_idx[i]]}: {trial_error=:5f}, {lowest_error=:5f}")

                # Update bag with new model
                assert lowest_error_index is not None
                bag_sum += sub_preds[lowest_error_index]
                bag_weights[lowest_error_index] += 1
                n_models += 1

                if lowest_error < best_weights_error:
                    best_weights_error = lowest_error
                    best_weights = bag_weights.copy()
                    num_no_improvement = 0

                else:
                    num_no_improvement += 1
                    if num_no_improvement >= self.max_no_improvement:
                        break

                if (self.max_sec is not None) and (time.time() - start_time >= self.max_sec / self.n_bags):
                    break

                if self.verbose >= 1:
                    print(f"{bag_i}-{iteration}/{self.max_iter}: {best_weights_error=:.5f}, "
                          f"{lowest_error=:.5f}, {n_models=}, no_improvement="
                          f"{num_no_improvement}/{self.max_no_improvement}")

                if iteration == self.max_iter - 1:
                    warnings.warn(f"{self.__class__.__name__} terminated early, reached {self.max_iter} iterations")

            # Update weights
            weights[sub_idx] += best_weights

        # Normalize and store weights
        self.weights_ = {model: w for model, w  in zip(names, weights / weights.sum()) if w > 0}
        self.required_cols_ = set(f"{model}-{i}" for model in self.weights_.keys() for i in indexes[model])
        return self

    def __myautoml_used_estimators__(self):
        return list(self.weights_.keys())

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




class GreedyWeightedEnsembleClassifier(ClassifierMixin, _BaseGreedyWeightedEnsemble):
    """Implements https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    Meant to be used with ``TabularFitter``.
    This should only be applied to X which contains predictions (ideally out-of-fold) of other models.
    The predictions should be of correct type (predict or predict_proba) for specified scoring.

    X must have format ``f"{model_name}-{output_i}"``.

    Note: this may go on until ``max_iter`` when total number of models is too low, or n_init is too high,
        or there is a single model with significantly better scores than all other models. This is because
        optimal weights are (1, 0, 0, ...); and if it is initialized to more than 1 top model, it will keep
        picking the best model until max_iter is reached, trying to bring weights closer to optimal.
        If that happens, set ``n_init`` to 1.

    Args:
        scoring: scoring method
        n_bags: number of bags. Defaults to 20.
        p: number/fraction of models in each bag. Defaults to 0.5.
        n_init: number/fraction of best-performing models to initialize each bag with. Defaults to 5.
        max_iter: maximum number of iterations per bag. Defaults to 1_000.
        max_no_improvement: maximum number of hill-climbing without improvement. Defaults to 3.
        subsample: number/fraction of rows to subsample in each bag, can make this much faster. Defaults to 1_000_000.
        max_sec: each bag will fit for no more than ``max_sec / n_bags``
    """
    is_classification = True
    def __init__(
        self,
        scoring,
        n_bags: int = 20,
        p: int | float = 0.5,
        n_init: int | float | None = 5,
        max_iter: int = 1_000,
        max_no_improvement: int = 3,
        subsample: int | float | None = 1_000_000,
        max_sec: float | None = None,
        random_state=0,
        verbose: int = 0
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict_proba(self, X):
        return self._predict_raw(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(-1)


class GreedyWeightedEnsembleRegressor(RegressorMixin, _BaseGreedyWeightedEnsemble):
    """Implements https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    Meant to be used with ``TabularFitter``.
    This should only be applied to X which contains predictions (ideally out-of-fold) of other models.
    The predictions should be of correct type (predict or predict_proba) for specified scoring.

    X must have format ``f"{model_name}-{output_i}"``.

    Note: this may go on until ``max_iter`` when total number of models is too low, or n_init is too high,
        or there is a single model with significantly better scores than all other models. This is because
        optimal weights are (1, 0, 0, ...); and if it is initialized to more than 1 top model, it will keep
        picking the best model until max_iter is reached, trying to bring weights closer to optimal.
        If that happens, set ``n_init`` to 1.

    Args:
        scoring: scoring method
        n_bags: number of bags. Defaults to 20.
        p: number/fraction of models in each bag. Defaults to 0.5.
        n_init: number/fraction of best-performing models to initialize each bag with. Defaults to 5.
        max_iter: maximum number of iterations per bag. Defaults to 1_000.
        max_no_improvement: maximum number of hill-climbing without improvement. Defaults to 3.
        subsample: number/fraction of rows to subsample in each bag, can make this much faster. Defaults to 1_000_000.
        max_sec: each bag will fit for no more than ``max_sec / n_bags``
    """
    is_classification = False

    def __init__(
        self,
        scoring,
        n_bags: int = 20,
        p: int | float = 0.5,
        n_init: int | float | None = 5,
        max_iter: int = 1_000,
        max_no_improvement: int = 3,
        subsample: int | float | None = 1_000_000,
        max_sec: float | None = None,
        random_state=0,
        verbose: int = 0
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict(self, X):
        return self._predict_raw(X)
