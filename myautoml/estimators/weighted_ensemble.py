import time
import logging
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from sklearn.utils import check_random_state

from ..metrics.scoring import get_scorer
from ..utils.polars_utils import to_dataframe, to_series


def _get_individual_preds(X) -> tuple[dict[str, np.ndarray], int]:
    X = to_dataframe(X)

    models = set()
    indexes = defaultdict(list)

    # get all models
    for col in sorted(set(X.columns)):
        model, index = col.rsplit("_", 1)
        models.add(model)
        indexes[model].append(int(index))

    # verify that all models have same number of columns
    first_ind = None
    for model,ind in indexes.items():
        assert len(ind) == len(set(ind))
        if first_ind is None: first_ind = set(ind)
        if set(ind) != first_ind:
            raise RuntimeError(f"Model {model} has indexes {sorted(ind)}, while another model had {sorted(first_ind)}.")

    # split into preds
    preds = {}
    assert first_ind is not None
    required_cols = sorted(first_ind)
    for model in sorted(models):
        model_cols = [f"{model}_{col}" for col in required_cols]
        preds[model] = X.select(model_cols).to_numpy()

    return preds, len(first_ind)

def _make_int(i: int | float, l: int):
    if isinstance(i, int):
        assert i >= 1
        return i

    if isinstance(i, float):
        assert 0 < i < 1
        return math.ceil(i * l)

    raise TypeError(type(i))

class GreedyWeightedEnsembleRegressor(TransformerMixin, BaseEstimator):
    """Implements https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    Meant to be used with ``TabularFitter``.
    This should only be applied to X which contains predictions (ideally out-of-fold) of other models.
    The predictions should be of correct type (predict or predict_proba) for specified scoring.

    X must have format ``f"{model_name}_{output_i}"``.

    Args:
        scoring: scoring method
        n_bags: number of bags. Set this to 1 to mimic autogluon and speed this up significantly. Defaults to 20.
        p: number/fraction of models in each bag. Defaults to 0.5.
        n_init: number/fraction of best-performing models to initialize each bag with. Defaults to 0.1.
        max_iter: maximum number of iterations per bag. Defaults to 1_000_000.
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
        n_init: int | float | None = 0.1,
        max_iter: int = 1_000_000,
        max_no_improvement: int = 3,
        subsample: int | float | None = 1_000_000,
        max_sec: float | None = None,
        random_state=0,
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

    def fit(self, X, y):
        validate_data(self, X=X, y=y, ensure_all_finite=False)
        random_state = check_random_state(self.random_state)

        if isinstance(y, (pl.Series, pl.DataFrame)):
            y = y.to_numpy()
        elif hasattr(y, "values"):
            y = getattr(y, "values")
        else:
            y = np.asarray(y)

        preds_dict, self.n_out_ = _get_individual_preds(X)
        test_pred = next(iter(preds_dict.values()))
        if self.is_classification:
            if np.squeeze(test_pred).ndim == 1: raise RuntimeError("X must contain probabilites predicted by each model.")

        preds_np = np.stack(list(preds_dict.values()), 0) # (n_models, n_rows, *pred_dims)
        names = np.asarray(list(preds_dict.keys()), dtype=np.str_)

        scorer = get_scorer(self.scoring)

        if self.is_classification:
            init_errors = np.asarray([scorer.error(y, preds=np.argmax(y_hat, -1), proba=y_hat) for y_hat in preds_np])
        else:
            init_errors = np.asarray([scorer.error(y, preds=y_hat, proba=None) for y_hat in preds_np])
        assert init_errors.ndim == 1

        weights = np.zeros(len(names), dtype=int)

        for i_bag in range(self.n_bags):

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

            # Update weights
            weights[sub_idx] += best_weights

        # Normalize and store weights
        self.weights_ = {model: w for model, w  in zip(names, weights / weights.sum()) if w > 0}
        self.required_cols_ = set(f"{model}_{i}" for model in self.weights_.keys() for i in range(self.n_out_))
        return self

    def __myautoml_used_models__(self):
        return list(self.weights_.keys())

    def predict(self, X):
        check_is_fitted(self)

        # can't use validate_data(self, X=X, reset=False, ensure_all_finite=False)
        # X might not have some columns seen during fit as they have a weight of 0
        X = to_dataframe(X)
        if not set(X.columns).issuperset(self.required_cols_):
            missing = self.required_cols_.difference(X.columns)
            raise RuntimeError(f"X is missing the following columns: {missing}")


        preds, n_out = _get_individual_preds(X)
        if n_out != self.n_out_:
            raise RuntimeError(f"X has {n_out} columns per model, while {self.n_out_} were seen during fit")

        ensemble_preds = np.zeros_like(next(iter(preds.values())))
        for k, w in self.weights_.items():
            ensemble_preds += preds[k] * w

        return ensemble_preds




class GreedyWeightedEnsembleClassifier(GreedyWeightedEnsembleRegressor):
    """Implements https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    Meant to be used with ``TabularFitter``.
    This should only be applied to X which contains predictions (ideally out-of-fold) of other models.
    The predictions should be of correct type (predict or predict_proba) for specified scoring.

    X must have format ``f"{model_name}_{output_i}"``.

    Args:
        scoring: scoring method
        n_bags: number of bags. Set this to 1 to mimic autogluon and speed this up significantly. Defaults to 20.
        p: number/fraction of models in each bag. Defaults to 0.5.
        n_init: number/fraction of best-performing models to initialize each bag with. Defaults to 0.1.
        max_iter: maximum number of iterations per bag. Defaults to 1_000_000.
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
        n_init: int | float | None = 0.1,
        max_iter: int = 1_000_000,
        max_no_improvement: int = 3,
        subsample: int | float | None = 1_000_000,
        max_sec: float | None = None,
        random_state=0,
    ):
        super().__init__(
            scoring=scoring,
            n_bags=n_bags,
            p=p,
            n_init=n_init,
            max_iter=max_iter,
            max_no_improvement=max_no_improvement,
            subsample=subsample,
            max_sec=max_sec,
            random_state=random_state,
        )

    def predict_proba(self, X):
        return super().predict(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(-1)