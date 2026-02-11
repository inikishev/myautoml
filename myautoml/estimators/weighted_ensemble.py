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

from ..metrics.scoring import make_scorer
from ..utils.polars_utils import to_dataframe, to_series


def _get_individual_preds(X) -> dict[str, np.ndarray]:
    X = to_dataframe(X)

    models = set()
    indexes = defaultdict(list)

    # get all models
    for col in X.columns:
        model, index = col.rsplit("_", 1)
        models.add(model)
        indexes[model].append(int(index))

    # verify that all models have same number of columns
    first_ind = None
    for model,ind in indexes.items():
        if len(ind) != len(set(ind)):
            raise RuntimeError(f"Somehow two colums have the same name among {X.columns}")
        if first_ind is None: first_ind = set(ind)
        if set(ind) != first_ind:
            raise RuntimeError(f"Model {model} has indexes {sorted(ind)}, while another model had {sorted(first_ind)}.")

    # split into preds
    preds = {}
    assert first_ind is not None
    required_cols = sorted(first_ind)
    for model in models:
        model_cols = [f"{model}_{col}" for col in required_cols]
        preds[model] = X.select(model_cols).to_numpy()

    return preds

class GreedyWeightedEnsembleRegressor(TransformerMixin, BaseEstimator):
    """Implements https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    This should only be applied to X which contains predictions (ideally out-of-fold) of other models.
    The predictions should be of correct type (predict or predict_proba) for specified scoring.

    X must have format ``f"{model_name}_{output_i}"``, which ``TabularFitter`` follows.
    Args:
        scoring: scoring method
        n_bags: number of bags. Defaults to 20.
        p: fraction of models in each bag. Defaults to 0.5.
        n_init: number of best-performing models to initialize each bag with. Defaults to 5.

    """
    is_classification = False
    def __init__(self, scoring, n_bags: int = 20, p: float = 0.5, n_init: int | float | None = 5):
        self.scoring = scoring
        self.n_bags = n_bags
        self.p = p
        self.n_init = n_init

    def fit(self, X, y):
        validate_data(self, X=X, y=y, ensure_all_finite=False)

        if isinstance(y, (pl.Series, pl.DataFrame)):
            y = y.to_numpy()
        else:
            y = np.asarray(y)

        scorer = make_scorer(self.scoring)
        preds = _get_individual_preds(X)
        test_pred = next(iter(preds.values()))
        if self.is_classification:
            if np.squeeze(test_pred).ndim == 1: raise RuntimeError("X must contain probabilites predicted by each model.")

        def evaluate_ensemble(model: Sequence[str]):
            assert isinstance(model, str) is False
            ensemble_preds = np.stack([preds[m] for m in models], 0).mean(0)
            return scorer.error(y, ensemble_preds, ensemble_preds)

        # precompute errors of all models for first iter in each bag
        errors = {}
        for model, y_hat in preds.items():
            errors[model] = scorer.error(y, y_hat, y_hat)

        weights: dict[str, int] = {}
        for i in range(self.n_bags):

            # select p models
            models = random.sample(list(preds.keys()), k=math.ceil(len(preds)*self.p))

            # initialize ensemble
            ensemble = []
            n_init = self.n_init
            if isinstance(n_init, float): n_init = math.floor(n_init) * len(preds)
            if (n_init is not None) and (n_init > 0):
                ensemble = sorted(preds.keys(), key=lambda k: errors[k])[:n_init]

            # hill climbing
            error = evaluate_ensemble(ensemble)

            while True:

                best_error = error
                best_model = None

                for model in models:

                    trial_error = evaluate_ensemble(ensemble + [model])
                    if trial_error < best_error:
                        best_error = trial_error
                        best_model = model

                if best_model is None: break
                ensemble.append(best_model)

            # update weights with selected models
            for model in ensemble:
                if model not in weights: weights[model] = 1
                else: weights[model] += 1

        # normalize weights
        total = sum(weights.values())
        self.weights_: dict[str, float] = {k: v / total for k,v in weights.items()}
        return self


    def predict(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, reset=False, ensure_all_finite=False)

        preds = _get_individual_preds(X)
        ensemble_preds = np.zeros_like(next(iter(preds.values())))
        for k, w in self.weights_.items():
            ensemble_preds += preds[k] * w

        return ensemble_preds




class GreedyWeightedEnsembleClassifier(GreedyWeightedEnsembleRegressor):
    """Implements https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    This should only be applied to X which contains predictions (ideally out-of-fold) of other models.
    The predictions should be of correct type (predict or predict_proba) for specified scoring.

    X must have format ``f"{model_name}_{output_i}"``, which ``TabularFitter`` follows.

    Args:
        scoring: scoring method
        n_bags: number of bags. Defaults to 20.
        p: fraction of models in each bag. Defaults to 0.5.
        n_init: number of best-performing models to initialize each bag with. Defaults to 5.

    """
    is_classification = True
    def __init__(self, scoring, n_bags: int = 20, p: float = 0.5, n_init: int | float | None = 5):
        super().__init__(scoring=scoring, n_bags=n_bags, p=p, n_init=n_init)

    def predict_proba(self, X):
        return super().predict(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(-1)