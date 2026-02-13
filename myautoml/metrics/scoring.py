import copy
import warnings
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import sklearn.metrics

from ..utils import numpy_utils, python_utils

if TYPE_CHECKING:
    from ..core._fitter_utils import ProblemType

class Scorer:
    def __init__(self, name: str, score_func: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float], greater_is_better:bool, optimum:float):
        self.name = name
        self.score_func = score_func
        self.greater_is_better= greater_is_better
        self.optimum = optimum

    def score(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        return self.score_func(targets, preds, proba)

    def score_and_error(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        score = self.score(targets=targets, preds=preds, proba=proba)
        sign = -1 if self.greater_is_better else 1
        error = sign * (score - self.optimum)
        return score, error

    def error(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        score, error = self.score_and_error(targets=targets, preds=preds, proba=proba)
        return error


# ---------------------------------- metrics --------------------------------- #

def _accuracy(targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
    return float(sklearn.metrics.accuracy_score(targets, preds))

def _mse(targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
    return float(sklearn.metrics.mean_squared_error(targets, preds))

def _roc_auc(targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):

    if proba is None: # fallback to hard targets
        proba = numpy_utils.one_hot(preds, max(np.max(preds), np.max(targets)) + 1)

    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1] # handle binary classification

    return float(sklearn.metrics.roc_auc_score(targets, proba, multi_class='ovr'))

def _spearmanr(targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
    import scipy.stats
    return scipy.stats.spearmanr(targets, preds).statistic # pyright:ignore[reportAttributeAccessIssue]

SCORERS: dict[str, Scorer] = {
    "accuracy": Scorer(name="accuracy", score_func=_accuracy, greater_is_better=True, optimum=1),
    "mse": Scorer(name="MSE", score_func=_mse, greater_is_better=False, optimum=0),
    "roc_auc": Scorer(name="ROC AUC", score_func=_roc_auc, greater_is_better=True, optimum=1),
    "spearmanr": Scorer(name="spearmanr", score_func=_spearmanr, greater_is_better=True, optimum=1),
}


DEFAULT_SCORERS: "dict[ProblemType, Scorer]" = {
    "binary": SCORERS["accuracy"],
    "multiclass": SCORERS["accuracy"],
    "regression": SCORERS["mse"],
}

class _CustomScoreFuncWrapper:
    def __init__(self, score_func, response_method):
        self.score_func = score_func
        self.response_method = response_method

    def __call__(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        if self.response_method == "predict": return float(self.score_func(targets, preds))
        if self.response_method == "predict_proba":
            if proba is None: proba = numpy_utils.one_hot(preds, max(np.max(preds), np.max(targets)) + 1)
            return float(self.score_func(targets, proba))
        raise ValueError(self.response_method)

class _SklearnScorerWrapper:
    def __init__(self, scoring: str, ):
        assert isinstance(scoring, str) # otherwise get_scorer won't return a Scorer.
        self.scorer: Any = sklearn.metrics.get_scorer(scoring)

    def __call__(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        if self.scorer._response_method == "predict": y_hat = preds
        elif self.scorer._response_method == "predict_proba":
            if proba is None: proba = numpy_utils.one_hot(preds, max(np.max(preds), np.max(targets)) + 1)
            y_hat = proba
        else: raise ValueError(self.scorer._response_method)

        return self.scorer._sign * self.scorer._score_func(targets, y_hat, **self.scorer._kwargs)


def get_scorer(metric: str | Callable | Scorer, /, response_method='predict', greater_is_better=True):
    if isinstance(metric, Scorer): return metric

    if isinstance(metric, str):
        if metric in SCORERS: return copy.deepcopy(SCORERS[metric])
        else:
            return Scorer(
                name = metric,
                score_func = _SklearnScorerWrapper(metric),
                greater_is_better = True, # sklearn always flips sign such that greater is better
                optimum = 0,
            )

    if callable(metric):

        return Scorer(
            name = python_utils.get_qualname(metric),
            score_func=_CustomScoreFuncWrapper(metric, response_method),
            greater_is_better=greater_is_better,
            optimum = 0,
        )

    raise TypeError(type(metric))


