import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

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

SCORERS: dict[str, Scorer] = {
    "accuracy": Scorer(name="accuracy", score_func=_accuracy, greater_is_better=True, optimum=1),
    "mse": Scorer(name="MSE", score_func=_mse, greater_is_better=False, optimum=0),
    "roc_auc": Scorer(name="ROC AUC", score_func=_roc_auc, greater_is_better=True, optimum=1),
}


DEFAULT_SCORERS: "dict[ProblemType, Scorer]" = {
    "binary": SCORERS["accuracy"],
    "multiclass": SCORERS["accuracy"],
    "regression": SCORERS["mse"],
}


class _ScoreFuncWrapper:
    def __init__(self, score_func, response_method):
        self.score_func = score_func
        self.response_method = response_method

    def __call__(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        if self.response_method == "predict": return float(self.score_func(targets, preds))
        if self.response_method == "predict_proba":
            if proba is None: proba = numpy_utils.one_hot(preds, max(np.max(preds), np.max(targets)) + 1)
            return float(self.score_func(targets, proba))
        raise ValueError(self.response_method)

def make_scorer(score_func, /, response_method='predict', greater_is_better=True):
    if isinstance(score_func, Scorer): return score_func
    if isinstance(score_func, str): return copy.deepcopy(SCORERS[score_func])
    if callable(score_func):

        return Scorer(
            name = python_utils.get_qualname(score_func),
            score_func=_ScoreFuncWrapper(score_func, response_method),
            greater_is_better=greater_is_better,
            optimum = 0,
        )
    raise TypeError(type(score_func))

# TODO wrapper for sklearn scorer and maybe autogluon?