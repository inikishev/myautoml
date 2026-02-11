from collections.abc import Callable
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
import sklearn.metrics

if TYPE_CHECKING:
    from .fitter_utils import ProblemType

class _ScoreInputs(NamedTuple):
    targets: np.ndarray
    preds: np.ndarray
    proba: np.ndarray | None

class Scorer:
    def __init__(self, name: str, score_func: Callable[[_ScoreInputs], float], greater_is_better:bool, optimum:float):
        self.name = name
        self.score_func = score_func
        self.greater_is_better= greater_is_better
        self.optimum = optimum

    def score(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        return self.score_func(_ScoreInputs(targets=targets, preds=preds, proba=proba))

    def score_and_error(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        score = self.score(targets=targets, preds=preds, proba=proba)
        sign = -1 if self.greater_is_better else 1
        error = sign * (score - self.optimum)
        return score, error

    def error(self, targets: np.ndarray, preds: np.ndarray, proba: np.ndarray | None):
        score, error = self.score_and_error(targets=targets, preds=preds, proba=proba)
        return error


# ---------------------------------- metrics --------------------------------- #
def _one_hot(preds: np.ndarray):
    num_classes = np.max(preds) + 1
    num_samples = preds.shape[0]
    one_hot = np.zeros((preds.size, num_classes))
    one_hot[np.arange(num_samples), preds.astype(np.uint64)] = 1
    return one_hot.reshape((*preds.shape, num_classes))

def _accuracy(x: _ScoreInputs):
    return float(sklearn.metrics.accuracy_score(x.targets, x.preds))

def _mse(x: _ScoreInputs):
    return float(sklearn.metrics.mean_squared_error(x.targets, x.preds))

def _roc_auc(x: _ScoreInputs):
    proba = x.proba

    if proba is None: # fallback to hard targets
        proba = _one_hot(x.preds)

    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1] # handle binary classification

    return float(sklearn.metrics.roc_auc_score(x.targets, proba, multi_class='ovr'))

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