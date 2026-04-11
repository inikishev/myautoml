import math
from typing import Literal, Any

import numpy as np
import copy
from scipy.special import expit, softmax # pylint:disable=no-name-in-module
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..metrics.scoring import get_scorer
from ..utils import numpy_utils
from . import ridge_proba

def _validate_test_indexes(cat_test_indexes, n_samples: int):
    if isinstance(cat_test_indexes, np.ndarray): cat_test_indexes = cat_test_indexes.tolist()

    if len(cat_test_indexes) != n_samples:
        raise RuntimeError(f"There are {len(cat_test_indexes)} test indices, but {n_samples} samples")

    if len(set(cat_test_indexes)) != len(cat_test_indexes):
        raise RuntimeError("There are repeating test indices")

    if set(cat_test_indexes) != set(range(n_samples)):
        raise RuntimeError(f"Test indices are different from list(range(n_samples)): "
                            f"{set(cat_test_indexes) ^ set(range(n_samples))}")


class _Loss:
    def gradient(self, target: np.ndarray, preds: np.ndarray) -> np.ndarray:
        """for multiclass classification target is one hot encoded."""
        raise NotImplementedError

class _MSELoss(_Loss):
    def gradient(self, target: np.ndarray, preds: np.ndarray) -> np.ndarray:
        return preds - target

class _MAELoss(_Loss):
    def gradient(self, target: np.ndarray, preds: np.ndarray) -> np.ndarray:
        return np.sign(preds - target)

class _CELoss(_Loss):
    def gradient(self, target: np.ndarray, preds: np.ndarray) -> np.ndarray:
        if preds.ndim == 1: return expit(preds) - target
        return softmax(preds, -1) - target

class _BaseOOFBoosting(BaseEstimator):
    is_classification: bool
    def __init__(
        self,
        estimator,
        n_rounds: int,
        scoring,
        step_size: float | None,
        ls_iters: int | None,
        loss: Literal["mse", "mae", "ce"] | _Loss,
        tol: float,
        max_no_improvement: int,
        cv: int | Any,
        shuffle: bool,
        random_state,
        verbose
    ):
        self.estimator = estimator
        self.n_rounds = n_rounds
        self.step_size = step_size
        self.ls_iters = ls_iters
        self.loss = loss
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.cv = cv
        self.shuffle = shuffle
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, **fit_kwargs):
        _, y = validate_data(self, X=X, y=y)
        if self.is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        scorer = get_scorer(self.scoring)
        random_state = check_random_state(self.random_state)

        cv = self.cv
        if isinstance(cv, int):
            if self.is_classification: cv = StratifiedKFold(cv, shuffle=self.shuffle, random_state=random_state)
            else: cv = KFold(cv, shuffle=self.shuffle, random_state=random_state)

        fold_indexes = list(cv.split(X, y))

        loss = self.loss
        if loss == "mse": loss = _MSELoss()
        elif loss == "mae": loss = _MAELoss()
        elif loss == "ce": loss = _CELoss()
        if not isinstance(loss, _Loss): raise RuntimeError(loss)

        # boost target should have same shape as estimator predict
        # for binary classification we only predict positive label
        if self.is_classification:
            if len(self.classes_) > 2: boost_target = numpy_utils.one_hot(y, len(self.classes_))
            else: boost_target = y
        else:
            boost_target = y # residual or gradient

        oob_boost_sum = np.zeros_like(boost_target)
        g = loss.gradient(boost_target, preds=oob_boost_sum)

        def get_error(preds: np.ndarray):
            if self.is_classification:
                if preds.ndim == 1:
                    preds = expit(preds)
                    return scorer.error(y, preds=preds>0.5, proba=np.stack([1-preds,preds],-1))

                preds = softmax(preds, -1)
                return scorer.error(y, preds=preds.argmax(-1), proba=preds)

            return scorer.error(y, preds=preds, proba=None)

        n_no_improvement = 0
        step_sizes = []
        self.step_sizes_ = []
        best_error = float("inf")
        self.estimators_ = []

        for round_ in range(self.n_rounds):

            # Compute out-of-fold gradient predictions
            test_g_preds_list = []
            test_indexes_list = []
            for fold, (train_index, test_index) in enumerate(fold_indexes):

                estimator = copy.deepcopy(self.estimator).fit(X[train_index], g[train_index], **fit_kwargs)
                self.estimators_.append(estimator)
                preds = estimator.predict(X[test_index]) # note that estimator is always a regressor
                test_indexes_list.extend(test_index)
                test_g_preds_list.extend(preds)

            _validate_test_indexes(test_indexes_list, len(y))
            argsort = np.argsort(test_indexes_list)
            oof_g_preds = np.asarray(test_g_preds_list)[argsort]

            # Find step size
            if self.ls_iters is None or self.ls_iters < 1:
                # No line search, use fixed step size
                assert self.step_size is not None
                step_size = self.step_size

            else:
                # Line search to minimize oof scorer error
                def objective(alpha: float):
                    return get_error(oob_boost_sum + oof_g_preds * alpha)

                from scipy.optimize import minimize_scalar
                options = dict(maxiter=self.ls_iters)
                if self.step_size is None:
                    res = minimize_scalar(objective, bracket=(0, 1), options=options)

                else:
                    res = minimize_scalar(objective, bracket=(0, self.step_size), bounds=(0, self.step_size), options=options)

                step_size = res.x # pyright:ignore[reportAttributeAccessIssue]

            # Make a step
            oob_boost_sum = oob_boost_sum + oof_g_preds * step_size
            error = get_error(oob_boost_sum)

            if self.verbose > 0:
                print(f"{round_}: {error = :.6f}, {best_error = :.6f}, {n_no_improvement = }")

            step_sizes.append(step_size)
            g = loss.gradient(boost_target, preds=oob_boost_sum)

            if math.isfinite(best_error) and error + self.tol >= best_error:
                n_no_improvement += 1
                if n_no_improvement >= self.max_no_improvement:
                    break

            else:
                n_no_improvement = 0
                best_error = error
                self.step_sizes_ = step_sizes.copy()

        return self

    def decision_function(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, reset=False)

        preds = None
        for est, step_size in zip(self.estimators_, self.step_sizes_):
            # this returns (n_samples, ) for regression and binary classification,
            # otherwise (n_samples, n_classes), and it returns logits
            g = est.predict(X)
            if preds is None: preds = g * step_size
            else: preds += g * step_size

        return preds

class OOFBoostingClassifier(ClassifierMixin, _BaseOOFBoosting):
    is_classification: bool = True

    def __init__(
        self,
        estimator,
        n_rounds: int,
        scoring,
        step_size: float | None = None,
        ls_iters: int | None = 32,
        loss: Literal["mse", "mae", "ce"] | _Loss = "ce",
        tol: float = 1e-12,
        max_no_improvement: int = 10,
        cv: Any = 10,
        shuffle: bool = True,
        random_state = None,
        verbose = 0
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        return ridge_proba._predict_proba(scores, len(self.classes_))

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class OOFBoostingRegressor(RegressorMixin, _BaseOOFBoosting):
    is_classification: bool = False

    def __init__(
        self,
        estimator,
        n_rounds: int,
        scoring,
        step_size: float | None = None,
        ls_iters: int | None = 32,
        loss: Literal["mse", "mae", "ce"] | _Loss = "mse",
        max_no_improvement: int = 10,
        tol: float = 1e-12,
        cv: Any = 10,
        shuffle: bool = True,
        random_state = None,
        verbose = 0
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict(self, X):
        return self.decision_function(X)
