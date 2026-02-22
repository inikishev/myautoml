from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from sklearn.utils import check_random_state
from ..metrics.scoring import get_scorer


from ..utils import torch_utils

if torch_utils.TORCH_INSTALLED:
    import torch
    CUDA_IF_AVAILBLE = 'cuda' if torch.cuda.is_available() else 'cpu'

else:
    torch = None
    CUDA_IF_AVAILBLE = None

if TYPE_CHECKING:
    import torch

class _BaseDFLinear(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        scoring,
        l1: float = 0,
        l2: float = 0,
        methods = None,
        jac='3-point',
        device=CUDA_IF_AVAILBLE,
        dtype=torch.float32,
        random_state=0,
        verbose=False,
    ):
        self.scoring = scoring
        self.l1 = l1
        self.l2 = l2
        self.methods = methods
        self.jac = jac
        self.device = device
        self.dtype = dtype
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X, y = validate_data(self, X=X, y=y)
        if self.is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        rng = check_random_state(self.random_state)
        scorer = get_scorer(self.scoring)

        n_features = X.shape[1]

        if self.is_classification:
            assert y.ndim == 1
            n_targets = len(np.unique(y))
            if n_targets == 2: n_targets = 1
        else:
            if y.ndim == 1: y = y[:, np.newaxis]
            n_targets = y.shape[1]

        assert torch is not None
        X = torch.tensor(X, device=self.device, dtype=self.dtype)

        # might make sklearnex issue later
        # -
        # tracemalloc.start()
        # process = psutil.Process(os.getpid())

        # def get_mem_stats():
        #     gc.collect()
        #     current, peak = tracemalloc.get_traced_memory()
        #     return {
        #         'rss_mb': process.memory_info().rss / 1024 / 1024,
        #         'trace_current_mb': current / 1024 / 1024,
        #         'trace_peak_mb': peak / 1024 / 1024
        #     }

        # baseline = get_mem_stats()
        # print(f"🔍 Baseline Memory: {baseline}")
        # -


        def objective(x: np.ndarray):
            # -
            # if not hasattr(objective, 'call_count'):
            #     objective.call_count = 0
            #     objective.mem_history = []
            # objective.call_count += 1

            # if objective.call_count % 50 == 0:
            #     stats = get_mem_stats()
            #     growth = stats['rss_mb'] - baseline['rss_mb']
            #     objective.mem_history.append(stats)
            #     print(f"🔍 Iter {objective.call_count}: RSS={stats['rss_mb']:.1f}MB "
            #         f"(+{growth:.1f}MB) | Tracemalloc={stats['trace_current_mb']:.1f}MB")

            #     # If growth > 100MB, snapshot top allocations
            #     if growth > 1000:
            #         snapshot = tracemalloc.take_snapshot()
            #         top_stats = snapshot.statistics('lineno')[:10]
            #         print("⚠️ Top 10 memory allocations:")
            #         for stat in top_stats:
            #             print(f"  {stat}")
            # -

            assert torch is not None
            x_torch = torch.tensor(x, device=self.device, dtype=self.dtype)
            W = x_torch[:(n_features * n_targets)].view(n_features, n_targets)
            b = x_torch[(n_features * n_targets):].view(n_targets)

            out = (X @ W + b)
            if self.is_classification:
                if n_targets == 1:
                    out = torch.squeeze(out, -1)
                    proba = torch.stack([1-out, out], -1)
                    preds = (out > 0.5).long()
                else:
                    proba = out
                    preds = proba.argmax(-1)
            else:
                preds = out
                proba = None

            preds = preds.numpy(force=True)
            if proba is not None: proba = proba.numpy(force=True)

            error = scorer.error(y, preds, proba)
            if self.l1 > 0: error = error + np.abs(x).sum() * self.l1
            if self.l2 > 0: error = error + (x ** 2).sum() * self.l2

            return error

        import scipy.optimize
        W_init = torch.tensor(rng.standard_normal((n_features, n_targets)), device=self.device, dtype=self.dtype) * 0.5
        b_init = torch.zeros(n_targets, device=self.device, dtype=self.dtype)
        params = torch.cat([W_init.ravel(), b_init.ravel()]).numpy(force=True)
        del W_init, b_init

        methods = self.methods
        if methods is None:
            if params.size < 8: methods = ["bfgs", "cobyqa", "cobyla", "powell", "nelder-mead"]
            elif params.size < 32: methods = ["cobyqa", "cobyla", "powell", "nelder-mead"]
            elif params.size < 128: methods = ["cobyla", "powell", "nelder-mead",]
            else: methods = ["powell", "nelder-mead",]

        self.eval_count_ = 0
        for method in methods:
            jac = self.jac
            if method.lower() in ("cobyla", "cobyqa", "powell", "nelder-mead"): jac = None

            with torch.inference_mode():
                self.res_ = scipy.optimize.minimize(
                    objective,
                    x0=params,
                    method=method,
                    jac=jac,
                )

            if self.verbose:
                print(method, self.res_)

            params = self.res_.x

        self.W_ = params[:(n_features * n_targets)].reshape(n_features, n_targets)
        self.b_ = params[(n_features * n_targets):].reshape(n_targets)

        # -
        # final = get_mem_stats()
        # print(f"🔍 Final Memory: {final}")
        # print(f"🔍 Total Growth: {final['rss_mb'] - baseline['rss_mb']:.1f}MB")

        # import objgraph
        # print("🔍 Most common types:")
        # objgraph.show_most_common_types(limit=10)

        # tracemalloc.stop()
        #

        return self

    def _predict_raw(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        X = X.astype(self.W_.dtype)
        return (X @ self.W_ + self.b_)



class DFLinearClassifier(ClassifierMixin, _BaseDFLinear):
    """Fit a linear model to optimize a score such as accuracy or ROC AUC directly using gradient-free optimization.
    This is considerably slower than LogisticRegression and only recommended when ``n_features * n_classes < 16``
    (for binary classification ``n_features < 16``).

    Note:
        Some metrics, like accuracy, are highly discontinuous when the dataset is very small, which can cause
        derivative-free solvers to fail.

    Warning:
        There is a memory leak in ``sklearnex`` versions of some sklearn metrics (like roc_auc).
        If you have a memory leak, disable ``sklearnex``.

    Args:
        scoring: scoring
        l1: L1 regularization. Defaults to 0.
        l2: L2 regularization. Defaults to 0.
        methods: sequence of strings - scipy.optimize.minimize methods. Each method continues from solution found
            by previous method. By default picks methods based on number of parameters. Defaults to None.
        jac: how to compute jacobian, 2-point or 3-point. Defaults to '3-point'.
        device: device for matrix multiplication, set to "cpu" for small datasets. Defaults to CUDA_IF_AVAILBLE.
        dtype: dtype. Defaults to torch.float32.
        random_state: seed. Defaults to 0.
        verbose: whether to print optimization results. Defaults to False.
    """

    is_classification: bool = True

    def __init__(
        self,
        scoring,
        l1: float = 0,
        l2: float = 0,
        methods = None,
        device=CUDA_IF_AVAILBLE,
        dtype=torch.float32,
        random_state=0,
        verbose=False,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict_proba(self, X):
        proba = self._predict_raw(X)
        if proba.shape[-1] == 1:
            proba = np.squeeze(proba, -1)
            proba = np.stack([1-proba, proba], -1)
        return proba

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]



class DFLinearRegressor(RegressorMixin, _BaseDFLinear):
    """Fit a linear model to optimize a score directly using gradient-free optimization.
    This is considerably slower than Ridge and only recommended when ``n_features * n_targets < 16``

    Note:
        Some metrics, like accuracy, are highly discontinuous when the dataset is very small, which can cause
        derivative-free solvers to fail.

    Warning:
        There is a memory leak in ``sklearnex`` versions of some sklearn metrics (like roc_auc).
        If you have a memory leak, disable ``sklearnex``.

    Args:
        scoring: scoring
        l1: L1 regularization. Defaults to 0.
        l2: L2 regularization. Defaults to 0.
        methods: sequence of strings - scipy.optimize.minimize methods. Each method continues from solution found
            by previous method. By default picks methods based on number of parameters. Defaults to None.
        jac: how to compute jacobian, 2-point or 3-point. Defaults to '3-point'.
        device: device for matrix multiplication, set to "cpu" for small datasets. Defaults to CUDA_IF_AVAILBLE.
        dtype: dtype. Defaults to torch.float32.
        random_state: seed. Defaults to 0.
        verbose: whether to print optimization results. Defaults to False.
    """

    is_classification: bool = False

    def __init__(
        self,
        scoring,
        l1: float = 0,
        l2: float = 0,
        methods = None,
        device=CUDA_IF_AVAILBLE,
        dtype=torch.float32,
        random_state=0,
        verbose=False,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)


    def predict(self, X):
        y = self._predict_raw(X)
        if y.shape[-1] == 1: y = np.squeeze(y, -1)
        return y