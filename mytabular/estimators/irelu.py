# pylint:disable=not-callable
import copy
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..metrics.scoring import get_scorer
from ..utils import torch_utils
from . import ridge_proba


class _IreluNet(nn.Module):
    def __init__(self, in_channels, out_channels, reg: float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg

        self.W_h = nn.Buffer(torch.empty((0, in_channels)))
        self.b_h = nn.Buffer(torch.empty((0,)))

        # Output weights (output_dim, K)
        self.head = nn.Linear(0, out_channels, bias=True)

    def forward(self, X: torch.Tensor):
        if self.W_h.shape[0] == 0: # bias only
            return torch.zeros((X.shape[0], self.out_channels), device=X.device, dtype=X.dtype) + self.head.bias

        h = F.relu(F.linear(X, self.W_h, self.b_h)) # pylint:disable=not-callable
        return self.head(h)

    def find_best_neuron(self, X, residuals, n_restarts: int, n_iters: int):
        """
        Find (w, b) that maximizes || sum_i grad_i * relu(w^T x_i + b) ||_2
        subject to ||(w, b)||_2 = 1.
        residuals: (B, output_dim)
        """
        device = X.device
        dtype = X.dtype
        B, D = X.shape
        _, O = residuals.shape

        best_val = -float('inf')
        best_w = None
        best_b = None

        x_aug = torch.cat([X, torch.ones((B, 1), device=device, dtype=dtype)], dim=1) # (B, D+1)

        for _ in range(n_restarts):
            wb = torch.randn(D + 1, device=device, dtype=dtype, requires_grad=True)
            with torch.no_grad():
                wb /= torch.linalg.vector_norm(wb)

            # optimizer = torch.optim.Adam([wb], lr=0.01)
            optimizer = torch.optim.LBFGS([wb], line_search_fn='strong_wolfe', max_iter=n_iters)

            def closure():
                optimizer.zero_grad()
                wb_norm = wb / torch.linalg.vector_norm(wb)

                # Projections onto neurons
                proj = F.relu(x_aug @ wb_norm) # (B,)

                # Correlation with residuals for each output dimension
                # corr_k = sum_i residuals_ik * proj_i
                corrs = torch.matmul(residuals.t(), proj) # (O,)

                # Objective: maximize L2 norm of correlations
                loss = -torch.linalg.vector_norm(corrs)

                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                wb_norm = wb / torch.linalg.vector_norm(wb)
                proj = F.relu(x_aug @ wb_norm)
                corrs = torch.matmul(residuals.t(), proj)
                val = torch.linalg.vector_norm(corrs).item()
                if val > best_val:
                    best_val = val
                    best_w = wb_norm[:D].clone()
                    best_b = wb_norm[D].clone()

        return best_w, best_b

    @torch.no_grad
    def add_neuron(self, w, b):
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        self.W_h.set_(torch.cat([self.W_h.clone(), w.unsqueeze(0)], dim=0)) # pyright:ignore[reportArgumentType]
        self.b_h.set_(torch.cat([self.b_h.clone(), b.unsqueeze(0)], dim=0)) # pyright:ignore[reportArgumentType]

        # Update output layer
        K_old = self.head.in_features
        K_new = K_old + 1
        new_head = nn.Linear(K_new, self.out_channels, bias=True, device=device, dtype=dtype)

        if K_old > 0:
            new_head.weight[:, :K_old] = self.head.weight
        new_head.weight[:, K_old:] = 0 # Initialize new neuron weight to 0
        new_head.bias.set_(self.head.bias) # pyright:ignore[reportArgumentType]

        self.head = new_head

    def optimize_output_weights(self, X, y, criterion, n_iters: int):
        """
        Solve min Loss(f(x), y) + lambda * sum_j ||a_j||_2
        using proximal gradient descent or Adam.
        """
        if self.W_h.shape[0] == 0:
            return

        # optimizer = torch.optim.Adam(self.head.parameters(), lr=0.01)
        optimizer = torch.optim.LBFGS(self.head.parameters(), max_iter=n_iters, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            y_hat = self.forward(X)
            loss = criterion(y_hat, y)

            reg = torch.linalg.vector_norm(self.head.weight, ord=2, dim=0).sum()

            loss = loss + self.reg * reg
            loss.backward()
            return loss

        optimizer.step(closure)
        # TODO add proximal step for exact zeros

    @torch.no_grad
    def prune_neurons(self, threshold: float):
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        norms = torch.linalg.vector_norm(self.head.weight, ord=2, dim=0)
        keep_idx = norms > threshold

        if keep_idx.sum() == 0:
            return

        self.W_h.set_(self.W_h[keep_idx.clone()]) # pyright:ignore[reportArgumentType]
        self.b_h.set_(self.b_h[keep_idx.clone()]) # pyright:ignore[reportArgumentType]

        K_new = int(keep_idx.sum().item())
        new_head = nn.Linear(K_new, self.out_channels, bias=True, device=device, dtype=dtype)
        new_head.weight.set_(self.head.weight[:, keep_idx].clone()) # pyright:ignore[reportArgumentType]
        new_head.bias.set_(self.head.bias.clone()) # pyright:ignore[reportArgumentType]
        self.head = new_head


def _train_irelunet_classifier(
    model: _IreluNet,
    X: torch.Tensor,
    y: torch.Tensor,
    max_neurons: int,
    n_restarts: int,
    hidden_iters: int,
    output_iters: int,
    prune_threshold: float,
    final_prune_threshold: float,
    criterion,
    verbose: bool,
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    scorer=None,
    max_no_improvement: int = 10,
):
    y_oh = F.one_hot(y, num_classes=model.out_channels).float() # pylint:disable=not-callable

    best_model = None
    best_error = float('inf')
    n_no_improvement_score = 0

    for i in range(max_neurons):
        model.eval()
        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            residuals = y_oh - probs

        w, b = model.find_best_neuron(X, residuals, n_restarts=n_restarts, n_iters=hidden_iters)
        model.add_neuron(w, b)

        model.train()
        model.optimize_output_weights(X, y, criterion=criterion, n_iters=output_iters)

        # Early stopping
        if X_test is not None and y_test is not None and scorer is not None:
            with torch.no_grad():
                model.eval()
                logits_test = model(X_test)
                probs_test = F.softmax(logits_test, dim=1)
                preds_test = torch.argmax(probs_test, dim=1).cpu().numpy()

            error = scorer.error(targets=y_test.cpu().numpy(), preds=preds_test, proba=probs_test.cpu().numpy())

            if error < best_error:
                best_error = error
                best_model = copy.deepcopy(model)
                n_no_improvement_score = 0
            else:
                n_no_improvement_score += 1

            if verbose: print(f"Iter {i+1}, val_error={error:.8f}, n_no_improve={n_no_improvement_score}")
            if n_no_improvement_score >= max_no_improvement:
                if verbose: print(f"Early stopping at iter {i+1}")
                break

        if (i+1) % 5 == 0:
            model.prune_neurons(prune_threshold)
            if verbose: print(f"Iter {i+1}, Neurons: {model.W_h.shape[0]}")

    if X_test is not None and best_model is not None:
        model = best_model

    model.prune_neurons(final_prune_threshold)
    return model

def _train_irelu_regressor(
    model: _IreluNet,
    X: torch.Tensor,
    y: torch.Tensor,
    max_neurons: int,
    n_restarts: int,
    hidden_iters: int,
    output_iters: int,
    prune_threshold: float,
    final_prune_threshold: float,
    criterion,
    verbose: bool,
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    scorer=None,
    max_no_improvement: int = 10,
):

    best_model = None
    best_error = float('inf')
    n_no_improvement_score = 0

    for i in range(max_neurons):
        model.eval()
        with torch.no_grad():
            y_hat = model(X)
            residuals = y - y_hat

        w, b = model.find_best_neuron(X, residuals, n_restarts=n_restarts, n_iters=hidden_iters)
        model.add_neuron(w, b)

        model.train()
        model.optimize_output_weights(X, y, criterion=criterion, n_iters=output_iters)

        # Early stopping
        if (X_test is not None) and (y_test is not None) and (scorer is not None):
            with torch.no_grad():
                model.eval()
                y_test_pred = model(X_test)

            error = scorer.error(targets=y_test.cpu().numpy(), preds=y_test_pred.cpu().numpy(), proba=None)

            if error < best_error:
                best_error = error
                best_model = copy.deepcopy(model)
                n_no_improvement_score = 0

            else:
                n_no_improvement_score += 1

            if verbose: print(f"Iter {i+1}, val_error={error:.8f}, n_no_improve={n_no_improvement_score}")
            if n_no_improvement_score >= max_no_improvement:
                if verbose: print(f"Early stopping at iter {i+1}")
                break

        if (i+1) % 5 == 0:
            model.prune_neurons(prune_threshold)
            if verbose: print(f"Iter {i+1}, Neurons: {model.W_h.shape[0]}")

    if X_test is not None and best_model is not None:
        model = best_model

    model.prune_neurons(final_prune_threshold)
    return model

CUDA_IF_AVAILABLE = 'cuda' if torch.cuda.is_available() else 'cpu'

class _BaseIrelu(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        max_neurons: int,
        n_restarts: int,
        reg: float,
        hidden_iters: int,
        output_iters: int,
        prune_threshold: float,
        final_prune_threshold: float,
        criterion: Callable,
        max_no_improvement: int,
        scoring,
        device,
        dtype,
        verbose: bool,
    ):
        self.max_neurons = max_neurons
        self.n_restarts = n_restarts
        self.reg = reg
        self.hidden_iters = hidden_iters
        self.output_iters = output_iters
        self.prune_threshold = prune_threshold
        self.final_prune_threshold = final_prune_threshold
        self.criterion = criterion
        self.verbose = verbose
        self.max_no_improvement = max_no_improvement
        self.scoring = scoring
        self.device = device
        self.dtype = dtype

    def fit(self, X, y, X_test=None, y_test=None):
        X, y = validate_data(self, X=X, y=y)

        if self.is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        # Validate that X_test and y_test are both provided or neither
        if (X_test is None) != (y_test is None):
            raise ValueError("Both X_test and y_test must be provided together, or neither.")

        if X_test is not None:
            X_test, y_test = validate_data(self, X=X_test, y=y_test)
            if self.is_classification:
                y_test = self.classes_[y_test.astype(int)]

        # convert train data to tensor
        X_tensor = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        y_tensor = torch.as_tensor(y, device=self.device)

        if self.is_classification:
            y_tensor = y_tensor.long()
            out_channels = len(self.classes_)
        else:
            y_tensor = y_tensor.to(dtype=self.dtype)
            if y_tensor.ndim == 1: y_tensor = y_tensor.unsqueeze(-1)
            out_channels = y_tensor.size(-1)

        # Convert test data to tensors if provided
        X_test_tensor = None
        y_test_tensor = None
        if X_test is not None:
            X_test_tensor = torch.as_tensor(X_test, device=self.device, dtype=self.dtype)
            y_test_tensor = torch.as_tensor(y_test, device=self.device)
            if self.is_classification:
                y_test_tensor = y_test_tensor.long()
            else:
                y_test_tensor = y_test_tensor.to(dtype=self.dtype)
                if y_test_tensor.ndim == 1:
                    y_test_tensor = y_test_tensor.unsqueeze(-1)

        # Setup scorer for early stopping
        scorer = get_scorer(self.scoring) if X_test is not None else None

        train_fn = _train_irelunet_classifier if self.is_classification else _train_irelu_regressor
        self.model_ = train_fn(

            model = _IreluNet(in_channels=X_tensor.shape[1], out_channels=out_channels, reg=self.reg
                              ).to(device=self.device, dtype=self.dtype),

            X = X_tensor,
            y = y_tensor,
            max_neurons = self.max_neurons,
            n_restarts = self.n_restarts,
            hidden_iters = self.hidden_iters,
            output_iters = self.output_iters,
            prune_threshold = self.prune_threshold,
            final_prune_threshold = self.final_prune_threshold,
            criterion = self.criterion,
            verbose = self.verbose,
            X_test = X_test_tensor,
            y_test = y_test_tensor,
            scorer = scorer,
            max_no_improvement = self.max_no_improvement,
        )

        return self


    @torch.inference_mode()
    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        self.model_.eval()
        y = self.model_(X)

        if y.shape[-1] == 2:
            if self.is_classification: assert len(self.classes_) == 2
            y = y[:, -1]

        if y.shape[-1] == 1:
            y = np.squeeze(y, -1)

        return y.numpy(force=True)


class IreluClassifier(ClassifierMixin, _BaseIrelu):
    """IReLU classifier.

    Args:
        max_neurons: maximum number of neurons. Defaults to 50.
        n_restarts: number of random restarts for neuron optimization. Defaults to 5.
        reg: regularization strength. Defaults to 1e-3.
        hidden_iters: number of iterations for hidden neuron optimization. Defaults to 100.
        output_iters: number of iterations for output weight optimization. Defaults to 200.
        prune_threshold: threshold for pruning neurons during training. Defaults to 1e-4.
        final_prune_threshold: threshold for final pruning. Defaults to 1e-5.
        criterion: loss criterion. Defaults to F.cross_entropy.
        max_no_improvement: max number of consecutive neurons without validation score improvement. Defaults to 10.
        scoring: scoring metric for early stopping. Defaults to "roc_auc".
        device: device. Defaults to 'cuda' if available else 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
    """

    is_classification: bool = True

    def __init__(
        self,
        max_neurons: int = 50,
        n_restarts: int = 5,
        reg: float = 1e-3,
        hidden_iters: int = 100,
        output_iters: int = 200,
        prune_threshold: float = 1e-4,
        final_prune_threshold: float = 1e-5,
        criterion: Callable = F.cross_entropy,
        max_no_improvement: int = 10,
        scoring = "roc_auc",
        device = CUDA_IF_AVAILABLE,
        dtype = torch.float64,
        verbose: bool = False,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict_proba(self, X):
        scores = self.decision_function(X) # returns (n_samples, n_classes) or (n_samples, ) for binary
        return ridge_proba._predict_proba(scores, len(self.classes_))

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class IreluRegressor(RegressorMixin, _BaseIrelu):
    """IReLU regressor.

    Args:
        max_neurons: maximum number of neurons. Defaults to 50.
        n_restarts: number of random restarts for neuron optimization. Defaults to 5.
        reg: regularization strength. Defaults to 1e-3.
        hidden_iters: number of iterations for hidden neuron optimization. Defaults to 100.
        output_iters: number of iterations for output weight optimization. Defaults to 200.
        prune_threshold: threshold for pruning neurons during training. Defaults to 1e-4.
        final_prune_threshold: threshold for final pruning. Defaults to 1e-5.
        criterion: loss criterion. Defaults to F.mse_loss.
        max_no_improvement: max number of consecutive neurons without validation score improvement. Defaults to 10.
        scoring: scoring metric for early stopping. Defaults to "mse".
        device: device. Defaults to 'cuda' if available else 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
    """

    is_classification: bool = False

    def __init__(
        self,
        max_neurons: int = 50,
        n_restarts: int = 5,
        reg: float = 1e-3,
        hidden_iters: int = 100,
        output_iters: int = 200,
        prune_threshold: float = 1e-4,
        final_prune_threshold: float = 1e-5,
        criterion: Callable = F.mse_loss,
        max_no_improvement: int = 10,
        scoring = "mse",
        device = CUDA_IF_AVAILABLE,
        dtype = torch.float64,
        verbose: bool = False,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict(self, X):
        return self.decision_function(X)


class _BaseIreluEarlyStoppingCV(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        max_neurons: int,
        n_restarts: int,
        reg: float,
        hidden_iters: int,
        output_iters: int,
        prune_threshold: float,
        final_prune_threshold: float,
        criterion: Callable,
        max_no_improvement: int,
        scoring,
        device,
        dtype,
        verbose: bool,
        cv: Any,
        shuffle: bool,
        random_state: Any,
    ):
        self.max_neurons = max_neurons
        self.n_restarts = n_restarts
        self.reg = reg
        self.hidden_iters = hidden_iters
        self.output_iters = output_iters
        self.prune_threshold = prune_threshold
        self.final_prune_threshold = final_prune_threshold
        self.criterion = criterion
        self.verbose = verbose
        self.max_no_improvement = max_no_improvement
        self.scoring = scoring
        self.device = device
        self.dtype = dtype
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X=X, y=y)
        if self.is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        cv = self.cv

        if isinstance(cv, int):
            if self.is_classification: cv = StratifiedKFold(cv, shuffle=self.shuffle, random_state=self.random_state)
            else: cv = KFold(cv, shuffle=self.shuffle, random_state=self.random_state)

        fold_indexes = list(cv.split(X, y))

        self.estimators_: list = []

        cls = IreluClassifier if self.is_classification else IreluRegressor

        for fold, (train_index, test_index) in enumerate(fold_indexes):

            estimator = cls(
                max_neurons = self.max_neurons,
                n_restarts = self.n_restarts,
                reg = self.reg,
                hidden_iters = self.hidden_iters,
                output_iters = self.output_iters,
                prune_threshold = self.prune_threshold,
                final_prune_threshold = self.final_prune_threshold,
                criterion = self.criterion,
                max_no_improvement = self.max_no_improvement,
                scoring = self.scoring,
                device = self.device,
                dtype = self.dtype,
                verbose = self.verbose,
            )
            estimator.fit(
                X[train_index],
                y[train_index],
                X_test = X[test_index],
                y_test = y[test_index]
            )
            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        proba = None
        for est in self.estimators_:
            if proba is None: proba = est.predict_proba(X)
            else: proba += est.predict_proba(X)

        assert proba is not None
        return proba / len(self.estimators_)

    def predict(self, X):
        if self.is_classification:
            return self.classes_[np.argmax(self.predict_proba(X), -1)]

        preds = None
        for est in self.estimators_:
            if preds is None: preds = est.predict(X)
            else: preds += est.predict(X)

        assert preds is not None
        return preds / len(self.estimators_)

    @torch.inference_mode()
    def decision_function(self, X):
        check_is_fitted(self)
        scores = None
        for est in self.estimators_:
            if scores is None: scores = est.decision_function(X)
            else: scores += est.decision_function(X)

        assert scores is not None
        return scores / len(self.estimators_)


class IreluEarlyStoppingClassifierCV(ClassifierMixin, _BaseIreluEarlyStoppingCV):
    """IReLU classifier with early stopping using cross-validation.

    Trains multiple models using K-Fold CV where each fold uses early stopping
    with the validation fold. Averages predictions across all fold models.

    Args:
        max_neurons: maximum number of neurons. Defaults to 50.
        n_restarts: number of random restarts for neuron optimization. Defaults to 5.
        reg: regularization strength. Defaults to 1e-3.
        hidden_iters: number of iterations for hidden neuron optimization. Defaults to 100.
        output_iters: number of iterations for output weight optimization. Defaults to 200.
        prune_threshold: threshold for pruning neurons during training. Defaults to 1e-4.
        final_prune_threshold: threshold for final pruning. Defaults to 1e-5.
        criterion: loss criterion. Defaults to F.cross_entropy.
        max_no_improvement: max number of consecutive neurons without validation score improvement. Defaults to 10.
        scoring: scoring metric for early stopping. Defaults to "roc_auc".
        device: device. Defaults to 'cuda' if available else 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
        cv: number of folds or CV splitter. Defaults to 10.
        shuffle: whether to shuffle data before splitting. Defaults to True.
        random_state: random state for reproducibility. Defaults to None.
    """

    is_classification: bool = True

    def __init__(
        self,
        max_neurons: int = 50,
        n_restarts: int = 5,
        reg: float = 1e-3,
        hidden_iters: int = 100,
        output_iters: int = 200,
        prune_threshold: float = 1e-4,
        final_prune_threshold: float = 1e-5,
        criterion: Callable = F.cross_entropy,
        max_no_improvement: int = 10,
        scoring = "roc_auc",
        device = CUDA_IF_AVAILABLE,
        dtype = torch.float64,
        verbose: bool = False,
        cv: Any = 10,
        shuffle: bool = True,
        random_state: Any = None,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)


class IreluEarlyStoppingRegressorCV(RegressorMixin, _BaseIreluEarlyStoppingCV):
    """IReLU regressor with early stopping using cross-validation.

    Trains multiple models using K-Fold CV where each fold uses early stopping
    with the validation fold. Averages predictions across all fold models.

    Args:
        max_neurons: maximum number of neurons. Defaults to 50.
        n_restarts: number of random restarts for neuron optimization. Defaults to 5.
        reg: regularization strength. Defaults to 1e-3.
        hidden_iters: number of iterations for hidden neuron optimization. Defaults to 100.
        output_iters: number of iterations for output weight optimization. Defaults to 200.
        prune_threshold: threshold for pruning neurons during training. Defaults to 1e-4.
        final_prune_threshold: threshold for final pruning. Defaults to 1e-5.
        criterion: loss criterion. Defaults to F.mse_loss.
        max_no_improvement: max number of consecutive neurons without validation score improvement. Defaults to 10.
        scoring: scoring metric for early stopping. Defaults to "mse".
        device: device. Defaults to 'cuda' if available else 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
        cv: number of folds or CV splitter. Defaults to 10.
        shuffle: whether to shuffle data before splitting. Defaults to True.
        random_state: random state for reproducibility. Defaults to None.
    """

    is_classification: bool = False

    def __init__(
        self,
        max_neurons: int = 50,
        n_restarts: int = 5,
        reg: float = 1e-3,
        hidden_iters: int = 100,
        output_iters: int = 200,
        prune_threshold: float = 1e-4,
        final_prune_threshold: float = 1e-5,
        criterion: Callable = F.mse_loss,
        max_no_improvement: int = 10,
        scoring = "mse",
        device = CUDA_IF_AVAILABLE,
        dtype = torch.float64,
        verbose: bool = False,
        cv: Any = 10,
        shuffle: bool = True,
        random_state: Any = None,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)
