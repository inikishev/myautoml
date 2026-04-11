from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..metrics.scoring import get_scorer
from ..utils import torch_utils
from . import ridge_proba


class _BisineNet(nn.Module):
    def __init__(self, input_dim: int, num_targets: int, num_units: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_targets = num_targets
        self.num_units = num_units
        self.params_per_unit = 2 * input_dim + 3
        self.params_per_target = num_units * self.params_per_unit

        self.params = nn.Parameter(torch.randn(num_targets, self.params_per_target) * 0.1)

    @torch.inference_mode()
    def forward(self, x):
        b = x.shape[0]
        z = torch.zeros(b, self.num_targets, device=x.device, dtype=x.dtype)

        for c in range(self.num_targets):
            p_c = self.params[c]
            z[:, c] = self._forward_class(x, p_c)

        return z

    @torch.inference_mode()
    def _forward_class(self, x, p_c):
        # p_c: (params_per_class,)
        z_c = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for k in range(self.num_units):
            start = k * self.params_per_unit
            a = p_c[start]
            w1 = p_c[start + 1 : start + 1 + self.input_dim]
            b1 = p_c[start + 1 + self.input_dim]
            w2 = p_c[start + 2 + self.input_dim : start + 2 + 2 * self.input_dim]
            b2 = p_c[start + 2 + 2 * self.input_dim]

            u1 = torch.matmul(x, w1) + b1
            u2 = torch.matmul(x, w2) + b2
            z_c += a * torch.sin(u1) * torch.sin(u2)
        return z_c

    @torch.inference_mode()
    def get_flat_params(self):
        return self.params.view(-1)

    @torch.inference_mode()
    def set_flat_params(self, flat_params):
        self.params.data = flat_params.view(self.num_targets, self.params_per_target)

    @torch.inference_mode()
    def compute_grad_and_hessian(self, x):
        """
        Computes gradient and Hessian for each class output separately.
        Returns:
            G: (batch_size, num_classes, params_per_class)
            H: (batch_size, num_classes, params_per_class, params_per_class)
        """
        N = x.shape[0]
        D = self.input_dim
        C = self.num_targets
        K = self.num_units
        P_c = self.params_per_target
        P_u = self.params_per_unit

        G = torch.zeros(N, C, P_c, device=x.device, dtype=x.dtype)
        H = torch.zeros(N, C, P_c, P_c, device=x.device, dtype=x.dtype)

        xx = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)) # (N, D, D)

        for c in range(C):
            p_c = self.params[c]
            for k in range(K):
                start = k * P_u
                a = p_c[start]
                w1 = p_c[start + 1 : start + 1 + D]
                b1 = p_c[start + 1 + D]
                w2 = p_c[start + 2 + D : start + 2 + 2 * D]
                b2 = p_c[start + 2 + 2 * D]

                u1 = torch.matmul(x, w1) + b1
                u2 = torch.matmul(x, w2) + b2

                s1, s2 = torch.sin(u1), torch.sin(u2)
                c1, c2 = torch.cos(u1), torch.cos(u2)

                # Gradient
                G[..., c, start] = s1 * s2
                G[..., c, start + 1 : start + 1 + D] = (a * c1 * s2).unsqueeze(1) * x
                G[..., c, start + 1 + D] = a * c1 * s2
                G[..., c, start + 2 + D : start + 2 + 2 * D] = (a * s1 * c2).unsqueeze(1) * x
                G[..., c, start + 2 + 2 * D] = a * s1 * c2

                # Hessian
                idx_a = start
                idx_w1 = slice(start + 1, start + 1 + D)
                idx_b1 = start + 1 + D
                idx_w2 = slice(start + 2 + D, start + 2 + 2 * D)
                idx_b2 = start + 2 + 2 * D

                H[..., c, idx_a, idx_w1] = (c1 * s2).unsqueeze(1) * x
                H[..., c, idx_a, idx_b1] = c1 * s2
                H[..., c, idx_a, idx_w2] = (s1 * c2).unsqueeze(1) * x
                H[..., c, idx_a, idx_b2] = s1 * c2

                H[..., c, idx_w1, idx_a] = H[..., c, idx_a, idx_w1]
                H[..., c, idx_b1, idx_a] = H[..., c, idx_a, idx_b1]
                H[..., c, idx_w2, idx_a] = H[..., c, idx_a, idx_w2]
                H[..., c, idx_b2, idx_a] = H[..., c, idx_a, idx_b2]

                val_w1w1 = -a * s1 * s2
                H[..., c, idx_w1, idx_w1] = val_w1w1.view(N, 1, 1) * xx
                H[..., c, idx_w1, idx_b1] = val_w1w1.unsqueeze(1) * x
                H[..., c, idx_b1, idx_w1] = H[..., c, idx_w1, idx_b1]
                H[..., c, idx_b1, idx_b1] = val_w1w1

                val_w1w2 = a * c1 * c2
                H[..., c, idx_w1, idx_w2] = val_w1w2.view(N, 1, 1) * xx
                H[..., c, idx_w2, idx_w1] = H[..., c, idx_w1, idx_w2]
                H[..., c, idx_w1, idx_b2] = val_w1w2.unsqueeze(1) * x
                H[..., c, idx_b2, idx_w1] = H[..., c, idx_w1, idx_b2]
                H[..., c, idx_b1, idx_w2] = val_w1w2.unsqueeze(1) * x
                H[..., c, idx_w2, idx_b1] = H[..., c, idx_b1, idx_w2]
                H[..., c, idx_b1, idx_b2] = val_w1w2
                H[..., c, idx_b2, idx_b1] = val_w1w2

                val_w2w2 = -a * s1 * s2
                H[..., c, idx_w2, idx_w2] = val_w2w2.view(N, 1, 1) * xx
                H[..., c, idx_w2, idx_b2] = val_w2w2.unsqueeze(1) * x
                H[..., c, idx_b2, idx_w2] = H[..., c, idx_w2, idx_b2]
                H[..., c, idx_b2, idx_b2] = val_w2w2

        return G, H

class _SFN:
    def __init__(self, model: _BisineNet, lr: float, eps: float, damping: float):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.damping = damping
        self.C = model.num_targets
        self.P_c = model.params_per_target
        self.P = self.C * self.P_c

    @torch.inference_mode()
    def step(self, x: torch.Tensor, y: torch.Tensor, is_classification):
        N = x.shape[0]
        device = x.device
        z = self.model(x)
        def loss_fn(z):
            if is_classification:
                p = torch.softmax(z, dim=1)
                loss = -torch.mean(torch.sum(y * torch.log(p + 1e-10), dim=1))
                return loss, p
            else:
                loss = (z - y).square().mean()
                return loss, z

        loss, p = loss_fn(z)
        G_model, H_model = self.model.compute_grad_and_hessian(x)
        dL_dz = (p - y) / N
        grad = torch.sum(dL_dz.unsqueeze(2) * G_model, dim=0)
        grad_flat = grad.view(-1)
        H = torch.zeros(self.P, self.P, device=device, dtype=x.dtype)
        for c in range(self.C):
            gnc = G_model[:, c, :]
            w1 = (p[:, c] - p[:, c]**2) / N
            w2 = (p[:, c] - y[:, c]) / N
            H_cc = (gnc.t() * w1) @ gnc
            H_cc += torch.sum(w2.view(N, 1, 1) * H_model[:, c, :, :], dim=0)
            H[c*self.P_c : (c+1)*self.P_c, c*self.P_c : (c+1)*self.P_c] = H_cc
            for d in range(c + 1, self.C):
                gnd = G_model[:, d, :]
                w_cd = (-p[:, c] * p[:, d]) / N
                H_cd = (gnc.t() * w_cd) @ gnd
                H[c*self.P_c : (c+1)*self.P_c, d*self.P_c : (d+1)*self.P_c] = H_cd
                H[d*self.P_c : (d+1)*self.P_c, c*self.P_c : (c+1)*self.P_c] = H_cd.t()
        L, V = torch.linalg.eigh(H) # pylint:disable=not-callable
        L_abs = torch.abs(L)
        L_inv = 1.0 / (L_abs + self.damping)
        delta_theta = - (V @ torch.diag(L_inv) @ V.t() @ grad_flat)
        current_params = self.model.get_flat_params()
        alpha = self.lr
        c_ls = 1e-4
        tau = 0.5
        best_alpha = 0.0
        orig_loss = loss.item()
        for _ in range(10):
            new_params = current_params + alpha * delta_theta
            self.model.set_flat_params(new_params)
            with torch.no_grad():
                new_loss, _ = loss_fn(self.model(x))
                new_loss = new_loss.item()
            if new_loss < orig_loss + c_ls * alpha * torch.dot(grad_flat, delta_theta):
                best_alpha = alpha
                break
            alpha *= tau
        if best_alpha == 0.0:
            self.model.set_flat_params(current_params)
        else:
            self.model.set_flat_params(current_params + best_alpha * delta_theta)
        return orig_loss, L

class _BaseBisine(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        num_units: int,
        max_iter: int,
        tol: float,
        lr: float,
        eps: float,
        damping: float,
        max_no_improvement: int,
        scoring,
        device: torch.types.Device,
        dtype: torch.dtype,
        verbose: bool,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.eps = eps
        self.damping = damping
        self.num_units = num_units
        self.verbose = verbose
        self.max_no_improvement = max_no_improvement
        self.device = device
        self.dtype = dtype
        self.scoring = scoring

    @torch.inference_mode()
    def _fit(
        self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        max_iter: int,
    ):
        self.losses_ = []

        n_no_improvement = 0
        best_loss = float("inf")
        best_error = float("inf")

        best_state_dict = None
        steps_until_overfit = None
        n_no_improvement_score = 0

        optimizer = _SFN(self.model_, lr=self.lr, eps=self.eps, damping=self.damping)
        scorer = get_scorer(self.scoring)

        for step in range(max_iter):
            loss, L = optimizer.step(X_tensor, y_tensor, self.is_classification)
            self.losses_.append(float(loss))

            if loss + self.tol < best_loss: n_no_improvement = 0
            else: n_no_improvement += 1
            if n_no_improvement >= self.max_no_improvement: break

            if loss < best_loss:
                best_loss = loss

            if self.verbose: print(f"SFN {step}: loss={loss:.8f}, {n_no_improvement = }")

            if X_test is not None:
                assert y_test is not None
                # compute test score for early stopping

                if self.is_classification:
                    proba = getattr(self, "predict_proba")(X_test)
                    preds = np.argmax(proba, -1)
                else:
                    preds = getattr(self, "predict")(X_test)
                    proba = None

                error = scorer.error(targets=y_test, preds=preds, proba=proba)

                if error < best_error:
                    best_error = error
                    best_state_dict = torch_utils.copy_state_dict(self.model_.state_dict(), 'cpu')
                    steps_until_overfit = step
                    n_no_improvement_score = 0

                else:
                    steps_until_overfit = step
                    n_no_improvement_score += 1

                if self.verbose: print(f"score error={error:.8f}, {n_no_improvement_score = }")
                if n_no_improvement_score >= self.max_no_improvement: break

        return best_state_dict, steps_until_overfit


    def _get_y_tensor(self, y: np.ndarray):
        y_tensor = torch.as_tensor(y, device=self.device)
        if self.is_classification:
            y_oh = torch.zeros(y_tensor.shape[0], len(self.classes_), device=y_tensor.device)
            y_oh.scatter_(1, y_tensor.long().unsqueeze(1), 1.0)
            y_tensor = y_oh

        else:
            y_tensor = y_tensor.to(dtype=self.dtype)
            if y_tensor.ndim == 1: y_tensor = y_tensor.unsqueeze(-1)

        return y_tensor

    def fit(self, X, y, X_test=None, y_test=None):
        X, y = validate_data(self, X=X, y=y)

        if self.is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        if X_test is not None:
            X_test, y_test = validate_data(self, X=X_test, y=y_test)
            if self.is_classification: y_test = self.classes_[y_test]

        # convert train data to tensor
        X_tensor = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        y_tensor = self._get_y_tensor(y)

        # create model if not warm started
        self.model_ = _BisineNet(
            input_dim=X.shape[-1], num_targets=y_tensor.size(-1), num_units=self.num_units
        ).to(device=self.device, dtype=self.dtype)

        # fit
        best_state_dict, steps_until_overfit = self._fit(X_tensor, y_tensor, X_test, y_test, max_iter=self.max_iter)

        if X_test is not None:
            # load state dict with best test score
            assert best_state_dict is not None
            self.model_.load_state_dict(torch_utils.copy_state_dict(best_state_dict, self.device))

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



class BisineClassifier(ClassifierMixin, _BaseBisine):
    """Neural net with single hidden layer using sinusoidal nonlinearity trained with saddle-free Newton.
    Very fast to fit on small datasets, but slow for large datasets.

    In most cases should be used in BaggingClassifier or similar estimators because on its own it overfits.

    Args:
        num_units: number of bisine units. Defaults to 2.
        max_iter: maximum number of iterations. Defaults to 1000.
        tol: tolerance on loss change for convergence. Defaults to 1e-16.
        lr: base step size of the backtracking line search. Defaults to 1.0.
        eps: eps in optimizer.
        damping: damping in optimizer.
        max_no_improvement: max number of consecutive steps that attained no loss improvement larger than ``tol``,
            or validation score imporvement. Defaults to 10.
        scoring: scoring for early stopping. Defaults to "roc_auc".
        device: device. Defaults to 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
    """

    is_classification: bool = True

    def __init__(
        self,
        num_units: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-16,
        lr: float = 1.0,
        eps: float = 1e-16,
        damping: float = 1e-3,
        max_no_improvement: int = 10,
        scoring = "roc_auc",
        device: torch.types.Device = 'cpu',
        dtype: torch.dtype = torch.float64,
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


class BisineRegressor(RegressorMixin, _BaseBisine):
    """Neural net with single hidden layer using sinusoidal nonlinearity trained with saddle-free Newton.
    Very fast to fit on small datasets, but slow for large datasets.

    In most cases should be used in BaggingRegressor or similar estimators because on its own it overfits.

    Args:
        num_units: number of bisine units. Defaults to 2.
        max_iter: maximum number of iterations. Defaults to 1000.
        tol: tolerance on loss change for convergence. Defaults to 1e-16.
        lr: base step size of the backtracking line search. Defaults to 1.0.
        eps: eps in optimizer.
        damping: damping in optimizer.
        max_no_improvement: max number of consecutive steps that attained no loss improvement larger than ``tol``,
            or validation score imporvement. Defaults to 10.
        scoring: scoring for early stopping. Defaults to "mse".
        device: device. Defaults to 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
    """

    is_classification: bool = False

    def __init__(
        self,
        num_units: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-16,
        lr: float = 1.0,
        eps: float = 1e-16,
        damping: float = 1e-3,
        max_no_improvement: int = 10,
        scoring = "mse",
        device: torch.types.Device = 'cpu',
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict(self, X):
        return self.decision_function(X)




class _BaseBisineEarlyStoppingCV(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        num_units: int,
        max_iter: int,
        tol: float,
        lr: float,
        eps: float,
        damping: float,
        max_no_improvement: int,
        scoring,
        device: torch.types.Device,
        dtype: torch.dtype,
        verbose: bool,
        cv: Any,
        shuffle: bool,
        random_state,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.eps = eps
        self.damping = damping
        self.num_units = num_units
        self.verbose = verbose
        self.max_no_improvement = max_no_improvement
        self.device = device
        self.dtype = dtype
        self.scoring = scoring
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        _, y = validate_data(self, X=X, y=y)
        if self.is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        cv = self.cv

        if isinstance(cv, int):
            if self.is_classification: cv = StratifiedKFold(cv, shuffle=self.shuffle, random_state=self.random_state)
            else: cv = KFold(cv, shuffle=self.shuffle, random_state=self.random_state)

        fold_indexes = list(cv.split(X, y))

        self.estimators_: list = []

        cls = BisineClassifier if self.is_classification else BisineRegressor

        for fold, (train_index, test_index) in enumerate(fold_indexes):

            estimator = cls(
                num_units = self.num_units,
                max_iter = self.max_iter,
                tol = self.tol,
                lr = self.lr,
                eps = self.eps,
                damping = self.damping,
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

class BisineEarlyStoppingClassifierCV(ClassifierMixin, _BaseBisineEarlyStoppingCV):
    """Neural net with single hidden layer using sinusoidal nonlinearity trained with saddle-free Newton.
    Very fast to fit on small datasets, but slow for large datasets.

    In most cases should be used in BaggingClassifier or similar estimators because on its own it overfits.

    Args:
        num_units: number of bisine units. Defaults to 2.
        max_iter: maximum number of iterations. Defaults to 1000.
        tol: tolerance on loss change for convergence. Defaults to 1e-16.
        lr: base step size of the backtracking line search. Defaults to 1.0.
        eps: eps in optimizer.
        damping: damping in optimizer.
        max_no_improvement: max number of consecutive steps that attained no loss improvement larger than ``tol``,
            or validation score imporvement. Defaults to 10.
        scoring: scoring for early stopping. Defaults to "roc_auc".
        device: device. Defaults to 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
    """

    is_classification: bool = True

    def __init__(
        self,
        num_units: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-16,
        lr: float = 1.0,
        eps: float = 1e-16,
        damping: float = 1e-3,
        max_no_improvement: int = 10,
        scoring = "roc_auc",
        device: torch.types.Device = 'cpu',
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
        cv: Any = 10,
        shuffle: bool = True,
        random_state = None,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)


class BisineEarlyStoppingRegressorCV(RegressorMixin, _BaseBisineEarlyStoppingCV):
    """Neural net with single hidden layer using sinusoidal nonlinearity trained with saddle-free Newton.
    Very fast to fit on small datasets, but slow for large datasets.

    In most cases should be used in BaggingClassifier or similar estimators because on its own it overfits.

    Args:
        num_units: number of bisine units. Defaults to 2.
        max_iter: maximum number of iterations. Defaults to 1000.
        tol: tolerance on loss change for convergence. Defaults to 1e-16.
        lr: base step size of the backtracking line search. Defaults to 1.0.
        eps: eps in optimizer.
        damping: damping in optimizer.
        max_no_improvement: max number of consecutive steps that attained no loss improvement larger than ``tol``,
            or validation score imporvement. Defaults to 10.
        scoring: scoring for early stopping. Defaults to "roc_auc".
        device: device. Defaults to 'cpu'.
        dtype: dtype. Defaults to torch.float64.
        verbose: verbose. Defaults to False.
    """

    is_classification: bool = False

    def __init__(
        self,
        num_units: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-16,
        lr: float = 1.0,
        eps: float = 1e-16,
        damping: float = 1e-3,
        max_no_improvement: int = 10,
        scoring = "mse",
        device: torch.types.Device = 'cpu',
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
        cv: Any = 10,
        shuffle: bool = True,
        random_state = None,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

