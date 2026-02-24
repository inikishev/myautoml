from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from torch import nn

from ..metrics.scoring import get_scorer
from ..utils import torch_utils
from .torch_embeddings import TorchEmbeddings

CUDA_IF_AVAILABLE = 'cuda' if torch.cuda.is_available() else "cpu"

class _BaseLearnableELM(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        act_cls: Callable[..., torch.nn.Module],
        hidden_dim: int,
        emb_dim: int,
        emb_config: dict | None,
        rprop_iters,
        lbfgs_iters,
        device,
        verbose: int
    ):
        self.criterion = criterion
        self.device = device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.emb_config = emb_config
        self.act_cls = act_cls
        self.rprop_iters = rprop_iters
        self.lbfgs_iters = lbfgs_iters
        self.verbose = verbose

    def fit(self, X, y):
        validate_data(self, X=X, y=y)

        y = np.asarray(y)

        criterion = self.criterion
        if self.is_classification:
            assert y.ndim == 1
            self.classes_, y = np.unique(y, return_inverse=True)
            if len(self.classes_) > 2:
                n_targets = len(self.classes_)
                if criterion is None: criterion = F.cross_entropy
            else:
                n_targets = 1
                if criterion is None: criterion = F.binary_cross_entropy_with_logits

        else:
            if y.ndim == 1: y = y[:, np.newaxis]
            n_targets = y.shape[-1]

        assert criterion is not None

        emb_config = self.emb_config if self.emb_config is not None else {}
        emb_config.setdefault("emb_dim", self.emb_dim)
        self.embeddings_ = TorchEmbeddings(**emb_config).to(device=self.device).fit(X)

        y_loss = torch.tensor(y, device=self.device)
        if self.is_classification:
            if n_targets > 2:
                y_lstsq = F.one_hot(y_loss, len(self.classes_)).float() # pylint:disable=not-callable
                # y_loss remains long
            else:
                y_lstsq = y_loss = y_loss.unsqueeze(-1).float()
        else:
            y_lstsq = y_loss = y_loss.float()

        assert y_lstsq.ndim == 2 and y_lstsq.shape[-1] == n_targets, y_loss.shape

        X_num, X_cat = self.embeddings_.get_inputs(X)
        self.linear_ = nn.Linear(self.embeddings_.out_channels_, self.hidden_dim).to(device=self.device)
        self.act_ = self.act_cls().to(device=self.device)
        intercept = torch.ones((y_loss.shape[0], 1), device=self.device)

        self.W2_ = None
        self.losses_ = []

        def closure():
            X = self.embeddings_(X_num, X_cat)
            X_hidden = self.act_(self.linear_(X)) # (n_samples, hidden_dim)
            X_hidden = torch.cat([X_hidden, intercept], -1)
            self.W2_ = torch.linalg.lstsq(X_hidden, y_lstsq).solution # (hidden_dim+1, n_targets) # pylint:disable=not-callable
            y_hat = X_hidden @ self.W2_
            loss = criterion(y_hat, y_loss)
            self.zero_grad()
            loss.backward()
            self.losses_.append(loss.detach().cpu().item())
            if self.verbose >= 2: print(self.losses_[-1])
            return loss

        self.train()
        params = [*self.embeddings_.parameters(), *self.linear_.parameters(), *self.act_.parameters()]

        optimizer = torch.optim.Rprop(params)
        n_no_improvement = 0
        best_loss = float('inf')

        for _ in range(self.rprop_iters):
            loss = optimizer.step(closure)
            assert isinstance(loss, torch.Tensor)

            if loss < best_loss:
                best_loss = loss.detach().cpu()
                n_no_improvement = 0
            else:
                n_no_improvement += 1

            if n_no_improvement > 10:
                break

        optimizer = torch.optim.LBFGS(params, max_iter=self.lbfgs_iters, line_search_fn='strong_wolfe')
        optimizer.step(closure)

        return self

    def zero_grad(self):
        self.embeddings_.zero_grad()
        self.linear_.zero_grad()
        self.act_.zero_grad()

    def train(self):
        self.embeddings_.train()
        self.linear_.train()
        self.act_.train()

    def eval(self):
        self.embeddings_.eval()
        self.linear_.eval()
        self.act_.eval()

    def _predict_raw(self, X):
        check_is_fitted(self)
        validate_data(self, X=X, reset=False)
        assert self.W2_ is not None

        self.eval()
        X_emb = self.embeddings_.transform(X)
        X_hidden = self.act_(self.linear_(X_emb))
        intercept = torch.ones((X_hidden.shape[0], 1), device=self.device)
        X_hidden = torch.cat([X_hidden, intercept], -1)

        return (X_hidden @ self.W2_).numpy(force=True)

class LearnableELMClassifier(ClassifierMixin, _BaseLearnableELM):
    """Extreme learning machine with learnable first layer. Second layer weight is computed via least squares.

    Args:
        criterion: criterion, None to use F.cross_entropy or F.binary_cross_entropy_with_logits. Defaults to None.
        act_cls: nonlinearity between first and second layers. Defaults to nn.ELU.
        hidden_dim: hidden dim. Defaults to 512.
        emb_dim: output dimension of embeddings, only has effect when categorical features are present. Defaults to 256.
        emb_config: keyword arguments for embedding. Defaults to None.
        rprop_iters: number of RProp iterations before LBFGS. Defaults to 1000.
        lbfgs_iters: number of LBFGS iterations. Defaults to 1000.
        device: device. Defaults to CUDA_IF_AVAILABLE.
    """
    is_classification: bool = True

    def __init__(
        self,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        act_cls: Callable[..., torch.nn.Module] = nn.ELU,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        emb_config: dict | None = None,
        rprop_iters = 1000,
        lbfgs_iters = 1000,
        device = CUDA_IF_AVAILABLE,
        verbose: int = 0,
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


class LearnableELMRegressor(RegressorMixin, _BaseLearnableELM):
    """Extreme learning machine with learnable first layer. Second layer weight is computed via least squares.

    Args:
        criterion: criterion. Defaults to F.mse_loss.
        act_cls: nonlinearity between first and second layers. Defaults to nn.ELU.
        hidden_dim: hidden dim. Defaults to 512.
        emb_dim: output dimension of embeddings, only has effect when categorical features are present. Defaults to 256.
        emb_config: keyword arguments for embedding. Defaults to None.
        rprop_iters: number of RProp iterations before LBFGS. Defaults to 1000.
        lbfgs_iters: number of LBFGS iterations. Defaults to 1000.
        device: device. Defaults to CUDA_IF_AVAILABLE.
    """

    is_classification: bool = False

    def __init__(
        self,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        act_cls: Callable[..., torch.nn.Module] = nn.ELU,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        emb_config: dict | None = None,
        rprop_iters=1000,
        lbfgs_iters=1000,
        device=CUDA_IF_AVAILABLE,
        verbose: int = 0,
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def predict(self, X):
        y = self._predict_raw(X)
        if y.shape[-1] == 1: y = np.squeeze(y, -1)
        return y