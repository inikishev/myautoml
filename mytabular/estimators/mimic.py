import copy

import numpy as np
import polars as pl
import torch
from sklearn.base import BaseEstimator, MultiOutputMixin, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from torch.nn import functional as F

from ..utils import polars_utils

CUDA_IF_AVAILABLE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _eps_clip(x: np.ndarray):
    return x.clip(np.finfo(x.dtype).eps) # pylint:disable=no-member


def mimic_iterative(cov_ref: torch.Tensor, is_XTX: bool, X_new: torch.Tensor, criterion, rprop_iters, lbfgs_iters, n_resets):
    W = torch.eye(X_new.shape[1], device=X_new.device, dtype=X_new.dtype, requires_grad=True)

    def get_cov(X):
        if is_XTX: return X.T @ X
        return X @ X.T

    losses = []
    def objective():
        X_mimic = X_new @ W
        cov_mimic = get_cov(X_mimic)
        loss = criterion(cov_mimic, cov_ref)
        W.grad = None
        loss.backward()
        losses.append(loss.detach().cpu().item())
        return loss

    for _ in range(n_resets):
        optimizer = torch.optim.Rprop([W], lr=1e-6, step_sizes=(1e-6, 1e5))
        for _ in range(rprop_iters): optimizer.step(objective)

        optimizer = torch.optim.LBFGS(
            [W],
            line_search_fn="strong_wolfe",
            max_iter=lbfgs_iters,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )
        optimizer.step(objective)

    return W.detach(), losses

def mimic_eigh(cov_ref: torch.Tensor, X_new: torch.Tensor, reg: float):

    cov_new = X_new.T @ X_new
    L_ref, Q_ref = torch.linalg.eigh(cov_ref) # pylint:disable=not-callable
    L_new, Q_new = torch.linalg.eigh(cov_new) # pylint:disable=not-callable
    L_ref = L_ref.clip(min=reg)
    L_new = L_new.clip(min=reg)

    cov_ref_sqrt = Q_ref @ L_ref.sqrt().diag_embed() @ Q_ref.T
    cov_new_inv_sqrt = Q_new @ L_new.sqrt().reciprocal().diag_embed() @ Q_new.T

    return cov_new_inv_sqrt @ cov_ref_sqrt


class _BaseMimic(TransformerMixin, BaseEstimator):
    include: polars_utils.PolarsColumnSelector | None
    exclude: polars_utils.PolarsColumnSelector | None

    def _fit(self, X_ref: np.ndarray, X_new: np.ndarray) -> None:
        raise NotImplementedError

    def _transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X_ref, X_new):
        """Fit this estimator and store coefficients needed to transform new samples from ``X_new`` to mimic ``X_ref``.

        Args:
            X_ref: reference dataset, shape ``(n_samples, n_features)``.
            X_new: new dataset, shape ``(n_samples, n_features)``.
        """
        X_ref = polars_utils.to_dataframe(X_ref)
        X_new = polars_utils.to_dataframe(X_new)
        X_ref = polars_utils.include_exclude_cols(X_ref, self.include, self.exclude)
        X_new = polars_utils.include_exclude_cols(X_new, self.include, self.exclude)
        self.cols_ = X_ref.columns
        self.schema_ = dict(X_ref.schema)

        X_ref = validate_data(self, X=X_ref)
        X_new = validate_data(self, X=X_new)
        self._fit(X_ref=X_ref, X_new=X_new)
        return self

    def transform(self, X):
        """Transform a batch of samples from same distribution as ``X_new`` passed to ``fit``
        to mimic the distribution of ``X_ref``.

        Args:
            X: batch of samples from the same distribution as ``X_new``, shape ``(n_samples, n_features)``.
        """
        check_is_fitted(self)

        X = polars_utils.to_dataframe(X)
        X_cols = X.select(self.cols_)

        X_np = validate_data(self, X=X_cols, reset=False)
        X_mimic = self._transform(X_np)

        X = X.with_columns(*pl.from_numpy(X_mimic, schema=self.schema_))
        return X

class DiagMimic(_BaseMimic):
    """Mimics specifed columns from new dataset to have same means and variances as reference dataset.

    Notes:
        - This estimator can't be fitted within a pipeline or column transformer.
        - All columns will be casted back to original dtype after mimicing, which may be integer. If you don't want to loose any information, cast all columns to float before fitting. Categorical columns are not supported

    Args:
        include: columns to mimic. Set to None to include all columns.
        exclude: columns to exclude, applies after ``include``. Defaults to None.

    """
    def __init__(self, include, exclude=None):
        self.include = include
        self.exclude = exclude

    def _fit(self, X_ref, X_new):
        self.mean_ = X_ref.mean(0)
        self.std_ = X_ref.std(0)

        X_ref = (X_ref - self.mean_) / self.std_
        X_new = (X_new - self.mean_) / self.std_

        self.shift_ = X_ref.mean(0) - X_new.mean(0)
        self.scale_ = X_ref.std(0) / _eps_clip(X_new.std(0))

    def _transform(self, X):
        X = (X - self.mean_) / self.std_
        X = (X * self.scale_) + self.shift_
        X = (X * self.std_) + self.mean_
        return X

class IterativeMimic(_BaseMimic):
    """Mimics specifed columns from new dataset to have same covariance as reference dataset by optimizing a projection matrix, this computes a least squares solution.

    Notes:
        - This estimator can't be fitted within a pipeline or column transformer.
        - All columns will be casted back to original dtype after mimicing, which may be integer. If you don't want to loose any information, cast all columns to float before fitting. Categorical columns are not supported.
        - If ``n_samples << n_features``, this mimics gram matrix which should produce the same resoluts but more efficiently.

    Args:
        include: columns to mimic. Set to None to include all columns.
        exclude: columns to exclude, applies after ``include``. Defaults to None.
        rprop_iters: maximum number of RProp iterations per loop. Defaults to 1000.
        lbfgs_iters: maximum number of L-BFGS iterations per loop. Defaults to 1000.
        n_resets: number of optimization loops. Defaults to 2.
        criterion: loss function for covariance matrices. Defaults to F.mse_loss.
        device: device to perform optimization on. Defaults to CUDA_IF_AVAILABLE.
        dtype: dtype for optimization. Defaults to torch.float32.
    """
    def __init__(
        self,
        include,
        exclude=None,
        rprop_iters=1000,
        lbfgs_iters=1000,
        n_resets: int = 2,
        criterion=F.mse_loss,
        device=CUDA_IF_AVAILABLE,
        dtype=torch.float32,
    ):
        self.include = include
        self.exclude = exclude
        self.device = device
        self.dtype = dtype
        self.criterion = criterion
        self.rprop_iters = rprop_iters
        self.lbfgs_iters = lbfgs_iters
        self.n_resets = n_resets

    def _fit(self, X_ref, X_new):
        self.mean_ = X_ref.mean(0)
        self.std_ = X_ref.std(0).clip(min=np.finfo(X_ref.dtype).eps) # pylint:disable=no-member

        X_ref = (X_ref - self.mean_) / self.std_
        X_new = (X_new - self.mean_) / self.std_

        # computing covariance can be performed on CPU to save VRAM
        X_ref = torch.as_tensor(X_ref, dtype=self.dtype)
        is_XTX = X_ref.shape[0] >= X_ref.shape[1]
        if is_XTX: cov_ref = X_ref.T @ X_ref
        else: cov_ref = X_ref @ X_ref.T

        del X_ref

        cov_ref = cov_ref.to(device=self.device)
        X_new = torch.as_tensor(X_new, device=self.device, dtype=self.dtype)

        W, self.losses_ = mimic_iterative(
            cov_ref=cov_ref, is_XTX=is_XTX, X_new=X_new, criterion=self.criterion,
            rprop_iters=self.rprop_iters, lbfgs_iters=self.lbfgs_iters, n_resets=self.n_resets)

        self.W_ = W.numpy(force=True)

    def _transform(self, X):
        X = (X - self.mean_) / self.std_
        dtype = X.dtype
        X = X.astype(self.W_.dtype) @ self.W_
        X = (X * self.std_) + self.mean_
        return X.astype(dtype)


class EighMimic(_BaseMimic):
    """Mimics specifed columns from new dataset to have same covariance as reference dataset through eigendecomposition.

    Notes:
        - This estimator can't be fitted within a pipeline or column transformer.
        - All columns will be casted back to original dtype after mimicing, which may be integer. If you don't want to loose any information, cast all columns to float before fitting. Categorical columns are not supported
        - ``IterativeMimic`` usually produces better results.

    Args:
        include: columns to mimic. Set to None to include all columns.
        exclude: columns to exclude, applies after ``include``. Defaults to None.
        reg: regularization parameter for stability, clips eigenvalues to be no less than this value. Defaults to 1e-6.
        device: device to fit on. Defaults to CUDA_IF_AVAILABLE.
        dtype: dtype to fit in, float64 recommended. Defaults to torch.float64.
    """
    def __init__(self, include, exclude=None, reg=1-6, device=CUDA_IF_AVAILABLE, dtype=torch.float64):
        self.include = include
        self.exclude = exclude
        self.reg = reg
        self.device = device
        self.dtype = dtype

    def _fit(self, X_ref, X_new):
        self.mean_ = X_ref.mean(0)
        self.std_ = X_ref.std(0).clip(min=np.finfo(X_ref.dtype).eps) # pylint:disable=no-member

        X_ref = (X_ref - self.mean_) / self.std_
        X_new = (X_new - self.mean_) / self.std_

        # computing covariance can be performed on CPU to save VRAM
        X_ref = torch.as_tensor(X_ref, dtype=self.dtype)
        cov_ref = X_ref.T @ X_ref
        del X_ref

        cov_ref = cov_ref.to(device=self.device)
        X_new = torch.as_tensor(X_new, device=self.device, dtype=self.dtype)

        W = mimic_eigh(cov_ref=cov_ref, X_new=X_new, reg=self.reg)
        self.W_ = W.numpy(force=True)

    def _transform(self, X):
        X = (X - self.mean_) / self.std_
        dtype = X.dtype
        X = X.astype(self.W_.dtype) @ self.W_
        X = (X * self.std_) + self.mean_
        return X.astype(dtype)

