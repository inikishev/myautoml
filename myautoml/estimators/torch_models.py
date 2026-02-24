import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal
from collections import defaultdict
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
from ..utils.light_dataloader import TensorDataLoader
from .torch_embeddings import TorchEmbeddings

CUDA_IF_AVAILABLE = 'cuda' if torch.cuda.is_available() else "cpu"

class _BaseTorchModel(BaseEstimator):
    is_classification: bool

    def __init__(
        self,
        scorer,
        model_cls: Callable[[int, int], nn.Module],
        batch_size: int | None,
        test_batch_size: int | None,
        epochs: int,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        optimizer_cls: Callable,
        scheduler_cls: Callable | None | Literal["one-cycle"],
        max_lr: float,
        test_frac: int | float,
        max_no_improvement_epochs: int,
        emb_dim: int,
        emb_config: dict | None,
        device: torch.Device,
        data_device: torch.Device | Literal["same"],
        verbose: int,
    ):
        self.scorer = scorer
        self.model_cls = model_cls
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls: Callable | None | Literal["one-cycle"] = scheduler_cls
        self.max_lr = max_lr
        self.emb_dim = emb_dim
        self.emb_config = emb_config
        self.test_frac = test_frac
        self.max_no_improvement_epochs = max_no_improvement_epochs
        self.device: torch.Device | Literal["same"] = device
        self.data_device = data_device
        self.verbose = verbose

    def fit(self, X, y):
        validate_data(self, X=X, y=y)
        scorer = get_scorer(self.scorer)

        y = np.asarray(y)

        criterion = self.criterion
        if self.is_classification:
            assert y.ndim == 1
            self.classes_, y = np.unique(y, return_inverse=True)
            if len(self.classes_) > 2:
                out_channels = len(self.classes_)
                if criterion is None: criterion = F.cross_entropy
            else:
                out_channels = 1
                y = y[:, np.newaxis]
                if criterion is None: criterion = F.binary_cross_entropy_with_logits

        else:
            if y.ndim == 1: y = y[:, np.newaxis]
            out_channels = y.shape[-1]

        assert criterion is not None

        emb_config = self.emb_config if self.emb_config is not None else {}
        emb_config.setdefault("emb_dim", self.emb_dim)

        data_device = self.data_device
        if data_device == "same": data_device = self.device
        # we need X_num and X_cat on data_device
        self.embeddings_ = TorchEmbeddings(**emb_config).to(device=data_device).fit(X)

        X_num, X_cat = self.embeddings_.get_inputs(X)

        # ----------------------------- train-test split ----------------------------- #
        test_frac = self.test_frac
        if isinstance(test_frac, float): test_frac = math.ceil(test_frac * X_num.shape[0])
        randperm = torch.randperm(X.shape[0])[:-test_frac]
        train_index = randperm[:-test_frac]
        test_index = randperm[-test_frac:]
        X_num_train = X_num[train_index]
        X_num_test = X_num[test_index]
        y_train = torch.as_tensor(y[train_index], device=data_device)
        y_test = torch.as_tensor(y[test_index], device=data_device)

        if X_cat is not None:
            X_cat_train = X_cat[train_index]
            X_cat_test = X_cat[test_index]
        else:
            X_cat_train = X_cat_test = None


        def make_dataloader(batch_size: int | None, shuffle: bool, *X: torch.Tensor | None):
            tensors = [t for t in X if t is not None]
            if batch_size is None: return [tensors]
            return TensorDataLoader(tensors, batch_size=batch_size, shuffle=shuffle, memory_efficient=True)

        dl_train = make_dataloader(self.batch_size, True, X_num_train, X_cat_train, y_train)
        dl_test = make_dataloader(self.test_batch_size, False, X_num_test, X_cat_test, y_test)
        del X_num, X_cat, X_num_train, X_num_test, X_cat_train, X_cat_test

        # ------------------------ create model and optimizer ------------------------ #
        self.embeddings_ = self.embeddings_.to(self.device)
        self.model_ = self.model_cls(self.embeddings_.out_channels_, out_channels)
        best_state_dict = torch_utils.copy_state_dict(self.model_.state_dict(), device='cpu')
        lowest_error = float("inf")

        optimizer = self.optimizer_cls(self.model_.parameters())

        scheduler_cls = self.scheduler_cls
        if scheduler_cls == 'one-cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, self.max_lr, epochs=self.epochs, steps_per_epoch=len(dl_train))
        else:
            scheduler = None if scheduler_cls is None else scheduler_cls(optimizer)

        train_losses = defaultdict(dict)
        test_losses = defaultdict(dict)

        for epoch in range(self.epochs):
            ...
