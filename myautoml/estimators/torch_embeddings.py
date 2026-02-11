"""Embeddings from categorical features"""
import math
from functools import partial

import polars as pl
import torch
from torch import nn

from ..polars_transformers.auto_encoder import AutoEncoder
from ..polars_transformers.ordinal import OrdinalEncoder
from ..utils.polars_utils import to_dataframe


def _clip(x, min, max):
    if min is not None and x < min: return min
    if max is not None and x > max: return max
    return x

class TorchEmbeddings(nn.Module):
    """Learnable embeddings for categorical features. If there are no categorical features, this acts as identity.

    Args:
        df: dataframe without the target column.
        out_dim: target total dimension of embedded features.
            May be less or more if ``min_dim``, ``max_dim`` or ``max_params`` can't be satisfied.
        min_dim: minimal embedding dim (unless ``max_params`` can not be satisfied). Defaults to 2.
        max_dim: maximal embedding dim per embedding. Defaults to 1024.
        max_params: maximal number of total parameters on all embeddings. Defaults to 1_000_000.
        other: kwargs for nn embedding

    Example:

    ```python
    embeddings = TorchEmbeddings(100).cuda().fit(df)
    X_num, X_cat = embeddings.get_inputs(df)
    model = nn.Linear(embeddings.out_channels_, 10)

    for _ in range(100):
        X = embeddings(X_num, X_cat)
        loss = model(X)
        ...

    X = embeddings.transform(df_test)
    preds = model(X)
    ```
    """

    def __init__(
        self,
        out_dim: int,
        min_dim: int | None = 2,
        max_dim: int | None = 1024,
        max_params: int | None = 1_000_000,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.max_params = max_params
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

    def fit(self, df):
        """Make sure ``df`` doesn't contain the label."""
        df = to_dataframe(df)
        self.encoder_ = AutoEncoder().fit(df, y=None)
        df = self.encoder_.transform_X(df)

        # Process the dataframe
        # Extract categorical and numeric cols
        cat_cols = df.select(pl.selectors.categorical())
        num_cols = df.select([c for c in df.columns if c not in cat_cols.columns])
        self.noop_ = (len(cat_cols.columns) == 0)
        if self.noop_: return self

        self.ordinal_ = OrdinalEncoder(include=cat_cols.columns, allow_unknown=True).fit(df)

        # Distribute embedding dims among categorical cols
        col_to_num = cat_cols.select(pl.all().n_unique()).to_dicts()[0]
        total_num = sum(col_to_num.values())
        ratio = self.out_dim / total_num

        emb_dims = {
            col: _clip(math.ceil(num * ratio), self.min_dim, self.max_dim)
            for col, num in col_to_num.items()
        }
        if (self.max_params is not None) and (sum(emb_dims.values()) > self.max_params):
            ratio = sum(emb_dims.values()) / self.max_params
            emb_dims = {col: math.ceil(dim * ratio) for col, dim in emb_dims.items()}

        # Create embeddings
        Emb = partial(
            nn.Embedding,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        self.embeddings_ = nn.ModuleList(Emb(col_to_num[col], dim) for col, dim in emb_dims.items())

        self.out_channels_ = sum(emb_dims.values()) + len(num_cols.columns)
        self.cat_cols_ = cat_cols.columns.copy()
        self.num_cols_ = num_cols.columns.copy()

        return self

    def get_inputs(self, df) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns ``(X_num, X_cat)``. ``X_cat`` is either an integer tensor or None."""
        _ = next(iter(self.parameters()))
        device = _.device
        dtype = _.dtype

        df = self.encoder_.transform_X(df)
        df = self.ordinal_.transform(df).collect()
        num_cols = df.select(self.num_cols_)
        X_num = num_cols.to_torch(return_type='tensor', dtype=pl.Float32).to(device=device, dtype=dtype)
        if self.noop_: return X_num, None

        cat_cols = df.select(self.cat_cols_)
        X_cat = cat_cols.to_torch(return_type='tensor', dtype=pl.Int64).to(device=device)

        return X_num, X_cat

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor | None = None):
        """Passes ``X_cat`` through embeddings and returns the concatenated tensor."""
        if isinstance(X_num, tuple): # compatibility with Sequential
            assert X_cat is None
            X_num, X_cat = X_num

        if X_cat is None: return X_num
        return torch.cat([emb(t) for emb, t in zip(self.embeddings_, X_cat.unbind(1))] + [X_num], 1)

    def transform(self, df):
        """returns ``self(*self.get_inputs())``."""
        X_num, X_cat = self.get_inputs(df)
        return self(X_num, X_cat)