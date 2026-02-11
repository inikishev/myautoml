from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from typing import Any, TypeAlias, overload

import numpy as np
import polars as pl
from polars._typing import ColumnNameOrSelector

from .python_utils import flatiter
from .torch_utils import TORCH_INSTALLED

PANDAS_INSTALLED = find_spec("pandas") is not None

def to_dataframe(x) -> pl.DataFrame:
    """Helper function to convert ``polars`` and ``pandas`` ``DataFrame``, ``LazyFrame`` and ``Series`` to ``pl.DataFrame``"""

    # Polars types
    if isinstance(x, pl.DataFrame): return x
    if isinstance(x, pl.LazyFrame): return x.collect()
    if isinstance(x, pl.Series): return x.to_frame()

    # Pandas
    if PANDAS_INSTALLED:
        import pandas as pd
        if isinstance(x, pd.DataFrame): return pl.from_pandas(x)
        if isinstance(x, pd.Series): return pl.from_pandas(x).to_frame()

    # Numpy
    if isinstance(x, np.ndarray): return pl.from_numpy(x)

    # Torch
    if TORCH_INSTALLED:
        import torch
        if isinstance(x, torch.Tensor): return pl.from_torch(x, force=True)

    return pl.from_numpy(np.asarray(x))

def to_lazyframe(x) -> pl.LazyFrame:
    """Helper function to convert ``polars`` and ``pandas`` ``DataFrame``, ``LazyFrame`` and ``Series`` to ``pl.LazyFrame``"""

    # Polars types
    if isinstance(x, pl.LazyFrame): return x
    if isinstance(x, pl.DataFrame): return x.lazy()
    if isinstance(x, pl.Series): return x.to_frame().lazy()

    # Pandas
    if PANDAS_INSTALLED:
        import pandas as pd
        if isinstance(x, pd.DataFrame): return pl.from_pandas(x).lazy()
        if isinstance(x, pd.Series): return pl.from_pandas(x).to_frame().lazy()


    # Numpy
    if isinstance(x, np.ndarray): return pl.from_numpy(x).lazy()

    # Torch
    if TORCH_INSTALLED:
        import torch
        if isinstance(x, torch.Tensor): return pl.from_torch(x, force=True).lazy()

    return pl.from_numpy(np.asarray(x)).lazy()


def to_series(x) -> pl.Series:
    if isinstance(x, pl.Series): return x
    if isinstance(x, pl.DataFrame): return x.to_series()
    if isinstance(x, pl.LazyFrame): return x.collect().to_series()
    if isinstance(x, np.ndarray):
        if np.squeeze(x).ndim > 1: raise RuntimeError(f"Can't convert x of shape {x.shape} to Series.")
        return pl.Series("ndarray", np.squeeze(x))
    if PANDAS_INSTALLED:
        import pandas as pd
        if isinstance(x, pd.DataFrame): return pl.from_pandas(x).to_series()
        if isinstance(x, pd.Series): return pl.from_pandas(x)
    if TORCH_INSTALLED:
        import torch
        if isinstance(x, torch.Tensor):
            if torch.squeeze(x).ndim > 1: raise RuntimeError(f"Can't convert x of shape {x.shape} to Series.")
            return pl.Series("tensor", torch.squeeze(x).numpy(force=True))
    x = np.asarray(x)
    if np.squeeze(x).ndim > 1: raise RuntimeError(f"Can't convert x of shape {x.shape} to Series.")
    return pl.Series(x.__class__.__name__, np.squeeze(x))


def maybe_stack[T: pl.DataFrame | pl.LazyFrame](*items: T | None) -> T | None:
    dfs = [df for df in items if df is not None]
    if len(dfs) == 0: return None
    return pl.concat(dfs)

PolarsColumnSelector = ColumnNameOrSelector | Sequence[ColumnNameOrSelector]
"""Type for valid args in ``pl.with_columns``, ``pl.select``, ``pl.drop``"""

@overload
def include_exclude_cols(
    df: pl.DataFrame,
    include: PolarsColumnSelector | None,
    exclude: PolarsColumnSelector | None
) -> pl.DataFrame: ...
@overload
def include_exclude_cols(
    df: pl.LazyFrame,
    include: PolarsColumnSelector | None,
    exclude: PolarsColumnSelector | None
) -> pl.LazyFrame: ...
def include_exclude_cols(
    df: pl.DataFrame | pl.LazyFrame,
    include: PolarsColumnSelector | None,
    exclude: PolarsColumnSelector | None
):
    """Helper function to include and exclude specified columns, used in many transforms"""
    if include is not None: df = df.select(include)
    if exclude is not None: df = df.drop(exclude, strict=False)
    return df

def with_columns_nonstrict(df: pl.LazyFrame, exprs: Mapping[str, pl.Expr | Sequence[pl.Expr]], names=None):
    """Adds all ``exprs`` whose keys are columns in ``df``."""
    if names is None: names = df.collect_schema().names()
    return df.with_columns(flatiter(v for k,v in exprs.items() if k in names))
