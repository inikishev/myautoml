from collections.abc import Mapping, Sequence
from typing import Literal, Never

import polars as pl

from ..utils.polars_utils import (
    PolarsColumnSelector,
    include_exclude_cols,
    to_dataframe,
    to_lazyframe,
)
from ..utils.python_utils import flatten
from .bases import PolarsTransformer, _get_feature_names


class SelectCols(PolarsTransformer):
    """Selects specified columns."""
    def __init__(self, *cols: PolarsColumnSelector):
        self.cols = cols

    def fit(self, df):
        self.feature_names_in_ = _get_feature_names(df)
        return self

    def transform(self, df) -> pl.LazyFrame:
        return include_exclude_cols(to_lazyframe(df), include=flatten(self.cols), exclude=None)

class DropCols(PolarsTransformer):
    """Drops specified columns."""
    def __init__(self, *cols: PolarsColumnSelector):
        self.cols = cols

    def fit(self, df):
        self.feature_names_in_ = _get_feature_names(df)
        return self

    def transform(self, df) -> pl.LazyFrame:
        return include_exclude_cols(to_lazyframe(df), include=None, exclude=flatten(self.cols))

class Cast(PolarsTransformer):
    def __init__(
        self,
        include: PolarsColumnSelector | None,
        dtype: pl.DataType | type[pl.DataType] | pl.DataTypeExpr,
        exclude: PolarsColumnSelector | None = None,
    ):
        self.include = include
        self.dtype = dtype
        self.exclude = exclude

    def fit(self, df):
        self.feature_names_in_ = _get_feature_names(df)
        return self

    @pl.StringCache()
    def transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        cols = include_exclude_cols(df, include=self.include, exclude=self.exclude).cast(self.dtype)
        df = df.with_columns(*cols.collect())
        return df

class Collect(PolarsTransformer):

    def fit(self, df):
        self.feature_names_in_ = _get_feature_names(df)
        return self

    def transform(self, df):
        return to_dataframe(df)

class DropConstant(PolarsTransformer):
    """Drops columns with a single unique value.
    This also defines an inverse transform which adds dropped columns back filled with their constant values.

    Args:
        include: Columns to process. None means all columns are processed. Defaults to None.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
    ):
        self.include = include
        self.exclude = exclude

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        n_unique = df.select(pl.all().n_unique()).collect().to_dicts()[0]

        self.drop_cols_ = [k for k,v in n_unique.items() if v == 1]
        if len(self.drop_cols_) == 0: return self

        drop_vals = df.select(self.drop_cols_).first().collect()
        self.drop_vals_ = drop_vals.to_dicts()[0]
        self.dtypes_ = drop_vals.schema
        self.order_ = tuple(n_unique.keys())

        return self

    def transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if len(self.drop_cols_) == 0: return df
        return df.drop(self.drop_cols_, strict=False)

    def inverse_transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if len(self.drop_cols_) == 0: return df
        df = df.with_columns(pl.lit(v, dtype=self.dtypes_[c]).alias(c) for c, v in self.drop_vals_.items())
        return df.select(self.order_)

class RemoveDuplicates(PolarsTransformer):
    """Removes duplicate rows. This simply calls ``df.unique()`` and has the same arguments.
    Note that by default it doesn't maintain order, so the output will have a different order of rows."""

    def __init__(
        self,
        subset: PolarsColumnSelector | None = None,
        keep: Literal["first", "last", "any", "none"] = "any",
        maintain_order: bool = False,
    ):
        self.subset = subset
        self.keep: Literal["first", "last", "any", "none"] = keep
        self.maintain_order = maintain_order

    def fit(self, df):
        self.feature_names_in_ = _get_feature_names(df)
        return self

    def transform(self, df) -> pl.LazyFrame:
        return to_lazyframe(df).unique(subset=self.subset, keep=self.keep, maintain_order=self.maintain_order)