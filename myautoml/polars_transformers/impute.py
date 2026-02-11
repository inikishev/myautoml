from collections.abc import Mapping, Sequence
from typing import Any, Literal, Never

import polars as pl
from polars._typing import PolarsDataType

from ..utils.polars_utils import include_exclude_cols, to_lazyframe, PolarsColumnSelector, with_columns_nonstrict
from .bases import PolarsTransformer

class MissingIndicator(PolarsTransformer):
    """Adds columns indicating whether a feature is missing or not to all features with missing values.

    Args:
        include: Columns to process. None means all columns are processed. Defaults to None.
        exclude: Columns to skip, applies after ``include``. None means no columns are skipped. Defaults to None.
        suffix: string to append to names of new indicator columns. Defaults to "__is_missing".
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
        suffix: str = "__is_missing"
    ):
        self.include = include
        self.exclude = exclude
        self.suffix = suffix

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        has_missing = df.select(pl.all().has_nulls()).collect().to_dicts()[0]
        self.cols_ = [col for col, has in has_missing.items() if has]
        self.drop_cols_ = [f"{col}{self.suffix}" for col in self.cols_]
        self.expr_ = pl.col(self.cols_).is_null().cast(pl.UInt8).name.suffix(self.suffix)

        return self

    def transform(self, df) -> pl.LazyFrame:
        return to_lazyframe(df).with_columns(self.expr_)

    def inverse_transform(self, df) -> pl.LazyFrame:
        return to_lazyframe(df).drop(self.drop_cols_, strict=False)

class MissingStatistics(PolarsTransformer):
    """Adds a column indicating the fraction of missing values.

    Args:
        include: Columns to process. None means all columns are processed. Defaults to None.
        exclude: Columns to skip, applies after ``include``. None means no columns are skipped. Defaults to None.
        col_name: name of the new column. Defaults to "frac_missing_columns".
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
        col_name: str = "frac_missing_columns",
    ):
        self.include = include
        self.exclude = exclude
        self.col_name = col_name

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        names = df.collect_schema().names()

        # the expression should be strictly with same columns as during fit, to get same statistics
        # so we collect schema here and don't use with_columns_nonstrict
        self.expr_ = pl.mean_horizontal(pl.col(names).is_null()).alias(self.col_name)
        return self

    def transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        df = df.with_columns(self.expr_)
        return df

    def inverse_transform(self, df) -> pl.LazyFrame:
        return to_lazyframe(df).drop(self.col_name, strict=False)


class SimpleImputer(PolarsTransformer):
    """Imputes missing values using ``strategy``. Each column is treated separately.
    Categorical values use ``"mode"`` strategy unless ``strategy="constant"``.

    Args:
        include: Columns to process. None means all columns are processed. Defaults to None.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
        strategy: imputation strategy. Defaults to 'median'.
        fill_value: fill value for when ``strategy="constant"``. Defaults to 0.
        add_indicator: if True, adds indicator features indicating whether a value was missing.
            This is a shorthand for ``MissingIndicator`` transform. Defaults to False.

    Notes:
        If all values in a column are null, it will only be imputed with ``"constant"`` strategy.
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
        strategy: Literal["min", "max", "mean", "median", "mode", "constant"] = 'median',
        fill_value: Any = 0,
        add_indicator: bool = False,
    ):
        self.include = include
        self.exclude = exclude
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        # Fit sub-transforms
        if self.add_indicator:
            self.indicator_ = MissingIndicator(include=self.include, exclude=self.exclude).fit(df)

        # For constant strategy we don't need any statistics
        if self.strategy == 'constant':
            self.exprs_ = {col: pl.col(col).fill_null(self.fill_value) for col in df.collect_schema().keys()}
            return self

        # Create expressions to get stats
        if self.strategy in ('min', 'max', 'mean', 'median'):
            expr = (
                getattr(pl.selectors.numeric(), self.strategy)(), # get stat out of numeric cols
                (~pl.selectors.numeric()).drop_nulls().mode().first(), # mode of non-numeric cols excluding nulls
            )

        elif self.strategy == 'mode':
            expr = pl.selectors.all().mode().drop_nulls().first() # mode of all cols

        else:
            raise RuntimeError(f'Unknown strategy "{self.strategy}", must be one of: '
                                    '["min", "max", "mean", "median", "mode", "constant"]')

        fill_vals = df.select(expr).collect().to_dicts()[0]

        # all-null cols should be skipped because expr.fill_null(None) raises an exception
        # since None is the default value that indicates that full_null(strategy=...) is specified
        self.exprs_ = {k: pl.col(k).fill_null(v) for k,v in fill_vals.items() if v is not None}
        return self

    def transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if self.add_indicator: df = self.indicator_.transform(df)
        return with_columns_nonstrict(df, self.exprs_)

    def inverse_transform(self, df) -> pl.LazyFrame: # inverse transform is only to remove indicators
        df = to_lazyframe(df)
        if self.add_indicator: df = self.indicator_.inverse_transform(df)
        return df