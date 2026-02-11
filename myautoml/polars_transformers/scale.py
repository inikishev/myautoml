import polars as pl

from ..utils.polars_utils import (
    PolarsColumnSelector,
    include_exclude_cols,
    to_lazyframe,
    with_columns_nonstrict,
)
from .bases import PolarsTransformer


class StandardScaler(PolarsTransformer):
    """Scales each column to have zero mean and unit variance.

    Args:
        include: Columns to process. None means all numeric columns are processed. Defaults to None.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
        with_mean: whether to center the data. Defaults to True.
        with_std: whether to scale data by standard deviation. Defaults to True.
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
        with_mean: bool = True,
        with_std: bool = True,
    ):
        self.include = include
        self.exclude = exclude
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        df = df.select(pl.selectors.numeric())

        self.mean_ = df.fill_nan(None).mean().collect().to_dicts()[0]
        self.std_ = df.fill_nan(None).std().with_columns(pl.all().clip(lower_bound=1e-16)).collect().to_dicts()[0]

        self.expr_ = {}
        self.inv_expr_ = {}

        for col, mean in self.mean_.items():
            expr = pl.col(col)
            inv_expr = pl.col(col)

            if self.with_mean:
                expr = expr.sub(mean)

            if self.with_std:
                std = self.std_[col]
                expr = expr.truediv(std)
                inv_expr = inv_expr.mul(std)

            if self.with_mean:
                inv_expr = inv_expr.add(mean)

            self.expr_[col] = expr
            self.inv_expr_[col] = inv_expr

        return self

    def transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.expr_)

    def inverse_transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.inv_expr_)

class MinMaxScaler(PolarsTransformer):
    """Scales each column to have values in a range defined by ``feature_range``.

    Args:
        include: Columns to process. None means all columns are processed. Defaults to None.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
        feature_range: range of values. Defaults to (0, 1).
        clip: if True, ``transform`` will clip columns to ``feature_range`` after scaling.
            That means ``inverse_transform`` might not recover original values. Defaults to False.
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
        feature_range=(0, 1),
        clip=False,
    ):
        self.include = include
        self.exclude = exclude
        self.feature_range = feature_range
        self.clip = clip

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        df = df.select(pl.selectors.numeric())

        self.min_ = df.fill_nan(None).min().collect().to_dicts()[0]
        self.max_ = df.fill_nan(None).max().collect().to_dicts()[0]

        self.expr_ = {}
        self.inv_expr_ = {}

        for col, vmin in self.min_.items():
            vmax = self.max_[col]
            denom = max(vmax - vmin, 1e-16)

            expr = pl.col(col).sub(vmin).truediv(denom)
            inv_expr = pl.col(col)

            if self.feature_range != (0, 1):
                lb, ub = self.feature_range
                expr = expr.mul(ub - lb).add(lb)
                inv_expr = inv_expr.sub(lb).truediv(ub - lb)

            inv_expr = inv_expr.mul(denom).add(vmin)

            if self.clip:
                expr = expr.clip(*self.feature_range)

            self.expr_[col] = expr
            self.inv_expr_[col] = inv_expr

        return self

    def transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.expr_)

    def inverse_transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.inv_expr_)


class SampleNormalizer(PolarsTransformer):
    """Scales each row to have zero mean and unit variance.

    Args:
        include: Columns to process. None means all columns are processed. Defaults to None.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
        with_mean: whether to center the data. Defaults to True.
        with_std: whether to scale data by standard deviation. Defaults to True.
    """

    def __init__(
        self,
        include: PolarsColumnSelector | None = None,
        exclude: PolarsColumnSelector | None = None,
        with_mean: bool = True,
        with_std: bool = True,
    ):
        self.include = include
        self.exclude = exclude
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        df = df.select(pl.selectors.numeric())
        names = df.collect_schema().names()

        # the expression should be strictly with same columns as during fit, to get same statistics
        # so we collect schema here and don't use with_columns_nonstrict
        list_expr = pl.concat_list(names)
        self.expr_ = pl.col(names)

        if self.with_mean:
            self.expr_ = self.expr_.sub(list_expr.list.mean())

        if self.with_std:
            self.expr_ = self.expr_.truediv(list_expr.list.std().clip(1e-16))

        return self

    def transform(self, df) -> pl.LazyFrame:
        return to_lazyframe(df).with_columns(self.expr_)
