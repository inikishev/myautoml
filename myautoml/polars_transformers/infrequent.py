from collections.abc import Mapping, Sequence
from typing import Any, Literal

import polars as pl

from ..utils.polars_utils import (
    PolarsColumnSelector,
    include_exclude_cols,
    to_lazyframe,
    with_columns_nonstrict,
)
from .bases import PolarsTransformer


class MergeInfrequent(PolarsTransformer):
    """In each column, maps infrequent categories into one feature.

    Args:
        include: Columns to process. None means all columns are processed. Defaults to None.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
        min_frequency: categories with less than this frequency will be considered infrequent.
            If float, relative to total number of samples.
        max_categories: categories other than top-``max_categories`` will be considered infrequent.
        propagate_nulls: if True, null always maps to null and is not considered for frequency computations.
            Otherwise it is treated as a separate category. Defaults to True.
        infrequent_value: Value to replace infrequent categories by. Defaults to "__infrequent".
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None,
        exclude: PolarsColumnSelector | None = None,
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
        propagate_nulls: bool = True,
        infrequent_value: Any = "__infrequent"
    ):
        self.include = include
        self.exclude = exclude
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.propagate_nulls = propagate_nulls
        self.infrequent_value = infrequent_value

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()

        self.noop_ = (self.min_frequency is None) and (self.max_categories is None)
        if self.noop_: return self

        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        sort = (self.max_categories is not None)
        counts = df.select(pl.all().value_counts(parallel=True, sort=sort).implode()).collect()

        if len(counts.columns) == 0:
            self.noop_ = True
            return self

        self.exprs_ = {}
        for col_name, categories in counts.to_dicts()[0].items():

            # here `categories` is like this:
            # [{col_name: category1_name, 'count': 513}, {col_name: category2_name, 'count': 141}]
            # change to a simpler dict {category: count}
            if self.propagate_nulls:
                categories = {cat[col_name]: cat["count"] for cat in categories if cat is not None}
            else:
                categories = {cat[col_name]: cat["count"] for cat in categories}

            frequent_cats = list(categories.keys())
            infrequent_cats = set()

            # Filter based on min_frequency
            min_frequency = self.min_frequency
            if min_frequency is not None:
                if isinstance(min_frequency, float): min_frequency = min_frequency * sum(categories.values())
                infrequent_cats = {k for k, v in categories.items() if v < min_frequency}

            # Filter based on max_categories
            if self.max_categories is not None:
                if len(categories) > self.max_categories:
                    # one_hot is sorted in descending if max_categories is defined
                    infrequent_cats.update(frequent_cats[self.max_categories:])

            if len(infrequent_cats) == 0:
                continue

            frequent_cats = [c for c in frequent_cats if c not in infrequent_cats]

            col = pl.col(col_name)
            expr = (
                # if nulls_equal=False, is_in for null returns null
                pl.when(col.is_in(infrequent_cats, nulls_equal=(not self.propagate_nulls)))
                .then(pl.lit(self.infrequent_value))
                .otherwise(col)
            )

            self.exprs_[col_name] = expr.alias(col_name)

        return self


    def transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if self.noop_: return df
        return with_columns_nonstrict(df, self.exprs_)