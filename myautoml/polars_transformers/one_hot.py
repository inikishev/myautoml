from collections.abc import Mapping, Sequence
from typing import Any, Literal

import polars as pl

from ..utils.polars_utils import (
    PolarsColumnSelector,
    include_exclude_cols,
    to_lazyframe,
    with_columns_nonstrict,
)
from ..utils.python_utils import inner_reorder
from .bases import PolarsTransformer
from .infrequent import MergeInfrequent


def _one_hot_expr(
    col_name: str,
    categories: list[Any],
    propagate_nulls: bool,
    drop_first: Literal["all", "binary", "none"],
    dtype,
):
    """Expressions that generate one-hot columns from a categorical column"""
    if (drop_first == 'all') or (drop_first == "binary" and len(categories) == 2):
        categories = categories[1:]

    if propagate_nulls:
        return [
            # In polars comparing with None always results in null
            # so whenever category is None, all cols become null
            (pl.col(col_name).eq(cat)).cast(dtype).alias(f"{col_name}__{cat}")
            for cat in categories if cat is not None
        ]

    return [
        pl.col(col_name).eq_missing(cat).cast(dtype).alias(f"{col_name}__{cat}")
        for cat in categories
    ]

def _inverse_expr(
    col_name: str,
    categories: list[Any],
    one_hot_col_names: list[str], # list of f"{col_name}__{category}", includes dropped
    drop_first: Literal["all", "binary", "none"],
    dtype: pl.DataType,
) -> pl.Expr:
    """Expression to turn one-hot encoded columns for a feature back into initial column"""

    if (drop_first == 'all') or (drop_first == "binary" and len(categories) == 2):

        return (
            # If all one-hot columns are 0, dropped column is assumed to be 1
            pl.when(pl.max_horizontal(one_hot_col_names[1:]) == 0).then(pl.lit(categories[0]))

            # Otherwise use argmax
            .otherwise(
                pl.lit(categories[1:], dtype=pl.List(dtype))
                .list.get(pl.concat_arr(one_hot_col_names[1:]).arr.arg_max())
            )

        ).alias(col_name)

    return (
        # List of categories
        pl.lit(categories, dtype=pl.List(dtype))

         # Get category with index corresponding to argmax of one-hot columns
        .list.get(pl.concat_arr(one_hot_col_names).arr.arg_max())

    ).alias(col_name)


class OneHotEncoder(PolarsTransformer):
    """Applies one-hot encoding to each feature. Inverse transform takes argmax of one-hotted columns.

    The names of one-hot encoded columns are ``f"{col_name}__{category}"``.

    Args:
        include: Columns to process. None means all columns are processed.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
        drop_first: Whether to drop first category per feature to prevent multi-colinearity.
            - ``"all"`` drops first column in all one-hot encoded columns.
            - ``"binary"`` drops first column in binary features only, making them ordinal (default).
            - ``"none"`` - doesn't drop first columns.

            Note that inverse transform reconstructs dropped columns by assuming a zero-vector maps to the dropped category,
            that means ``inverse_transform`` should not be applied to logits/probabilities unless ``drop_first="none"``.
        min_frequency: features with less than this frequency will be bundled into one.
            If float, relative to total number of samples. Defaults to None.
        max_categories: if specified, one-hot encodes top ``max_categories``
            most common categories and the rest are bundled into one. Defaults to None.
        propagate_nulls: if True, whenever category is ``null``, all one-hot encoded cols become ``null``.
            if False, nulls are treated as another category.
        maintain_order: If False, the relative order of the generated one-hot columns may vary between runs,
            but can make this faster. This will also affect which column is dropped by ``drop_first``.
        infrequent_value: suffix for infrequent columns, and value that inverse transform maps them to.
            Defaults to "__infrequent".
        dtype: Data type for one-hot encoded columns. Defaults to pl.Boolean.

    Notes:
        - This transform does not affect constant columns (columns with a single category), they are always passed through unchanged.
        - ``transform`` currently maps unseen categories to all-zero columns.
    """
    def __init__(
        self,
        include: PolarsColumnSelector | None,
        exclude: PolarsColumnSelector | None = None,
        drop_first: Literal["all", "binary", "none"] = "binary",
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
        propagate_nulls: bool = True,
        maintain_order: bool = True,
        infrequent_value: Any = "__infrequent",
        dtype: pl.DataType | type[pl.DataType] | pl.DataTypeExpr = pl.Boolean,
    ):
        self.include = include
        self.exclude = exclude

        self.drop_first: Literal["all", "binary", "none"] = drop_first
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.propagate_nulls = propagate_nulls
        self.maintain_order = maintain_order
        self.infrequent_value = infrequent_value
        self.dtype = dtype

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()

        # Merge infrequent via MergeInfrequent transform
        if (self.min_frequency is not None) or (self.max_categories is not None):
            self.merge_infrequent_ = MergeInfrequent(
                include = self.include,
                exclude = self.exclude,
                min_frequency = self.min_frequency,
                max_categories = self.max_categories,
                propagate_nulls = self.propagate_nulls,
                infrequent_value = self.infrequent_value,
            )
            df = self.merge_infrequent_.fit_transform(df)

        # Collect column order and dtypes for inverse transform, after merging infrequent
        full_schema = df.collect_schema()
        self.order_ = full_schema.names()
        if len(self.order_) == 0: return self

        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        unique = df.select(pl.all().unique(maintain_order=self.maintain_order).implode()).collect()
        if len(unique.columns) == 0: # this means empty selectors
            self.order_.clear() # make order empty to make this transform a no-op
            return self

        col_to_categories = {k:v for k,v in unique.to_dicts()[0].items() if len(v) > 1}
        self.drop_cols_ = tuple(col_to_categories.keys())

        # transform expressions
        self.exprs_ = {
            col_name: _one_hot_expr(
                col_name = col_name,
                categories = categories,
                propagate_nulls = self.propagate_nulls,
                drop_first = self.drop_first,
                dtype=self.dtype,
            )
            for col_name, categories in col_to_categories.items()
        }

        # inverse expressions
        # if propagate_nulls, we do not encode nulls, so inverse should skip them
        if self.propagate_nulls:
            col_to_categories = {k: [cat for cat in v if cat is not None] for k,v in col_to_categories.items()}

        self.inv_exprs_ = {
            f"{col_name}__{categories[1]}": _inverse_expr(
                col_name = col_name,
                categories = categories,
                one_hot_col_names = [f"{col_name}__{cat}" for cat in categories],
                drop_first = self.drop_first,
                dtype = full_schema[col_name], # pyright:ignore[reportAttributeAccessIssue]
            )
            for col_name, categories in col_to_categories.items()
        }

        self.inv_drop_cols_ = [f"{col}__{cat}" for col, categories in col_to_categories.items() for cat in categories]

        return self

    def transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if len(self.order_) == 0: return df

        if (self.min_frequency is not None) or (self.max_categories is not None):
            df = self.merge_infrequent_.transform(df)

        return with_columns_nonstrict(df, self.exprs_).drop(self.drop_cols_, strict=False)

    def inverse_transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if len(self.order_) == 0: return df

        df = with_columns_nonstrict(df, self.inv_exprs_).drop(self.inv_drop_cols_, strict=False)

        columns = df.collect_schema().names()
        return df.select(inner_reorder(columns, self.order_))