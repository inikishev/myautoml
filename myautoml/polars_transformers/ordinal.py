from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Literal

import polars as pl

from ..utils.polars_utils import to_lazyframe, include_exclude_cols, with_columns_nonstrict
from .bases import PolarsTransformer

class OrdinalEncoder(PolarsTransformer):
    """Applies ordinal encoding to each column, replacing it with integers from ``0`` to ``n_categories-1``.

    If you want control over what categories map into what integers, use ``MapEncoder``.

    Args:
        include: Columns to process. None means all columns are processed.
        exclude: Columns to not process and pass through as is, applies after ``include``.
            None means no columns are skipped. Defaults to None.
        allow_unknown: if True, unknown values will be mapped to null. Otherwise an error will be raised.
        propagate_nulls: if True, null values remain nulls. If False, they are treated as another category.
        maintain_order: Setting this to False makes this transform non-deterministic,
            i.e. ``fit`` method will produce different orders, but can make this faster.
        dtype: Data type of ordinally encoded columns. Defaults to pl.Uint64.
    """
    def __init__(
        self,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None = None,
        allow_unknown: bool = False,
        propagate_nulls: bool = True,
        maintain_order: bool = True,
        dtype: pl.DataType | type[pl.DataType] | pl.DataTypeExpr = pl.UInt64
    ):
        self.include = include
        self.exclude = exclude
        self.allow_unknown = allow_unknown
        self.propagate_nulls = propagate_nulls
        self.maintain_order = maintain_order
        self.dtype = dtype

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        self.maps_: dict[str, dict[Any, int | None]] = {}
        self.inverse_maps_: dict[str, dict[int | None, Any]] = {}

        unique = df.select(pl.all().unique(maintain_order=self.maintain_order).implode()).collect()

        # Create mappings from values to their ordinal encodings
        for col_name, unique_vals in unique.to_dicts()[0].items():

            if self.propagate_nulls and (None in unique_vals):
                unique_vals.remove(None) # null won't be mapped to integer

            self.maps_[col_name] = {v: i for i, v in enumerate(unique_vals)}

            if self.propagate_nulls:
                self.maps_[col_name][None] = None # instead null is mapped to null, for inverse map too.

            self.inverse_maps_[col_name] = {i: v for v, i in self.maps_[col_name].items()}

        # Create transform expressions
        kw = {"default": None} if self.allow_unknown else {}
        self.exprs_ = {col_name:
            pl.col(col_name).replace_strict(map, return_dtype=self.dtype, **kw)
            for col_name, map in self.maps_.items()
        }

        self.inv_exprs_ = {col_name:
            pl.col(col_name)
            .replace_strict(inv_map, return_dtype=unique.schema[col_name].inner) # pyright:ignore[reportAttributeAccessIssue]
            for col_name, inv_map in self.inverse_maps_.items()
        }

        return self

    def transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.exprs_)

    def inverse_transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.inv_exprs_)

class MapEncoder(PolarsTransformer):
    """Replaces elements according to a specified mapping, this is useful for ordinal encoding with specific order.

    Args:
        map: Dictionary with column names as keys, and replacement mappings as values:
            ``{column_name: {value: replace_value}}``.
        unknown_strategy: how to handle values not in map.
    """
    def __init__(
        self,
        map: Mapping[str, Mapping[Any, Any]],
        unknown_strategy: Literal['passthrough', 'null', 'raise'] = 'passthrough',
        return_dtype = None
    ):
        self.map = map
        self.unknown_strategy: Literal['passthrough', 'null', 'raise'] = unknown_strategy
        self.return_dtype = return_dtype

    def fit(self, df):
        df = to_lazyframe(df)
        schema = df.collect_schema()
        self.feature_names_in_ = schema.names()

        self.maps_ = {k: dict(v) for k,v in self.map.items()} # this avoids mutating self.map
        self.inverse_maps_ = {col_name: {v: k for k, v in map.items()} for col_name, map in self.maps_.items()}

        # Create transform expressions
        if self.unknown_strategy in ('null', 'raise'):
            kw = {"default": None} if self.unknown_strategy == 'null' else {}
            self.exprs_ = {col_name:
                pl.col(col_name).replace_strict(map, return_dtype=self.return_dtype, **kw)
                for col_name, map in self.maps_.items()
            }

        elif self.unknown_strategy == 'passthrough':
            self.exprs_ = {col_name:
                pl.col(col_name).replace(map, return_dtype=self.return_dtype)
                for col_name, map in self.maps_.items()
            }

        else:
            raise RuntimeError(f'Unknown unknown_strategy "{self.unknown_strategy}"')



        try:
            self.inv_exprs_ = {col_name:
                pl.col(col_name).replace_strict(inv_map, return_dtype=schema[col_name])
                for col_name, inv_map in self.inverse_maps_.items()
            }
        except KeyError as e:
            raise KeyError(
                f'Column "{e.args[0]}" in map passed to MapEncoder is not '
                'present in dataframe passed to MapEncoder.fit method.'
            ) from None

        return self

    def transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.exprs_)

    def inverse_transform(self, df) -> pl.LazyFrame:
        return with_columns_nonstrict(to_lazyframe(df), self.inv_exprs_)


class BinaryToBool(PolarsTransformer):
    """Converts binary features to 0 or 1."""
    def __init__(
        self,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
        allow_unknown: bool = False,
        dtype: pl.DataType | type[pl.DataType] | pl.DataTypeExpr = pl.Boolean
    ):
        self.include = include
        self.exclude = exclude
        self.allow_unknown = allow_unknown
        self.dtype = dtype

    def fit(self, df):
        df = to_lazyframe(df)
        self.feature_names_in_ = df.collect_schema().names()
        df = include_exclude_cols(df, include=self.include, exclude=self.exclude)

        n_unique = df.select(pl.all().n_unique()).collect().to_dicts()[0]

        binary_cols = [k for k,v in n_unique.items() if v == 2]
        if len(binary_cols) == 0:
            self.encoder_ = None
        else:
            self.encoder_ = OrdinalEncoder(
                include=binary_cols,
                allow_unknown=self.allow_unknown,
                dtype=self.dtype,
            ).fit(df)
        return self

    def transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if self.encoder_ is None: return df
        return self.encoder_.transform(df)

    def inverse_transform(self, df) -> pl.LazyFrame:
        df = to_lazyframe(df)
        if self.encoder_ is None: return df
        return self.encoder_.inverse_transform(df)