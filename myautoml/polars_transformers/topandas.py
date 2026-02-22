import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import polars as pl
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..utils.polars_utils import (
    PolarsColumnSelector,
    include_exclude_cols,
    maybe_stack,
    to_dataframe,
    to_lazyframe,
)
from .bases import PolarsTransformer, _PolarsTransformWrapper
from .ordinal import  OrdinalEncoder

if TYPE_CHECKING:
    import pandas as pd

class ToPandas(PolarsTransformer):
    """Converts the dataframe to pandas.
    Also optionally makes categorical columns integers with categorical dtype for lightgbm."""
    def __init__(
        self,
        int_categorical: bool = False,
    ):
        self.int_categorical = int_categorical

    def fit(self, df):

        if self.int_categorical:
            df = to_lazyframe(df)
            self.cat_cols_ = df.select(pl.selectors.categorical()).collect_schema().names()
            self.ordinal_ = OrdinalEncoder(include=self.cat_cols_, allow_unknown=True).fit(df)
            self.astype_map_ = {c: "category" for c in self.cat_cols_}
            self.inverse_astype_map_ = {c: "string" for c in self.cat_cols_}

        else:
            self.cat_cols_ = self.ordinal_ = self.astype_map_ = self.inverse_astype_map_ = None

        return self

    def transform(self, df) -> "pd.DataFrame":
        self.feature_names_in_ = df.collect_schema().names()

        if self.int_categorical is False:
            return to_dataframe(df).to_pandas()

        assert self.cat_cols_ is not None
        assert self.ordinal_ is not None
        assert self.astype_map_ is not None
        df = self.ordinal_.transform(df)
        df = to_dataframe(df).to_pandas()
        return df.astype(self.astype_map_)

    def inverse_transform(self, df: "pd.DataFrame"):
        if self.int_categorical is False:
            return pl.from_pandas(df)

        assert self.cat_cols_ is not None
        assert self.ordinal_ is not None
        assert self.inverse_astype_map_ is not None
        df = df.astype(self.inverse_astype_map_)
        return self.ordinal_.inverse_transform(df)
