from collections.abc import Iterable
from typing import Any

from ..utils.python_utils import flatten
from .bases import PolarsTransformer

class Chain[Return](PolarsTransformer):
    """Chain other transforms"""
    def __init__(self, *stages: PolarsTransformer | Iterable[PolarsTransformer]):
        self.stages = flatten(stages)

    def fit(self, df):
        for i,stage in enumerate(self.stages):
            if i < len(self.stages) - 1: df = stage.fit_transform(df)
            else: stage.fit(df)

        self.feature_names_in_ = self.stages[0].feature_names_in_
        return self

    def transform(self, df) -> Return:
        for stage in self.stages: df = stage.transform(df)
        return df

    def fit_transform(self, df) -> Return:
        for stage in self.stages: df = stage.fit_transform(df)

        self.feature_names_in_ = self.stages[0].feature_names_in_
        return df

    def inverse_transform(self, df):
        for stage in reversed(self.stages):
            if hasattr(stage, "inverse_transform"):
                try: df = getattr(stage, "inverse_transform")(df)
                except (NotImplementedError, AttributeError): pass
        return df
