from collections import UserDict
from typing import Any
import numpy as np

class DictLogger(UserDict):
    def log(self, step: int, metric: str, value: Any):
        if metric not in self: self[metric] = {step: value}
        else: self[metric][step] = value

    def to_list(self, key: str):
        return list(self[key].values())

    def steps_list(self, key: str):
        return list(self[key].keys())

    def to_numpy(self, key: str):
        return np.asarray(self.to_list(key))

    def first(self, key):
        return next(iter(self[key].values()))

    def last(self, key):
        return self.to_list(key)[-1]

    def min(self, key):
        return self.to_numpy(key).min()

    def max(self, key):
        return self.to_numpy(key).max()

    def nanmin(self, key):
        return np.nanmin(self.to_numpy(key))

    def nanmax(self, key):
        return np.nanmax(self.to_numpy(key))

    def argmin(self, key):
        return self.steps_list(key)[np.nanargmin(self.to_numpy(key))]

    def argmax(self, key):
        return self.steps_list(key)[np.nanargmax(self.to_numpy(key))]

    def mean(self, key):
        return self.to_numpy(key).mean()

    def nanmean(self, key):
        return np.nanmean(self.to_numpy(key))

    def truncate(self, step):
        return DictLogger({k: {s:v for s,v in metric if s <= step} for k,metric in self.items()})