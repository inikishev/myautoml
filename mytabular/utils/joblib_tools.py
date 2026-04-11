import os
from pathlib import Path

import joblib


class UnloadedObject[T]:
    def __init__(self, obj: T, filepath: str | os.PathLike, compress=3):
        if os.path.exists(filepath):
            raise RuntimeError(f"{filepath} already exists")

        self.filepath = filepath
        joblib.dump(obj, self.filepath, compress=compress)

        self.use_count = 0

    def __call__(self): # not a named method to avoid conflict with obj methods
        return joblib.load(self.filepath)

    def __getattr__(self, attr: str):
        self.use_count += 1
        return getattr(self(), attr)