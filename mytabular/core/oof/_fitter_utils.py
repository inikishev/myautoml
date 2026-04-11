import inspect
import json
import logging
import math
import os
import time
from collections import UserDict, defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Any

import joblib
import numpy as np
import polars as pl

from ...utils import polars_utils, python_utils, torch_utils

if TYPE_CHECKING:
    from .fitter import ProblemType, TabularFitter

def _set_logging_file_handler_(self: "TabularFitter", root: Path):
    # Only keep file handler for current working directory
    if self._logging_file_handler is not None: self.logger.removeHandler(self._logging_file_handler)
    file_handler = logging.FileHandler(root / "mytabular.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    self.logger.addHandler(file_handler)
    self._logging_file_handler = file_handler


def _validate_and_log_features(X: pl.DataFrame, logger: logging.Logger):

    invalid_cols = (
        X
        .drop(pl.selectors.numeric())
        .drop(pl.selectors.string(include_categorical=True))
        .drop(pl.selectors.boolean())
    )

    if len(invalid_cols.columns) > 0:
        raise RuntimeError(f"Some columns have unsupported dtypes: {invalid_cols.schema}")

    num_cols = X.select(pl.selectors.numeric())
    logger.info("%i numeric columns: %r", len(num_cols.columns), num_cols.columns)

    bool_cols = X.select(pl.selectors.boolean())
    logger.info("%i boolean columns: %r", len(bool_cols.columns), bool_cols.columns)

    cat_cols = X.select(pl.selectors.categorical())
    logger.info("%i categorical columns: %r", len(cat_cols.columns), cat_cols.columns)

    text_cols = X.select(pl.selectors.string())
    logger.info("%i text columns: %r", len(text_cols.columns), text_cols.columns)



class _FoldSet(UserDict[int, dict[int, tuple[np.ndarray, np.ndarray]]]):
    """Holds fold indexes"""

    @classmethod
    def from_file(cls, path):
        fold_indexes_d = np.load(path)
        set_i = 0

        fold_indexes: defaultdict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
        while f"train_{set_i}_0" in fold_indexes_d:
            fold_i = 0
            while f'train_{set_i}_{fold_i}' in fold_indexes_d:
                train_index = fold_indexes_d[f'train_{set_i}_{fold_i}']
                test_index = fold_indexes_d[f'test_{set_i}_{fold_i}']
                fold_indexes[set_i][fold_i] = (train_index, test_index)
                fold_i += 1
            set_i += 1

        obj = cls(fold_indexes)
        obj.validate()
        return obj

    def validate(self):
        # validate that all folds have same number of indices
        for set_i, folds in self.items():
            for fold_i, (train_index, test_index) in folds.items():
                if len(train_index) + len(test_index) != self.n_samples:
                    train_index_0, test_index_0 = self[0][0]
                    raise RuntimeError(f"number of indices doesn't match for set 0 fold 0 and set {set_i} fold {fold_i}: "
                                       f"set 0 fold 0: {len(train_index_0)}, {len(test_index_0)}; "
                                       f"set {set_i} fold {fold_i}: {len(train_index)}, {len(test_index)}")

        # validate that folds and train/test splits don't have repeating indices
        for set_i, folds in self.items():
            cat_test_indexes = []
            for fold_i, (train_index, test_index) in folds.items():
                if len(np.intersect1d(train_index, test_index)) > 0:
                    raise RuntimeError(f"train and test have repeating indices for set {set_i} fold {fold_i}: "
                                       f"{np.intersect1d(train_index, test_index)}")

                if len(np.intersect1d(test_index, cat_test_indexes)) > 0:
                    raise RuntimeError(
                        f"In set {set_i} fold {fold_i} has test indices that already exist in previous folds: "
                        f"{np.intersect1d(test_index, cat_test_indexes)}"
                    )

                cat_test_indexes.extend(test_index.tolist())

            if set(cat_test_indexes) != set(range(self.n_samples)):
                raise RuntimeError(f"Set {set_i} has different indices from list(range(n_samples)): "
                                   f"{set(cat_test_indexes) ^ set(range(self.n_samples))}")


    @property
    def n_fold_sets(self): return len(self)
    @property
    def n_folds(self): return len(self[0])
    @property
    def n_samples(self):
        train_index, test_index = self[0][0]
        return len(train_index) + len(test_index)
    @property
    def n_estimators(self): return self.n_fold_sets * self.n_folds

    def merge_folds(self, n_folds: int | None = None) -> "tuple[_FoldSet, dict[int, int]]":
        """Merge groups of folds to get a new fold set with ``n_folds`` folds."""
        default_fold_map = {i: i for i in range(self.n_folds)}
        if n_folds is None: return self, default_fold_map
        if n_folds >= self.n_folds: return self, default_fold_map

        fold_map: dict[int,int] = {}
        inverse_map: defaultdict[int, list[int]] = defaultdict(list)
        group_i = 0

        for i in range(self.n_folds):
            fold_map[i] = group_i
            inverse_map[group_i].append(i)
            group_i += 1
            if group_i >= n_folds: group_i = 0

        merged_sets: defaultdict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
        for set_i in range(self.n_fold_sets):
            for group_i, folds in inverse_map.items():

                test_index = []
                for fold_i in folds:
                    test_index.extend(self[set_i][fold_i][1])

                train_index = set(range(self.n_samples)).difference(test_index)

                merged_sets[set_i][group_i] = (
                        np.asarray(sorted(train_index)),
                        np.asarray(sorted(test_index))
                    )

        merged = _FoldSet(merged_sets)
        assert merged.n_samples == self.n_samples, f"{merged.n_samples = }, {self.n_samples = }"
        assert merged.n_folds == n_folds, f"{merged.n_folds = }, {n_folds = }"
        merged.validate()

        return merged, fold_map



def _validate_test_indexes(cat_test_indexes, n_samples: int):
    if isinstance(cat_test_indexes, np.ndarray): cat_test_indexes = cat_test_indexes.tolist()

    if len(cat_test_indexes) != n_samples:
        raise RuntimeError(f"There are {len(cat_test_indexes)} test indices, but {n_samples} samples")

    if len(set(cat_test_indexes)) != len(cat_test_indexes):
        raise RuntimeError("There are repeating test indices")

    if set(cat_test_indexes) != set(range(n_samples)):
        raise RuntimeError(f"Test indices are different from list(range(n_samples)): "
                            f"{set(cat_test_indexes) ^ set(range(n_samples))}")



def _validate_preds(preds: np.ndarray, n_samples: int, n_targets: int):
    msg = (
        "``estimator.predict`` should return array of shape (n_samples, n_targets), "
        f"or (n_samples, ) is allowed if n_targets=1,"
        f"but {n_samples = }, {n_targets = }, and returned array has shape {preds.shape}"
    )

    if preds.ndim > 2: raise RuntimeError(msg)

    if preds.shape[0] != n_samples:
        raise RuntimeError(msg)

    if preds.ndim == 2:
        if preds.shape[1] != n_targets:
            raise RuntimeError(msg)

        if n_targets == 1:
            preds = preds.squeeze(-1)

    if preds.ndim == 1 and n_targets > 1:
        raise RuntimeError(msg)

    return preds

def _validate_probas(probas: np.ndarray, n_samples: int, n_classes: int):
    msg = (
        "``estimator.predict_proba`` should return array of shape (n_samples, n_classes), "
        f"but {n_samples = }, {n_classes = }, and returned array has shape {probas.shape}"
    )
    if probas.ndim != 2: raise RuntimeError(msg)
    if probas.shape[0] != n_samples: raise RuntimeError(msg)
    if probas.shape[1] != n_classes: raise RuntimeError(msg)

    return probas


def _sort_inputs(inputs: list[tuple[str | None, str | None]]) -> list[tuple[str | None, str | None]]:
    for i in inputs:
        # json converts tuples to lists which is fine, any sequence works except str
        assert isinstance(i, str) is False
        assert len(i) == 2
        if i[0] is None: assert i[1] is None

    def _strip(s: str | None):
        if isinstance(s, str): return s.strip()
        return s

    return sorted(
        set((_strip(i1), _strip(i2)) for i1,i2 in inputs),
        key = lambda x: tuple("" if i is None else i for i in x),
    )

def _validate_inputs(
    inputs: str | None | Sequence[str | None] | Sequence[tuple[str | None, str | None]],
) -> list[tuple[str | None, str | None]]:

    if inputs is None: return [(None, None)]
    if isinstance(inputs, str): inputs = [inputs,]

    validated: list[tuple[str | None, str | None]] = []
    for input in inputs:
        if input is None: validated.append((None, None))
        elif isinstance(input, str):
            if "/" in input:
                input, method = input.split("/")
                validated.append((input, method))
            else:
                validated.append((input, None))
        else:
            if len(input) != 2: raise RuntimeError(f"{input} should be length-2 tuple")
            estimator, method = input
            if estimator is None: method = None
            validated.append((estimator, method))

    return _sort_inputs(validated)


def _min_fit_sec_for_caching(X: np.ndarray | pl.DataFrame, min_sec: float = 0.1, size_per_sec: int = 1_000_000):
    numel = math.prod(X.shape)
    if numel > 125_000_000: return 1e100 # return very large value avoid caching more than ~1GB

     # 1_000_000 elements will cache if processing them takes over 1 second (8 MB if float64)
     # saving processing times under 0.2 seconds is not beneficial
    return max(min_sec, numel / size_per_sec)


class CacheKey(NamedTuple):
    estimator: str
    method: str
    set_i: int
    fold_i: int | None

class CachedFrame:
    """Dataframe that may be cached in RAM or on disk.

    Args:
        cache_file: path to parquet file to cache to.
        fn: function that computes and returns the dataframe.
        loaded: loaded DataFrame.
    """
    def __init__(
        self,
        cache_file: str | os.PathLike,
        fn: Callable[..., pl.DataFrame],
        logger: logging.Logger,
        loaded: pl.DataFrame | None = None,
    ):
        self.cache_file = Path(cache_file)
        self.fn = fn
        self.loaded = loaded
        self.logger = logger

    def load(self, max_ram_mb:float, max_disk_mb:float, min_cache_sec:float, cache_size_per_sec:int) -> pl.DataFrame:
        if self.loaded is not None:
            # we use level 1 for trace
            self.logger.log(1, "Loading %r dataframe from RAM", self.loaded.shape)
            return self.loaded

        if self.cache_file.exists():
            self.logger.log(1, "Loading cached dataframe from %s", str(self.cache_file))
            with pl.StringCache():
                return pl.read_parquet(self.cache_file)

        start = time.perf_counter()
        df = self.fn()

        time_sec = time.perf_counter() - start
        min_sec = _min_fit_sec_for_caching(df, min_sec=min_cache_sec, size_per_sec=cache_size_per_sec)

        if time_sec > min_sec:

            # 1 mb ~ 125_000 float64 elements
            numel = math.prod(df.shape)
            if numel < max_ram_mb * 125_000:
                self.logger.log(1, "Saving %r dataframe to RAM, because %.2f > %.2f", df.shape, time_sec, min_sec)
                self.loaded = df

            elif numel < max_disk_mb * 125_000:
                self.logger.log(1, "Saving %r dataframe to %s, because %.2f > %.2f",
                                df.shape, str(self.cache_file), time_sec, min_sec)

                with pl.StringCache():
                    df.write_parquet(self.cache_file, compression_level=3)

            # else: dataframe is too large and won't be cached

        else:
            self.logger.log(1, "Not saving %r dataframe, because %.2f <= %.2f", df.shape, time_sec, min_sec)

        return df

class SavedEstimator:
    """Takes care of processing estimator output for stacking.

    Args:
        dir: root directory of this estimator.
        set_i: set index.
        fold_i: fold index or None for estimator fitted to all folds.
        is_binary: whether problem_type is "binary".
        loaded: loaded estimator. Defaults to None.
    """
    def __init__(
        self,
        dir: str | os.PathLike,
        set_i: int,
        fold_i: int | None,
        problem_type: "ProblemType",
        logger: logging.Logger,
        loaded = None,
    ):
        self.dir = Path(dir)
        self.set_i = set_i
        self.fold_i = fold_i
        self.problem_type: "ProblemType" = problem_type
        self.loaded = loaded
        self.logger = logger

        self.cached_frames: dict[Path, CachedFrame] = {}

    def _total_size_cached_in_ram(self):
        """returns size (number of elements) cached"""
        total_size = 0
        for f in self.cached_frames.values():
            if f.loaded is not None:
                total_size += math.prod(f.loaded.shape)
        return total_size

    def _total_bytes_cached_on_disk(self):
        total_size = 0
        for f in self.cached_frames.values():
            if f.cache_file.exists():
                total_size += os.path.getsize(f.cache_file)
        return total_size


    @property
    def file(self): return self.dir / f"estimator-{self.set_i}-{self.fold_i}.joblib"
    @property
    def name(self): return self.dir.name

    def get_inputs(self, fitter: "TabularFitter", set_i: int, fold_i: int | None):
        config = self.get_config()
        inputs = config["inputs"]
        if config["used_estimators"] is None: return config["inputs"]

        inputs = [(e, m if m is not None else fitter.get_config(e)["method"]) for e,m in inputs]

        mapped_fold_i = config["fold_map"][str(fold_i)] if config["use_folds"] else None
        used_estimators = config["used_estimators"][str(set_i)][str(mapped_fold_i)]
        assert isinstance(used_estimators, list) and isinstance(used_estimators[0], str)
        new_inputs = [(e, m) for e, m in inputs if f"{e}.{m}" in used_estimators]
        missing = set(used_estimators).difference(f'{e}.{m}' for e,m in new_inputs)
        if len(missing) > 0:
            raise RuntimeError(f"Those columns are missing: {missing}. Available columns: {inputs}")

        self.logger.debug("Estimator has __mytabular_used_estimators__, some inputs will be skipped.")
        self.logger.debug("old inputs: %r", inputs)
        self.logger.debug("new inputs: %r", new_inputs)

        return new_inputs

    def get_config(self):
        return python_utils.read_json(self.dir / "config.json")

    def get_estimator(self):
        if self.loaded is None: return joblib.load(self.file)
        return self.loaded

    def load_test_index(self) -> np.ndarray:
        return pl.read_parquet(self.dir / "data" / f"test_index-{self.set_i}-{self.fold_i}.parquet").to_series().to_numpy()

    def predict_supervised(self, X) -> np.ndarray:
        """
        Passes ``X`` to ``estimator.predict``.

        If problem type is binary, predictions are binarized.
        """
        y = np.asarray(self.get_estimator().predict(X))

        if self.problem_type == "binary" and np.issubdtype(y.dtype, np.floating):
            y = y > 0.5

        return y

    def predict_proba_supervised(self, X) -> np.ndarray:
        """
        Passes ``X`` to ``estimator.predict_proba``.

        If problem type is binary and estimator doesn't support ``predict_proba``, it uses ``predict`` instead.
        """
        estimator = self.get_estimator()
        proba = None

        if hasattr(estimator, "predict_proba"):
            try: proba = np.asarray(estimator.predict_proba(X))
            except (NotImplementedError, AttributeError): pass

        if proba is None:
            if self.problem_type == "binary":
                pos = estimator.predict(X)
                proba = np.stack([1-pos, pos], -1)
            else:
                raise NotImplementedError(f"Estimator '{self.dir.name}' doesn't support predict_proba")

        return proba

    def _call_supervised_method_cached(
        self,
        X_fn: Callable[..., pl.DataFrame],
        cache_file: str | os.PathLike | None,
        method: str,
        max_ram_mb: float,
        max_disk_mb: float,
        min_cache_sec:float,
        cache_size_per_sec:int
    ) -> np.ndarray:
        """
        Args:
            X_fn: function that computes and returns X if needed.
            cache_file: path to cache file.
            method: method to call on self.
            max_ram_mb: won't cache to RAM if dataframe is approximately larger than this.
            max_disk_mb: won't cache to disk if dataframe is approximately larger than this.
        """
        if cache_file is None:
            return getattr(self, method)(X_fn())

        cache_file = Path(cache_file)

        if cache_file not in self.cached_frames:

            def fn():
                return pl.from_numpy(getattr(self, method)(X_fn()))

            self.cached_frames[cache_file] = CachedFrame(cache_file=cache_file, fn=fn, logger=self.logger, loaded=self.loaded)

        return self.cached_frames[cache_file].load(
            max_ram_mb=max_ram_mb,
            max_disk_mb=max_disk_mb,
            min_cache_sec=min_cache_sec,
            cache_size_per_sec=cache_size_per_sec,
        ).to_numpy()

    def predict_supervised_cached(
        self,
        X_fn: Callable[..., pl.DataFrame],
        cache_file: str | os.PathLike,
        max_ram_mb: float,
        max_disk_mb: float,
        min_cache_sec:float,
        cache_size_per_sec:int
    ) -> np.ndarray:
        """
        Args:
            X_fn: function that computes and returns X if needed.
            cache_file: path to cache file.
            method: method to call on self
            max_ram_mb: won't cache to RAM if dataframe is approximately larger than this.
            max_disk_mb: won't cache to disk if dataframe is approximately larger than this.
        """
        return self._call_supervised_method_cached(
            X_fn=X_fn,
            cache_file=cache_file,
            method="predict_supervised",
            max_ram_mb=max_ram_mb,
            max_disk_mb=max_disk_mb,
            min_cache_sec=min_cache_sec,
            cache_size_per_sec=cache_size_per_sec,
        )

    def predict_proba_supervised_cached(
        self,
        X_fn: Callable[..., pl.DataFrame],
        cache_file: str | os.PathLike,
        max_ram_mb: float,
        max_disk_mb: float,
        min_cache_sec:float,
        cache_size_per_sec:int
    ) -> np.ndarray:
        """
        Args:
            X_fn: function that computes and returns X if needed.
            cache_file: path to cache file.
            method: method to call on self.
            max_ram_mb: won't cache to RAM if dataframe is approximately larger than this.
            max_disk_mb: won't cache to disk if dataframe is approximately larger than this.
        """
        return self._call_supervised_method_cached(
            X_fn=X_fn,
            cache_file=cache_file,
            method="predict_proba_supervised",
            max_ram_mb=max_ram_mb,
            max_disk_mb=max_disk_mb,
            min_cache_sec=min_cache_sec,
            cache_size_per_sec=cache_size_per_sec,
        )

    def get_output_for_stacking(self, X, method: str | None) -> pl.DataFrame:
        """Passes ``X`` to estimator without any preprocessing and calls ``method``. Outputs are processed for stacking.

        For binary classification only positive label probability is kept.

        For multiclass classification, if ``method="predict"``, outputs are converted to categorical dtype.

        If output is supervised or is a numpy array, columns are named as ``f"{name}.{method}-{col_i}"``

        Args:
            X: input dataframe
            method: method to call on estimator
        """
        estimator = self.get_estimator()
        config = self.get_config()

        if method is None:
            method = config["method"]
            assert method is not None

        if method == config["method"]:
            is_categorical = config["is_categorical"]
        else:
            if config["is_supervised"] and method == "predict" and self.problem_type == "multiclass":
                is_categorical = True
            else:
                is_categorical = None # will be inferred later

        if method == "predict":
            if config["is_supervised"]: output = self.predict_supervised(X)
            else: output = estimator.predict(X)

        elif method == "predict_proba":
            if config["is_supervised"]:
                output = np.asarray(self.predict_proba_supervised(X))
                if self.problem_type == "binary":
                    assert output.shape[-1] == 2, output.shape
                    # keep only positive label
                    output = output[..., -1]

            else:
                output = estimator.predict_proba(X)

        else:
            output = getattr(estimator, method)(X)

        if torch_utils.is_array_or_tensor(output):
            if output.ndim == 1: output = output[:, None]
            schema = [f"{self.name}.{method}-{i}" for i in range(output.shape[1])]
            output = torch_utils.to_numpy(output)
            df = pl.from_numpy(output, schema=schema)

        else:
            df = polars_utils.to_dataframe(output)

        del output

        if is_categorical is None:
            is_categorical = (
                            df.width == 1 and
                            df.dtypes[0].is_integer() and
                            df[df.columns[0]].n_unique() > 2
                        )
            self.logger.log(1, "is_categorical for method %s inferred as %s", method, is_categorical)

        if is_categorical:
            with pl.StringCache():
                df = df.cast(pl.String()).cast(pl.Categorical())

        return df

    def get_output_for_stacking_cached(
        self,
        X_fn: Callable[..., pl.DataFrame],
        method: str | None,
        cache_file: str | os.PathLike | None,
        max_ram_mb: float,
        max_disk_mb: float,
        min_cache_sec:float,
        cache_size_per_sec:int
    ) -> pl.DataFrame:
        """
        Args:
            X_fn: function that computes and returns X if needed.
            cache_file: path to cache file.
            method: method to call on estimator
        """
        if cache_file is None:
            return self.get_output_for_stacking(X=X_fn(), method=method)

        cache_file = Path(cache_file)

        if cache_file not in self.cached_frames:

            def fn():
                return self.get_output_for_stacking(X=X_fn(), method=method)

            self.cached_frames[cache_file] = CachedFrame(cache_file=cache_file, fn=fn, logger=self.logger, loaded=self.loaded)

        return self.cached_frames[cache_file].load(
            max_ram_mb=max_ram_mb,
            max_disk_mb=max_disk_mb,
            min_cache_sec=min_cache_sec,
            cache_size_per_sec=cache_size_per_sec,
        )


def _get_fitted_configs(self: "TabularFitter") -> dict[str, dict[str, Any]]:

    configs = {}
    for estimator_dir in (self.root / "estimators").iterdir():
        files = os.listdir(estimator_dir)
        if "done.txt" not in files: continue

        with open(estimator_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        config["name"] = estimator_dir.name
        configs[estimator_dir.name] = config


    # determine stack level
    def get_children_(name: str) -> tuple[int, list[str]]:
        config = configs[name]

        if "stack_level" in config:
            return config["stack_level"], config["children"]

        stack_level = 0
        children = []

        for child, _ in config["inputs"]:
            if child is None: continue

            c_level, c_children = get_children_(child)
            children.append(child)
            children.extend(c_children)

            if configs[child]["is_supervised"]: stack_level = max(stack_level, c_level + 1)
            else: stack_level = max(stack_level, c_level)

        def sort_key(s):
            level, _ = get_children_(s)
            return level

        config["stack_level"] = stack_level
        config["children"] = sorted(set(children), key=sort_key)
        config["n_children"] = len(config["children"])
        return config["stack_level"], config["children"]

    for estimator,config in configs.items():
        if "stack_level" not in config:
            get_children_(estimator)

    return configs

def default_fit_fn(estimator, X: pl.DataFrame, y: pl.Series,
                   X_unlabeled: pl.DataFrame | None, sample_weight: np.ndarray | None):
    if X_unlabeled is not None:
        raise RuntimeError("`default_fit_fn` doesn't use X_unlabeled. "
                           "Specify a custom fit_fn, or use `mytabular.unlabeled_fit_fn` or "
                           "`mytabular.semi_supervised_classifier_fit_fn`")

    if sample_weight is not None: return estimator.fit(X, y.to_numpy(), sample_weight=sample_weight)
    return estimator.fit(X, y.to_numpy())


def semi_supervised_classifier_fit_fn(estimator, X: pl.DataFrame, y: pl.Series,
                                      X_unlabeled: pl.DataFrame | None, sample_weight: np.ndarray | None):
    if sample_weight is not None:
        raise RuntimeError("semi_supervised_classifier_fit_fn doesn't support sample_weight")

    if X_unlabeled is None:
        raise RuntimeError("semi_supervised_classifier_fit_fn requires X_unlabeled (make sure to set use_unlabeled=True")

    if y.dtype.is_float():
        raise RuntimeError(f"dtype of labels is {y.dtype}, semi_supervised_classifier_fit_fn requires integer")

    X_full = pl.concat([X, X_unlabeled], how='vertical_relaxed')

    y_unlabeled = np.full((X_unlabeled.height), fill_value=-1)
    y_full = pl.concat([y.cast(pl.Int64), pl.Series(y.name, y_unlabeled, dtype=pl.Int64)], how='vertical')

    return estimator.fit(X_full, y_full.to_numpy())

def unlabeled_fit_fn(estimator, X: pl.DataFrame, y: pl.Series,
                     X_unlabeled: pl.DataFrame | None, sample_weight: np.ndarray | None):

    if sample_weight is not None:
        raise RuntimeError("unlabeled_fit_fn doesn't support sample_weight")

    if X_unlabeled is None:
        raise RuntimeError("unlabeled_fit_fn requires X_unlabeled (make sure to set use_unlabeled=True")

    return estimator.fit(pl.concat([X, X_unlabeled], how='vertical_relaxed'))