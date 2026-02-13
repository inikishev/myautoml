import json
import logging
import math
import os
import string
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .fitter import TabularFitter

ProblemType = Literal["binary", "multiclass","regression", "multilabel", "multioutput", "multitask"]
# "multilabel", "multioutput", "multitask" are currently not supported but included for future support

_PROBLEM_TYPE_TO_TARGET_ENCODER: dict[ProblemType, Literal['standard', 'minmax', 'ordinal', 'none']] = {
    "binary": "ordinal",
    "multiclass": "ordinal",
    "regression": "minmax",
}


def _validate_and_log_features(X: pl.DataFrame, logger: logging.Logger):

    invalid_cols = X.drop(pl.selectors.numeric()).drop(pl.selectors.categorical()).drop(pl.selectors.boolean())
    if len(invalid_cols.columns) > 0:
        raise RuntimeError(f"Some columns have unsupported dtypes: {invalid_cols.schema}")

    num_cols = X.select(pl.selectors.numeric())
    logger.info("%i numeric columns: %r", len(num_cols.columns), num_cols.columns)

    bool_cols = X.select(pl.selectors.boolean())
    logger.info("%i boolean columns: %r", len(bool_cols.columns), bool_cols.columns)

    cat_cols = X.select(pl.selectors.categorical())
    logger.info("%i categorical columns: %r", len(cat_cols.columns), cat_cols.columns)



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
    def n_models(self): return self.n_fold_sets * self.n_folds

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
        "``model.predict`` should return array of shape (n_samples, n_targets), "
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
        "``model.predict_proba`` should return array of shape (n_samples, n_classes), "
        f"but {n_samples = }, {n_classes = }, and returned array has shape {probas.shape}"
    )
    if probas.ndim != 2: raise RuntimeError(msg)
    if probas.shape[0] != n_samples: raise RuntimeError(msg)
    if probas.shape[1] != n_classes: raise RuntimeError(msg)

    return probas



def _get_fitted_configs(self: "TabularFitter"):

    # ------------------------------- model configs ------------------------------ #
    model_configs = {}

    for model_dir in (self.root / "models").iterdir():
        if "done.txt" not in os.listdir(model_dir): continue

        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        config["name"] = model_dir.name
        if config["stack_models"] is None: config["stack_models"] = ()
        model_configs[model_dir.name] = config


    # ---------------------------- transformer configs --------------------------- #
    transformer_configs = {}

    for transformer_dir in (self.root / "transformers").iterdir():
        if "done.txt" not in os.listdir(transformer_dir): continue

        with open(transformer_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        config["name"] = transformer_dir.name
        if config["stack_models"] is None: config["stack_models"] = ()
        transformer_configs[transformer_dir.name] = config

    # config = {
    #     "transformer": transformer,
    #     "stack_models": stack_models,
    #     "passthrough": passthrough,
    #     "response_method": response_method,
    #     "fold_map": fold_map,
    #     "supports_proba": supports_proba,
    #     "n_models": fold_set.n_models,
    #     "start_time": start_time,
    #     "fit_sec": fit_sec,
    #     **scores,
    #     **{f"{k}_mean": np.mean(v) for k,v in scores},
    # }

    # determine stack level
    def get_children_(configs_: dict[str, dict], name: str) -> tuple[int, list[str], list[str]]:
        config = configs_[name]
        if "stack_level" in config: return config["stack_level"], config["child_models"], config["child_transformers"]

        stack_level = 1
        child_models = []
        child_transformers = []
        for c_model in config["stack_models"]:
            c_level, c_models, c_transformers = get_children_(model_configs, c_model)
            stack_level = max(stack_level, c_level + 1)
            child_models.extend(c_models)
            child_transformers.extend(c_transformers)

        transformer = config.get("transformer", config.get("pre_transformer", None))
        if transformer is not None:
            c_level, c_models, c_transformers = get_children_(transformer_configs, transformer)
            stack_level = max(stack_level, c_level)
            child_models.extend(c_models)
            child_transformers.extend(c_transformers)

        def model_key(s):
            level, _, _ = get_children_(model_configs, s)
            return level

        def transformer_key(s):
            level, _, _ = get_children_(transformer_configs, s)
            return level

        config["stack_level"] = stack_level
        config["child_models"] = sorted(set(child_models), key=model_key)
        config["child_transformers"] = sorted(set(child_transformers), key=transformer_key)
        config["n_child_models"] = len(config["child_models"])
        return config["stack_level"], config["child_models"], config["child_transformers"]

    for model,config in model_configs.items():
        if "stack_level" not in config:
            get_children_(model_configs, model)

    for transformer,config in transformer_configs.items():
        if "stack_level" not in config:
            get_children_(transformer_configs, transformer)

    return model_configs, transformer_configs

def rename_model(self: "TabularFitter", current_name: str, new_name: str):
    if current_name not in os.listdir(self.root / "models"):
        raise RuntimeError(f"model {current_name} doesn't exist")

    # rename dir
    os.rename(self.root / "models" / current_name, self.root / "models" / new_name)

    # rename in all configs
    dirs = list((self.root / "models").iterdir()) + list((self.root / "transformers").iterdir())
    for estimator in dirs:
        if "config.json" in os.listdir(estimator):

            with open(estimator / "config.json", "r", encoding="utf-8") as f: config = json.load(f)

            if config["stack_models"] is not None:
                config["stack_models"] = [m if m!=current_name else new_name for m in config["stack_models"]]
                with open(estimator / "config.json", "w", encoding="utf-8") as f: json.dump(config, f)


def rename_transformer(self: "TabularFitter", current_name: str, new_name: str):
    if current_name not in os.listdir(self.root / "transformers"):
        raise RuntimeError(f"transformer {current_name} doesn't exist")

    # rename dir
    os.rename(self.root / "transformers" / current_name, self.root / "transformers" / new_name)

    # rename in all configs
    dirs = list((self.root / "models").iterdir()) + list((self.root / "transformers").iterdir())
    for estimator in dirs:
        if "config.json" in os.listdir(estimator):
            with open(estimator / "config.json", "r", encoding="utf-8") as f: config = json.load(f)

            if "transformer" in config and config["transformer"] == current_name:
                config["transformer"] = new_name
                with open(estimator / "config.json", "w", encoding="utf-8") as f: json.dump(config, f)

            if "pre_transformer" in config and config["pre_transformer"] == current_name:
                config["pre_transformer"] = new_name
                with open(estimator / "config.json", "w", encoding="utf-8") as f: json.dump(config, f)

def _min_fit_sec_for_caching(X: np.ndarray | pl.DataFrame):
    numel = math.prod(X.shape)
    if numel > 10 ** 8: return 1e10 # avoid caching more than ~1GB
    return (math.prod(X.shape) * 100) ** 0.5


class _SavedPreds(UserDict[int, dict[int, dict[str, str]]]):
    def __init__(self, dir):
        self.dir = dir
        self.types: set[str] = set()

        folds = defaultdict(lambda: defaultdict(dict))
        for filename in os.listdir(dir):

            filepath = os.path.join(dir, filename)
            type, set_i, fold_i = ".".join(filename.rsplit(".", 1)[:-1]).rsplit("-", 2)

            assert type in ("test_index", "test_preds", "test_proba"), type
            self.types.add(type)

            set_i = int(set_i)
            fold_i = int(fold_i)

            folds[set_i][fold_i][type] = filepath

        super().__init__({k: dict(v) for k,v in folds.items()})

    @property
    def n_fold_sets(self): return len(self)
    @property
    def n_folds(self): return len(self[0])

    def load(self, type: Literal["test_index", "test_preds", "test_proba"], set_i: int, fold_i: int) -> np.ndarray:
        filename = f"{type}-{set_i}-{fold_i}.npz"
        d = np.load(os.path.join(self.dir, filename))
        assert len(d.keys()) == 1, list(d.keys())
        return d["data"]