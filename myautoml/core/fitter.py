import atexit
import json
import logging
import math
import os
import random
import shutil
import string
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

from ..metrics import scoring
from ..polars_transformers.auto_encoder import AutoEncoder, _AutoEncoderWrapper
from ..utils import numpy_utils, polars_utils, python_utils, torch_utils
from ..utils.rng import RNG
from . import _fitter_utils

ResponseMethod = Literal["decision_function", "predict", "predict_proba", "transform"] | str
ProblemType = Literal["binary", "multiclass", "regression", "multitarget", "multioutput", "multitask"]
_PROBLEM_TYPE_TO_TARGET_ENCODER: dict[ProblemType, Literal['standard', 'minmax', 'ordinal', 'none']] = {
    "binary": "ordinal",
    "multiclass": "ordinal",
    "regression": "minmax",
}

class TabularFitter:
    """Tabular fitter.

    Args:
        verbosity: logging verbosity, 0 means no logging and 4 is the most verbose. Defaults to 2.
        max_ram_cache_mb: maximum size of cached dataframes in RAM. Defaults to 1024.
        max_disk_cache_mb: maximum size of cached dataframes on disk. Defaults to 10240.
    """

    problem_type: ProblemType
    """One of 'binary', 'multiclass', 'multilabel', 'regression', 'multioutput', 'multitask'"""

    per_fold_info = False
    """if this is true, per-fold validation metrics are logged under INFO verbosity, otherwise DEBUG"""

    estimators: defaultdict[str, defaultdict[int, dict[int | None, _fitter_utils.SavedEstimator]]]
    """name > set_i > fold_i > SavedEstimator, where fold_i is mapped, i.e. some folds may be merged"""

    def __init__(
        self,
        verbosity: Literal[0, 1, 2, 3, 4] = 2,
        max_ram_cache_mb = 1024,
        max_disk_cache_mb = 10240,
    ):
        # Create a logger
        self.logger = logging.getLogger("myautoml.core.fitter.TabularFitter")
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        self.logger.setLevel(1)

        console_handler = logging.StreamHandler()
        console_handler.setLevel([logging.CRITICAL, logging.WARNING, logging.INFO, logging.DEBUG, 1][verbosity])
        self.logger.addHandler(console_handler)

        self._logging_file_handler: Any = None

        self.max_ram_cache_mb = max_ram_cache_mb
        self.max_disk_cache_mb = max_disk_cache_mb
        self.cached_frames: dict[Path, _fitter_utils.CachedFrame] = {}

        atexit.register(self._delete_temp_dir)

    def set_logging_level(self, level):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)

    def _delete_temp_dir(self):
        """runs at exit to make sure temp files are deleted"""
        if hasattr(self, 'root'):
            if os.path.exists(self.root / "temp"):
                try: shutil.rmtree(self.root / "temp", ignore_errors=True)
                except Exception: pass

    def _total_mb_cached_in_ram(self):
        """Returns an approximation (upper bound based on float64 size in memory)"""
        total_size = 0
        for f in self.cached_frames.values():
            if f.loaded is not None:
                total_size += math.prod(f.loaded.shape)

        for sets in self.estimators.values():
            for folds in sets.values():
                for estimator in folds.values():
                    total_size += estimator._total_size_cached_in_ram()

        return total_size / 125_000


    def _total_mb_cached_on_disk(self):
        total_bytes = 0
        for f in self.cached_frames.values():
            if f.cache_file.exists():
                total_bytes += os.path.getsize(f.cache_file)

        for sets in self.estimators.values():
            for folds in sets.values():
                for estimator in folds.values():
                    total_bytes += estimator._total_bytes_cached_on_disk()

        return total_bytes / 1_048_576

    def initialize(
        self,
        X: pl.DataFrame | Any,
        y: str | pl.Series | Any,
        X_unlabeled=None,
        problem_type: ProblemType | None = None,
        eval_metric: scoring.Scorer | str | Callable | None = None,
        dir: str | os.PathLike | None = None,
        n_folds: int = 8,
        n_fold_sets: int = 1,
        seed: int | None = 0,
        load_if_exists: bool = True,

        convert_categorical: bool = True,
        drop_constant: bool = True,
        binary_to_bool: bool = True,
        encode_target: bool = True,
        drop_cols: str | Sequence[str] | None = None,
        numeric_cols: str | Sequence[str] | None = None,
        categorical_cols: str | Sequence[str] | None = None,
        text_cols: str | Sequence[str] | None = None,

    ):
        """Initialize or load this ``TabularFitter``.

        Args:
            X: Training DataFrame.
            y: Either name of label column in ``X``, or a Series.
            X_unlabeled: unlabeled data for semi-supervised learning. Defaults to None.
            problem_type: 'binary', 'multiclass', or 'regression'. By default infers from ``y``.
            eval_metric: evaluation metric to use. By default uses a metric based on problem type.
            dir: directory to store everything in. Default directory is ``f"myautoml-{datetime}"``,
                and if such directory exists, it is loaded.
            n_folds: number of folds. Defaults to 8.
            n_fold_sets: number of fold sets. Total number of estimators fitted is ``n_folds * n_fold_sets``. Defaults to 1.
            seed: random seed for generating folds. Defaults to 0.
            load_if_exists: whether to load ``dir`` if it exists. Defaults to True.
            convert_categorical: whether to convert string columns to categorical. Defaults to True.
            drop_constant: whether to drop constant columns. Defaults to True.
            binary_to_bool: whether to convert binary features to bool. Defaults to True.
            encode_target: whether to encode target - ordinally for classification or to float64 for regression.
                Defaults to True.
            drop_cols: columns to ignore (like id),

        """
        if n_folds <= 1:
            raise RuntimeError(f"n_folds should be an integer larger than 1, got {n_folds}")
        if n_fold_sets <= 0:
            raise RuntimeError(f"n_fold_sets should be an integer larger than 0, got {n_fold_sets}")

        # create dir
        if dir is None:

            if load_if_exists:
                # check if myautoml already exists
                for d in sorted(os.listdir()):
                    if d.startswith("myautoml-"):
                        # another dir starting with myautoml was already assigned,
                        # so it is ambiguous which one to load and user must specify in that case
                        if dir is not None:
                            raise RuntimeError(
                                "dir is not specified but are multiple directories starting with 'myautoml-'. "
                                "Specify `dir` manually.")
                        dir = d

            if dir is None:
                nanos = time.time_ns()
                dt = datetime.fromtimestamp(nanos / 1e9)
                dir = f"myautoml-{dt.strftime('%Y-%m-%d %H-%M-%S')}-{(nanos % 1e9):09.0f}"
                self.logger.info("dir is not specified, creating a new directory %s", dir)

            else:
                self.logger.info("dir is not specified, loading %s", dir)

        root = Path(dir)
        if (root / "done.txt").exists():
            if load_if_exists:
                self.load(root)
                return
            raise RuntimeError(f"Directory {root} already exists. Set `load_if_exists=True` or use `load` method.")

        root.mkdir(exist_ok=True)

        # Only keep file handler for current working directory
        _fitter_utils._set_logging_file_handler_(self, root=root)

        # create the dir structure
        (root / "estimators").mkdir(exist_ok=True)

        # encode and save the processed dataframe and the encoder
        enc = AutoEncoder(
            convert_categorical = convert_categorical,
            drop_constant = drop_constant,
            binary_to_bool = binary_to_bool,
            encode_target = encode_target,
            drop_cols = drop_cols,
        )
        enc.logger = self.logger
        enc.fit(X=X, y=y, X_unlabeled=X_unlabeled, problem_type=problem_type)

        auto_encoder = enc.to_frozen()
        joblib.dump(auto_encoder, root / "auto_encoder.joblib", compress=3)

        X, y = auto_encoder.transform_X_y(X, y)
        _fitter_utils._validate_and_log_features(X, self.logger)

        with pl.StringCache():
            X.write_parquet(root / "X.parquet")
            y.to_frame().write_parquet(root / "y.parquet")

            if X_unlabeled is not None:
                X_unlabeled = auto_encoder.transform_X(X_unlabeled)
                X_unlabeled.write_parquet(root / "X_unlabeled.parquet")

        # save config
        problem_type = cast(ProblemType, auto_encoder.problem_type_)
        config = {"problem_type": problem_type}
        python_utils.write_json(config, root / "config.json")

        # infer and save scorer
        if eval_metric is None: eval_metric = scoring.DEFAULT_SCORERS[problem_type]
        scorer = scoring.get_scorer(eval_metric)
        joblib.dump(scorer, root / "scorer.joblib", compress=3)

        # save fold indexes
        folds = {}
        if seed is None: seed = random.randint(0, 10**10)
        for i in range(n_fold_sets):

            if problem_type in ('binary', 'multiclass'):
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed+i)
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed+i)

            for fold, (train_index, test_index) in enumerate(kf.split(np.empty(len(y)), y)):
                folds[f"train_{i}_{fold}"] = train_index
                folds[f"test_{i}_{fold}"] = test_index

        np.savez_compressed(root / "fold_indexes.npz", **folds)

        # mark as done
        with open(root / "done.txt", 'w', encoding='utf-8') as f:
            f.write("")

        # load from initialized dir
        self.load(dir=dir)

    def load(self, dir: str | os.PathLike):
        """Load from ``dir``."""
        self.root = Path(dir)

        # Create temporary directory
        if (self.root / "temp").exists(): shutil.rmtree(self.root / "temp")
        (self.root / "temp").mkdir()

        # add log file handler
        # Only keep file handler for current working directory
        _fitter_utils._set_logging_file_handler_(self, root=self.root)

        # load encoder
        self.auto_encoder: _AutoEncoderWrapper = joblib.load(self.root / "auto_encoder.joblib")

        # load dataframes
        with pl.StringCache():
            self.X = pl.read_parquet(self.root / "X.parquet")
            self.y = pl.read_parquet(self.root / "y.parquet").to_series()

            if (self.root / "X_unlabeled.parquet").exists():
                self.X_unlabeled = pl.read_parquet(self.root / "X_unlabeled.parquet")
            else:
                self.X_unlabeled = None

        # load other attributes
        config = python_utils.read_json(self.root / "config.json")
        self.problem_type: ProblemType = config["problem_type"]

        self.scorer: scoring.Scorer = joblib.load(self.root / "scorer.joblib")
        self.fold_set = _fitter_utils._FoldSet.from_file(self.root / "fold_indexes.npz")
        self._n_classes = None

        # load estimators
        self.estimators = defaultdict(lambda: defaultdict(dict))

        for dir in (self.root / "estimators").iterdir():
            if dir.is_file(): continue
            files = os.listdir(dir)
            if "done.txt" not in files: continue

            for filename in files:
                if filename.startswith("estimator") and filename.endswith(".joblib"):

                    _, set_i, fold_i = filename.replace(".joblib", "").rsplit("-", 2)
                    set_i = int(set_i)
                    fold_i = None if fold_i == "None" else int(fold_i)

                    self.estimators[dir.name][set_i][fold_i] = _fitter_utils.SavedEstimator(
                        dir=dir, set_i=set_i, fold_i=fold_i, problem_type=self.problem_type, logger=self.logger,
                    )


    def is_classification(self):
        return self.problem_type in ('binary', 'multiclass', 'multilabel')

    def is_single_target_classification(self):
        return self.problem_type in ('binary', 'multiclass')

    def is_regression(self):
        return self.problem_type in ('regression', 'multioutput')

    def is_single_target(self):
        return self.problem_type in ('binary', 'multiclass', 'regression')

    def is_multi_target(self):
        return self.problem_type in ('multilabel', 'multioutput', 'multitask')

    @property
    def n_features(self): return self.X.shape[1]

    @property
    def n_samples(self): return self.X.shape[0]

    @property
    def n_classes(self) -> int:
        if not self.is_classification(): raise RuntimeError(f"n_classes can't be used with problem_type={self.problem_type}")
        if self.problem_type == 'binary': return 2
        if self._n_classes is None: self._n_classes = self.y.n_unique()
        return self._n_classes

    @property
    def n_targets(self) -> int:
        if self.is_single_target(): return 1
        raise NotImplementedError(f"TODO support for {self.problem_type}")

    @property
    def n_fold_sets(self) -> int: return self.fold_set.n_fold_sets

    @property
    def n_folds(self) -> int: return self.fold_set.n_folds

    def get_config(self, estimator: str):
        if estimator not in self.estimators:
            raise KeyError(f"Estimator {estimator} doesn't exist.")

        return next(iter(
            next(iter(
                self.estimators[estimator].values()
            )).values()
        )).get_config()

    def _get_outputs_oof_uncached(
        self,
        set_i: int,
        estimator: str,
        method: str | None,
    ) -> pl.DataFrame:
        """Returns out-of-fold outputs of ``estimator.method`` for stacking.

        Args:
            set_i: fold set index.
            estimator: name of the estimator.
            method: method to call on the estimator. None calls default method specified during fit.
        """
        outputs: list[pl.DataFrame] = []

        config = self.get_config(estimator)

        if method is None:
            method = config["method"]
            assert method is not None

        X = self.stack_oof(set_i=set_i, inputs=config["inputs"])

        if config["use_folds"]:
            for fold_i, saved_estimator in self.estimators[estimator][set_i].items():
                test_index = saved_estimator.load_test_index()

                fold_test_outputs = saved_estimator.get_output_for_stacking(
                    X = X[test_index],
                    method = method
                )

                idx_col = pl.Series("__myautoml_col_id", test_index)
                outputs.append(fold_test_outputs.with_columns(idx_col))

            df = pl.concat(outputs).sort("__myautoml_col_id")
            _fitter_utils._validate_test_indexes(df["__myautoml_col_id"].to_list(), self.n_samples)
            return df.drop("__myautoml_col_id")

        # else: estimator doesn't use folds
        saved_estimator = self.estimators[estimator][set_i][None]
        return saved_estimator.get_output_for_stacking(X=X, method=method)

    def get_outputs_oof(
        self,
        set_i: int,
        estimator: str,
        method: str | None,
    ) -> pl.DataFrame:

        if method is None: method = self.get_config(estimator)["method"]
        cache_file = self.root / "estimators" / estimator / "data" / f"{method}-{set_i}.parquet"

        if cache_file not in self.cached_frames:

            fn = partial(self._get_outputs_oof_uncached, set_i=set_i, estimator=estimator, method=method)
            self.cached_frames[cache_file] = _fitter_utils.CachedFrame(cache_file=cache_file, fn=fn, logger=self.logger)

        ram_mb = self._total_mb_cached_in_ram()
        disk_mb = self._total_mb_cached_on_disk()

        # avoid going over limit, also avoid storing over 1% of total cache size in RAM
        max_ram_mb = min(self.max_ram_cache_mb - ram_mb, self.max_ram_cache_mb / 100)

        # also avoid storing more than 1GB on disk as it becomes too inefficient
        max_disk_mb = min(self.max_disk_cache_mb - disk_mb, self.max_disk_cache_mb / 10, 125_000_000)

        # we use log level 1 for trace
        self.logger.log(1, "used/max. allowed cache in RAM: %.2f/%.2f MB, on disk: %.2f/%.2f MB; ",
                        ram_mb, max_ram_mb, disk_mb, max_disk_mb)

        return self.cached_frames[cache_file].load(max_ram_mb=max_ram_mb, max_disk_mb=max_disk_mb)


    def stack_oof(
        self,
        set_i: int,
        inputs: list[tuple[str | None, str | None]],
    ) -> pl.DataFrame:
        """Returns stacked X and/or out-of-fold outputs of specified estimators.

        Args:
            set_i: fold set index.
            inputs: list of tuples ``(estimator, method)``, ``(None, None)`` means original features.
        """
        stacks = []
        for estimator, method in _fitter_utils._sort_inputs(inputs):

            if estimator is None:
                assert method is None, method
                stacks.append(self.X)

            else:
                stacks.append(self.get_outputs_oof(set_i, estimator, method))

        return pl.concat(stacks, how='horizontal', strict=True)

    def get_outputs_new(
        self,
        X: pl.DataFrame,
        set_i: int,
        fold_i: int,
        estimator: str,
        method: str | None,
        cache_dir: str | os.PathLike | None,
    ):
        """Returns outputs of ``estimator.method`` on new data for stacking.

        Args:
            X: input dataframe.
            set_i: fold set index.
            fold_i: fold index.
            estimator: name of the estimator.
            method: method to call on the estimator. None calls default method specified during fit.
            cache_dir: path to directory for storing cached outputs.
        """
        config = self.get_config(estimator)
        mapped_fold_i = config["fold_map"][str(fold_i)] if config["use_folds"] else None
        saved_estimator = self.estimators[estimator][set_i][mapped_fold_i]

        inputs = saved_estimator.get_inputs(self, set_i, fold_i)

        X_fn = partial(
            self.stack_new,
            X = X,
            set_i = set_i,
            fold_i = fold_i,
            inputs = inputs,
            cache_dir = cache_dir,
        )

        ram_mb = self._total_mb_cached_in_ram()
        disk_mb = self._total_mb_cached_on_disk()

        # avoid going over limit, also avoid storing over 1% of total cache size in RAM
        max_ram_mb = min(self.max_ram_cache_mb - ram_mb, self.max_ram_cache_mb / 100)

        # also avoid storing more than 1GB on disk as it becomes too inefficient
        max_disk_mb = min(self.max_disk_cache_mb - disk_mb, self.max_disk_cache_mb / 10, 125_000_000)

        # we use log level 1 for trace
        self.logger.log(1, "used/max. allowed cache in RAM: %.2f/%.2f MB, on disk: %.2f/%.2f MB; ",
                        ram_mb, max_ram_mb, disk_mb, max_disk_mb)

        if method is None: method = config["method"]
        if cache_dir is None: cache_file = None
        else: cache_file = Path(cache_dir) / f'{estimator}-{method}-{set_i}-{fold_i}.parquet'

        return saved_estimator.get_output_for_stacking_cached(
            X_fn,
            method=method,
            cache_file=cache_file,
            max_ram_mb=max_ram_mb,
            max_disk_mb=max_disk_mb,
        )

    def stack_new(
        self,
        X: pl.DataFrame,
        set_i: int,
        fold_i: int,
        inputs: list[tuple[str | None, str | None]],
        cache_dir: str | os.PathLike | None
    ) -> pl.DataFrame:
        """Returns stacked X and/or outputs of specified estimators on new data.

        Args:
            X: input dataframe.
            set_i: fold set index.
            fold_i: fold index.
            inputs: list of tuples ``(estimator, method)``, ``(None, None)`` means original features.
            cache_dir: path to directory for storing cached outputs.
        """
        stacks = []
        for estimator, method in _fitter_utils._sort_inputs(inputs):

            if estimator is None:
                assert method is None, method
                stacks.append(X)

            else:
                stacks.append(
                    self.get_outputs_new(
                        X = X,
                        set_i = set_i,
                        fold_i = fold_i,
                        estimator = estimator,
                        method = method,
                        cache_dir = cache_dir,
                    ))

        return pl.concat(stacks, how='horizontal', strict=True)

    def _fit_estimator(
        self,
        name: str,
        estimator,
        method: str | None,
        is_categorical: bool | None,
        use_folds: bool,
        max_folds: int | None,
        is_supervised: bool,

        inputs: list[tuple[str | None, str | None]],

        sample_weight: np.ndarray | tuple[str, str | None] | None,
        sample_weight_fn: Callable[[np.ndarray], np.ndarray] | None,

        groups: list[str],

        use_unlabeled: bool,
        fit_fn: Callable[[Any, pl.DataFrame, pl.Series, pl.DataFrame | None, np.ndarray | None], Any],
        info: Any,

        save: bool
    ) -> np.ndarray:
        """Fits an estimator to the dataset. See ``fit_supervised`` and ``fit_unsupervised`` for arguments."""
        if save:
            assert is_supervised is True

        if max_folds is not None:
            if max_folds <= 1:
                raise RuntimeError(f"max_folds should be an integer larger than 1, got {max_folds}")
            if max_folds >= self.n_folds:
                self.logger.warning("max_folds = %i, but n_folds = %i, so no folds are merged", max_folds, self.n_folds)

        start_time = time.time()

        # Create estimator folder
        dir = self.root / "estimators" / name
        if save:
            if (dir / "done.txt").exists():
                raise FileExistsError(f"Estimator {name} already exists, choose a different unique name.")

            dir.mkdir(exist_ok=True)
            (dir / "data").mkdir(exist_ok=True)
            (dir / "data_unlabeled").mkdir(exist_ok=True)

        # Load folds
        if use_folds:
            fold_set, fold_map = self.fold_set.merge_folds(n_folds=max_folds)
            n_fitted = fold_set.n_estimators
            self.logger.info('Fitting %i estimators "%s" to folds.', n_fitted, name)

        else:
            fold_set = fold_map = None
            n_fitted = self.n_fold_sets
            self.logger.info('Fitting %i estimators "%s" to full dataset.', n_fitted, name)

        if name in self.estimators:
            del self.estimators[name]

        scores = defaultdict(list)
        used_estimators = defaultdict(lambda: defaultdict(list))
        obj_qualname = None
        obj_repr = None
        supports_proba = False
        method_was_none = method is None
        is_categorical_was_none = is_categorical is None
        is_tested = False
        in_features = None
        out_features = None
        sample_weight_type = None

        # ------------------------- Fit to each set and fold ------------------------- #
        for set_i in range(self.n_fold_sets):

            X_oof = self.stack_oof(
                set_i = set_i,
                inputs = inputs,
            )

            # --- Load sample_weight
            if isinstance(sample_weight, tuple):
                sw_est, sw_method = sample_weight
                sample_weight_type = (sw_est, sw_method)
                set_sample_weight = self.get_outputs_oof(set_i, sw_est, sw_method).to_numpy()

            else:
                set_sample_weight = sample_weight
                if isinstance(sample_weight, np.ndarray): sample_weight_type = "array"
                elif sample_weight is not None:
                    raise RuntimeError("sample_weight must be np.ndarray, length-2 tuple or None.")

            if set_sample_weight is not None:
                if sample_weight_fn is not None: set_sample_weight = sample_weight_fn(set_sample_weight)
                if set_sample_weight.shape not in ((X_oof.height, 1), (X_oof.height, )):
                    raise RuntimeError(f"Sample weights have shape {set_sample_weight.shape}, but they must be "
                                       f"{(X_oof.height, 1)} or {(X_oof.height, )}")
            # ---

            in_features = X_oof.width

            oof_preds_list = []
            oof_proba_list = []
            oof_indexes_list = []

            for fold_i in (range(fold_set.n_folds) if fold_set is not None else (None, )):

                # --------- Get train samples of this fold or take the entire dataset -------- #
                if fold_i is not None:
                    assert fold_set is not None
                    train_index, test_index = fold_set[set_i][fold_i]
                    X_train = X_oof[train_index]
                    y_train = self.y[train_index]
                    if save:
                        pl.from_numpy(test_index).write_parquet(
                            dir / "data" / f"test_index-{set_i}-{fold_i}.parquet", compression_level=3)

                else:
                    train_index = test_index = None
                    X_train = X_oof
                    y_train = self.y

                # ----------------------------- Fit the estimator ---------------------------- #
                fitted_file = dir / f"estimator-{set_i}-{fold_i}.joblib"
                if fitted_file.exists():
                    self.logger.warning("%s already exists, it will be loaded", str(fitted_file))
                    fitted_estimator = joblib.load(fitted_file)

                else:
                    if use_unlabeled and self.X_unlabeled is not None:
                        X_unlabeled = self.stack_new(
                            X = self.X_unlabeled,
                            set_i = set_i,
                            # we can't meaningfully average dataframes,
                            # so if estimator is fitted to all data,
                            # we have to take the first fold.
                            fold_i = fold_i if fold_i is not None else 0,
                            inputs = inputs,
                            cache_dir = dir / "data_unlabeled"
                        )
                    else:
                        X_unlabeled = None

                    fold_sample_weight = set_sample_weight
                    if fold_sample_weight is not None: fold_sample_weight = fold_sample_weight[train_index]

                    fitted_estimator = fit_fn(estimator, X_train, y_train, X_unlabeled, fold_sample_weight)

                    if fitted_estimator is None: # pyright:ignore[reportUnnecessaryComparison]
                        raise RuntimeError(f"fit_fn for {name} returned None. Make sure estimator.fit returns self.")

                    if save:
                        try:
                            joblib.dump(fitted_estimator, fitted_file, compress=3)
                        except Exception as e:
                            if os.path.exists(fitted_file): os.remove(fitted_file)
                            raise e

                if hasattr(fitted_estimator, "__myautoml_used_estimators__"):
                    ret = getattr(fitted_estimator, "__myautoml_used_estimators__")()
                    if ret is not None:
                        assert isinstance(ret, str) is False
                        used_estimators[str(set_i)][str(fold_i)] = [str(v) for v in ret] # convert np.str_

                # add SavedEstimator to self.estimators
                saved_estimator = self.estimators[name][set_i][fold_i] = _fitter_utils.SavedEstimator(
                    dir=dir, set_i=set_i, fold_i=fold_i, problem_type=self.problem_type,
                    logger=self.logger, loaded=fitted_estimator
                )

                if obj_qualname is None: obj_qualname = python_utils.get_qualname(fitted_estimator)
                if obj_repr is None: obj_repr = repr(fitted_estimator)[:1_000_000]

                if is_supervised:

                    assert (train_index is not None) and (test_index is not None)
                    assert fold_i is not None
                    assert saved_estimator.loaded is not None

                    X_test = X_oof[test_index]
                    y_test = self.y[test_index]

                    oof_indexes_list.append(test_index)

                    preds_train = saved_estimator.predict_supervised(X_train)
                    preds_test = saved_estimator.predict_supervised(X_test)

                    preds_train = _fitter_utils._validate_preds(preds_train, X_train.shape[0], self.n_targets)
                    preds_test = _fitter_utils._validate_preds(preds_test, X_test.shape[0], self.n_targets)

                    oof_preds_list.append(preds_test)

                    proba_train = proba_test = None
                    if self.is_classification():

                        try:
                            proba_train = saved_estimator.predict_proba_supervised(X_train)
                            proba_test = saved_estimator.predict_proba_supervised(X_test)

                            proba_train = _fitter_utils._validate_probas(proba_train, X_train.shape[0], self.n_classes)
                            proba_test = _fitter_utils._validate_probas(proba_test, X_test.shape[0], self.n_classes)

                            oof_proba_list.append(proba_test)

                            if method is None: method = "predict_proba"

                            supports_proba = True

                        except (NotImplementedError, AttributeError) as e:
                            if method == "predict_proba":
                                raise e from None

                    if method is None: method = "predict"

                    # infer is_categorical after inferring method
                    if self.problem_type == "multiclass":
                        if method == "predict": is_categorical = True
                        elif method == "predict_proba": is_categorical = False
                    else:
                        # for "binary", binary features don't have to be categorical (they arent one hot encoded)
                        if method in ("predict", "predict_proba"):
                            is_categorical = False

                    # Score
                    score_train, error_train = self.scorer.score_and_error(
                        targets=y_train.to_numpy(), preds=preds_train, proba=proba_train)

                    score_test, error_test = self.scorer.score_and_error(
                        targets=y_test.to_numpy(), preds=preds_test, proba=proba_test)

                    (self.logger.info if self.per_fold_info else self.logger.debug)(
                        "Set %i fold %i - %s: train = %.8f, test = %.8f",
                        set_i, fold_i, self.scorer.name, float(score_train), float(score_test))

                    scores["score_train"].append(score_train)
                    scores["score_test"].append(score_test)
                    scores["error_train"].append(error_train)
                    scores["error_test"].append(error_test)

                    # we already tested predict and predict_proba
                    if method in ("predict", "predict_proba"):
                        assert is_categorical is not None
                        is_tested = True


                if not is_tested:

                    # do a quick estimator.method test on first 100 rows, this catches a lot of issues early
                    assert method is not None
                    height = min(100, X_oof.height)
                    output = polars_utils.to_dataframe(getattr(fitted_estimator, method)(X_oof.head(height)))

                    if output.height != height:
                        raise RuntimeError(
                            f"Estimator {name} received a frame of shape {X_oof.shape}, and returned shape {output.shape}")

                    out_features = output.width
                    if is_categorical is None:
                        is_categorical = (
                            output.width == 1 and
                            output.dtypes[0].is_integer() and
                            output[output.columns[0]].n_unique() > 2
                        )
                    is_tested = True


                # Unload fitted_estimator
                del fitted_estimator
                saved_estimator.loaded = None


            # After each fold set, store out-of-fold predictions for stacking
            if is_supervised and save:
                assert fold_set is not None
                assert len(oof_preds_list) == len(oof_indexes_list) == fold_set.n_folds

                oof_indexes = np.concatenate(oof_indexes_list)
                _fitter_utils._validate_test_indexes(oof_indexes, self.n_samples)
                argsort = np.argsort(oof_indexes)
                del oof_indexes

                oof_preds = np.concatenate(oof_preds_list)[argsort]

                # Apply the same logic as SavedEstimator.get_output_for_stacking
                if oof_preds.ndim == 1: oof_preds = oof_preds[:, None]
                assert method is not None
                schema = [f"{name}.{method}-{i}" for i in range(oof_preds.shape[1])]
                oof_preds = pl.from_numpy(oof_preds, schema)
                if method == "predict": out_features = oof_preds.width

                assert is_categorical is not None
                if is_categorical:
                    with pl.StringCache():
                        oof_preds = oof_preds.cast(pl.String()).cast(pl.Categorical())

                with pl.StringCache():
                    oof_preds.write_parquet(dir / "data" / f"predict-{set_i}.parquet")

                del oof_preds

                if supports_proba:
                    assert len(oof_proba_list) == fold_set.n_folds

                    oof_proba = np.concatenate(oof_proba_list)[argsort]
                    assert oof_proba.ndim == 2

                    if self.problem_type == "binary":
                        assert oof_proba.shape[-1] == 2, oof_proba.shape
                        # keep only positive label
                        oof_proba = oof_proba[..., -1, None]

                    assert method is not None
                    schema = [f"{name}.{method}-{i}" for i in range(oof_proba.shape[1])]
                    oof_proba = pl.from_numpy(oof_proba, schema)
                    if method == "predict_proba": out_features = oof_proba.width
                    oof_proba.write_parquet(dir / "data" / f"predict_proba-{set_i}.parquet")
                    del oof_proba

                else:
                    assert len(oof_proba_list) == 0

                del argsort


        fit_sec = time.time() - start_time

        # Log various info
        if method_was_none: self.logger.info('Inferred method as "%s"', method)
        if is_categorical_was_none: self.logger.info("Inferred is_categorical as %s", is_categorical)

        if "score_train" in scores:
            self.logger.info(
                "Mean %s: train = %.8f; test = %.8f; Took %.2f seconds",
                self.scorer.name, np.mean(scores["score_train"]), np.mean(scores["score_test"]), fit_sec)
        else:
            self.logger.info("Took %.2f seconds", fit_sec)

        if save is False:
            # if estimator is not saved, skip all saving logic (only supervised)
            return np.array(scores["error_test"])

        # all of those should be set by now
        assert method is not None
        assert is_categorical is not None
        assert obj_qualname is not None
        assert obj_repr is not None
        assert in_features is not None
        assert out_features is not None

        # Save config
        config = {
            "name": name,
            "inputs": inputs,
            "method": method,
            "is_categorical": is_categorical,
            "is_supervised": is_supervised,
            "supports_proba": supports_proba,
            "in_features": in_features,
            "out_features": out_features,
            "sample_weight_type": sample_weight_type,
            "sample_weight_fn": python_utils.get_qualname(sample_weight_fn) if sample_weight_fn is not None else None,
            "groups": groups,
            "has_info": info is not None,
            "use_folds": use_folds,
            "fold_map": fold_map, # note: int keys are converted to str by json
            "n_fitted": n_fitted,
            "used_estimators": used_estimators if len(used_estimators) > 0 else None,
            "start_time": start_time,
            "fit_sec": fit_sec,
            "obj_qualname": obj_qualname,
        }

        if is_supervised:
            assert len(scores) == 4
            config.update(scores)
            config.update({f"{k}_mean": np.mean(v) for k,v in scores.items()})

        if isinstance(method, str):
            config["response_method"] = method
        else:
            joblib.dump(method, dir / "response_method.joblib")

        with open(dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        # Save object repr for inspection
        assert obj_repr is not None
        with open(dir / "repr.txt", "w", encoding='utf-8') as f:
            f.write(obj_repr)

        # Save info
        if info is not None:
            try:
                joblib.dump(info, dir / "info.joblib", compress=3)
            except Exception as e:
                self.logger.error("Failed to save info for %s:\n%r", name, e)

        # Mark as done
        with open(dir / "done.txt", "w", encoding='utf-8') as f:
            f.write("")

        if is_supervised: return np.array(scores["error_test"])
        return np.empty(0)

    def fit_supervised(
        self,
        name: str,
        estimator,
        method: str | None = None,
        is_categorical: bool | None = None,
        inputs: str | None | Sequence[str | None] | Sequence[tuple[str | None, str | None]] = None,

        sample_weight: np.ndarray | str | tuple[str, str | None] | None = None,
        sample_weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,

        groups: str | Sequence[str] | None = None,

        use_unlabeled: bool = False,
        max_folds: int | None = None,
        fit_fn: Callable[
            [Any, pl.DataFrame, pl.Series, pl.DataFrame | None, np.ndarray | None], Any
        ] = _fitter_utils.default_fit_fn,
        info: Any = None,
        save: bool = True,
    ) -> np.ndarray:
        """Fit a supervised estimator to the dataset and score it.

        Args:
            name: unique name of the estimator.
            estimator: estimator to fit.
            method: default method on the estimator used to get the outputs. Note that if estimator is supervised,
                ``predict`` / ``predict_proba`` are always used for scoring. If None, it is set to
                to ``"predict_proba"`` or ``"predict"`` based on what the estimator supports. Defaults to None.
            is_categorical: whether output of ``estimator.method`` should be converted to categorical,
                may be inferred and ignored in some cases. If not specified, inferred from problem type or output dtype. If ``estimator`` already outputs a dataframe with categorical dtypes, this can be set tp False.
            inputs: inputs to fit the estimator to, ``None`` means original features.
                Can be a sequence of strings and None, or sequence of tuples ``(estimator, method)``,
                where original features are written as ``(None, None)``.
                If ``method`` is None or not provided, uses the default method specified for the estimator.
                Defaults to None.
            sample_weight: numpy array of shape ``(n_samples, )``, or name of estimator or tuple
                ``(estimator, method)``, which should output a single column that will be used as sample weights.
            sample_weight_fn: function applied to sample weights if they are specified.
            groups: string or list of string names of groups for managing estimators. You can select estimators
                from a group in ``select_fitted``.
            use_unlabeled: set this to True if ``estimator`` and ``fit_fn`` use unlabeled data,
                False (default) skips the potentially expensive operation of computing stacked unlabeled inputs.
            max_folds: merges folds if number of folds is larger than this value, for estimators that are slow to fit.
            fit_fn: Function called as ``fn(estimator, X, y, X_unlabeled)`` which should return fitted estimator.
                Defaults to ``estimator.fit(X, y, sample_weights=sample_weights)``.
            info: if specified, pickled and saved to estimator folder as ``info.joblib``.
                This can be used to store hyperparameters when tuning them for later reference.
            save: if False, estimator will not be saved, you can use this when tuning hyperparameters.
        """
        if isinstance(estimator, (list, tuple)):
            if len(estimator) == 1: estimator = estimator[0]
            else: estimator = make_pipeline(*estimator)

        inputs = _fitter_utils._validate_inputs(inputs)

        if isinstance(sample_weight, str): sample_weight = (sample_weight, None)
        if groups is None: groups = []
        elif isinstance(groups, str): groups = [groups, ]
        else: groups = list(groups)

        return self._fit_estimator(
            name = name,
            estimator = estimator,
            method = method,
            is_categorical = is_categorical,
            use_folds = True,
            is_supervised = True,
            inputs = inputs,
            sample_weight=sample_weight,
            sample_weight_fn=sample_weight_fn,
            groups=groups,
            max_folds = max_folds,
            use_unlabeled = use_unlabeled,
            fit_fn = fit_fn,
            info = info,
            save = save,
        )


    def fit_unsupervised(
        self,
        name: str,
        estimator,
        use_folds: bool,
        method: str = "transform",
        is_categorical: bool | None = None,

        inputs: str | None | Sequence[str | None] | Sequence[tuple[str | None, str | None]] = None,

        sample_weight: np.ndarray | str | tuple[str, str | None] | None = None,
        sample_weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,

        groups: str | Sequence[str] | None = None,

        use_unlabeled: bool = False,
        max_folds: int | None = None,
        fit_fn: Callable[
            [Any, pl.DataFrame, pl.Series, pl.DataFrame | None, np.ndarray | None], Any
        ] = _fitter_utils.default_fit_fn,
        info: Any = None,

    ) -> None:
        """Fit an unsupervised estimator or a feature transformer to the dataset.

        Args:
            name: unique name of the estimator.
            estimator: estimator to fit.
            use_folds: whether to fit estimator to each fold, or to all data. If estimator uses labels, you should
                set this to True to avoid label leakage.
            method: default method on the estimator used to get the outputs, e.g. ``"transform"``
            is_categorical: whether output of ``estimator.method`` should be converted to categorical,
                may be inferred and ignored in some cases. If not specified, inferred from problem type or output dtype. If ``estimator`` already outputs a dataframe with categorical dtypes, this can be set tp False.
            inputs: inputs to fit the estimator to, ``None`` means original features.
                Can be a sequence of strings and None, or sequence of tuples ``(estimator, method)``,
                where original features are written as ``(None, None)``.
                If ``method`` is None or not provided, uses the default method specified for the estimator.
                Defaults to None.
            sample_weight: numpy array of shape ``(n_samples, )``, or name of estimator or tuple
                ``(estimator, method)``, which should output a single column that will be used as sample weights.
            sample_weight_fn: function applied to sample weights if they are specified.
            groups: string or list of string names of groups for managing estimators. You can select estimators
                from a group in ``select_fitted``.
            use_unlabeled: set this to True if ``estimator`` and ``fit_fn`` use unlabeled data,
                False (default) skips the potentially expensive operation of computing stacked unlabeled inputs.
            max_folds: merges folds if number of folds is larger than this value, for estimators that are slow to fit.
                Ignored if ``use_folds=False``.
            fit_fn: Function called as ``fn(estimator, X, y, X_unlabeled)`` which should return fitted estimator.
                Defaults to ``estimator.fit(X, y)``.
            info: if specified, pickled and saved to estimator folder as ``info.joblib``.
                This can be used to store hyperparameters when tuning them for later reference.
        """
        if isinstance(estimator, (list, tuple)):
            if len(estimator) == 1: estimator = estimator[0]
            else: estimator = make_pipeline(*estimator)

        inputs = _fitter_utils._validate_inputs(inputs)

        if isinstance(sample_weight, str): sample_weight = (sample_weight, None)
        if groups is None: groups = []
        elif isinstance(groups, str): groups = [groups, ]
        else: groups = list(groups)

        self._fit_estimator(
            name = name,
            estimator = estimator,
            method = method,
            is_categorical = is_categorical,
            use_folds = use_folds,
            is_supervised = False,
            inputs = inputs,
            sample_weight = sample_weight,
            sample_weight_fn = sample_weight_fn,
            groups = groups,
            max_folds = max_folds,
            use_unlabeled = use_unlabeled,
            fit_fn = fit_fn,
            info = info,
            save = False,
        )


    def _predict_raw(
        self,
        X,
        estimator: str,
    ) -> np.ndarray:
        if isinstance(X, pl.LazyFrame): X = X.collect()
        self.auto_encoder.validate_data(X)
        X = self.auto_encoder.transform_X(X)

        config = self.get_config(estimator)
        preds = None
        n = 0

        for set_i in range(self.n_fold_sets):
            for fold_i in range(self.n_folds):

                mapped_fold_i = config["fold_map"][str(fold_i)] if config["use_folds"] else None
                saved_estimator = self.estimators[estimator][set_i][mapped_fold_i]

                config = saved_estimator.get_config()
                inputs = saved_estimator.get_inputs(self, set_i, fold_i)

                X_stacked = self.stack_new(
                    X,
                    set_i = set_i,
                    fold_i = fold_i,
                    inputs = inputs,
                    cache_dir = self.root / "temp"
                )

                if self.is_classification():
                    if config["supports_proba"]:
                        out = saved_estimator.predict_proba_supervised(X_stacked)
                    else:
                        out = saved_estimator.predict_supervised(X_stacked)
                        out = numpy_utils.one_hot(out, n_classes=self.n_classes)
                else:
                    out = saved_estimator.predict_supervised(X_stacked)

                if preds is None: preds = out
                else: preds += out
                n += 1

        self._delete_temp_dir()
        (self.root / "temp").mkdir()

        assert preds is not None
        return preds / n

    def predict(self, X, estimator: str) -> pl.Series:
        """Predict targets of ``X`` using ``estimator``, returns Series"""
        if self.is_classification():
            preds = np.argmax(self._predict_raw(X, estimator), axis=-1)
        else:
            preds = self._predict_raw(X, estimator)
        return self.auto_encoder.inverse_transform_y(preds)

    def predict_proba(self, X, estimator: str) -> np.ndarray:
        """Predict probabilities of ``X`` using ``estimator``"""
        if not self.is_classification():
            raise RuntimeError(f"predict_proba can only be used for classification, but {self.problem_type = }")

        return self._predict_raw(X, estimator)

    def list_fitted(
        self,
        sort="start_time",
        include: polars_utils.PolarsColumnSelector | None = None,
        exclude: polars_utils.PolarsColumnSelector | None = (
            # note: score_train, etc are arrays of per-fold scores
            # score_train_mean is the mean score and kept by default
            # arrays create a lot of line breaks when displayed and make dataframe less readable.
            "inputs", "children", "fold_map", "score_train", "score_test",
            "error_train", "error_test", "used_estimators", "groups"
        ),
        supervised: bool = True,
        unsupervised: bool = False,
    ):
        """Creates a dataframe which lists all fitted estimators and various info.

        Args:
            sort: column to sort the dataframe by. Defaults to "start_time".
            include: columns to keep, if None, keeps all columns.
            exclude: columns to drop, if None, keeps all columns.
                Defaults to array columns that tend to make the display less readable.
            supervised: whether to include supervised estimators. Defaults to True.
            unsupervised: whether to include unsupervised estimators. Defaults to False.
        """
        if supervised is False and unsupervised is False:
            raise RuntimeError("supervised=False, and unsupervised=False")

        configs = _fitter_utils._get_fitted_configs(self)
        if len(configs) == 0: return pl.DataFrame()

        df = pl.from_dicts(list(configs.values()), infer_schema_length=None)

        if supervised is False:
            df = df.filter(pl.col("is_supervised").not_())

        if unsupervised is False:
            df = df.filter(pl.col("is_supervised"))

        cols = ("name", "stack_level", "score_train_mean", "score_test_mean")
        df = df.sort(sort).select(*cols, pl.all().exclude(cols))

        return polars_utils.include_exclude_cols(df, include=include, exclude=exclude)


    def select_estimators(
        self,
        name_expr: str | None = None,
        supervised: bool = True,
        unsupervised: bool = False,
        groups: str | Sequence[str] | None = None,
        stack_level: int | None = None,
        min_stack_level: int | None = None,
        max_stack_level: int | None = None,
    ) -> list[str]:
        """Returns a list of estimator names that satisfy specified criteria.
        This is useful to pass as ``inputs`` argument of ``fit_supervised``/``fit_unsupervised``.
        With default arguments selects all supervised estimators.

        Args:
            name_expr: Selects estimators if the name contains a substring that matches a pattern. Defaults to None.
            supervised: whether to include supervised estimators. Defaults to True.
            unsupervised: whether to include unsupervised estimators. Defaults to False.
            groups: selects estimators present in any of those groups.
            stack_level: Selects estimators of specified stack level. Defaults to None.
            min_stack_level: Selects estimators with equal or higher stack level. Defaults to None.
            max_stack_level: Selects estimators with equal or lower stack level. Defaults to None.
        """
        estimators = self.list_fitted(exclude=None, supervised=supervised, unsupervised=unsupervised)
        if name_expr is not None:
            estimators = estimators.filter(pl.col("name").str.contains(name_expr))

        if stack_level is not None:
            estimators = estimators.filter(pl.col("stack_level") == stack_level)

        if min_stack_level is not None:
            estimators = estimators.filter(pl.col("stack_level") >= min_stack_level)

        if max_stack_level is not None:
            estimators = estimators.filter(pl.col("stack_level") <= max_stack_level)

        if groups is not None:
            if isinstance(groups, str): groups = [groups]
            estimators = estimators.filter(pl.col("groups").list.set_intersection(groups).len() > 0)

        if len(estimators) == 0:
            self.logger.warning("Found no estimators matching select_estimators arguments.")
            return []

        return estimators["name"].to_list()

    def delete_estimator(self, names: str | Sequence[str]):
        """Deletes estimator or estimators. If specified estimators are children of other estimators,
        an exception will be raised instead."""
        if isinstance(names, str): names = (names, )

        all_dirs = os.listdir(self.root / "estimators")
        for name in names:
            if name not in all_dirs:
                raise FileNotFoundError(f'estimator "{name}" doesn\'t exist.')\

        # make sure estimators aren't used in any other estimators
        other_estimators = self.list_fitted(exclude=None).filter(pl.col("name").is_in(names).not_())

        for name, children in zip(other_estimators["name"], other_estimators["children"]):
            matches = set(names).intersection(children)
            if len(matches) > 0:
                raise RuntimeError(f'Can\'t delete {matches} because they are used in "{name}"')

        for name in names:
            shutil.rmtree(self.root / "estimators" / name)

    def preview_stacked(
        self,
        inputs: str | None | Sequence[str | None] | Sequence[tuple[str | None, str | None]] = None,
        set_i: int = 0,
    ) -> pl.DataFrame:
        """Compute a stacked dataframe to preview what it looks like.

        Args:
            inputs: inputs (see ``TabularFitter.fit_supervised``).
            set_i: fold set to use estimators from. Defaults to 0.
        """
        inputs = _fitter_utils._validate_inputs(inputs)
        return self.stack_oof(set_i=set_i, inputs=inputs)


    def delete_unfitted(self) -> None:
        """Deletes all estimators that have not finished fitting."""
        for estimator in (self.root / "estimators").iterdir():
            if "done.txt" not in os.listdir(estimator):
                self.logger.info("Deleting unfitted estimator %s", str(estimator))
                shutil.rmtree(estimator)
