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
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import KFold, StratifiedKFold

from ..metrics import scoring
from ..polars_transformers.auto_encoder import AutoEncoder, _AutoEncoderWrapper
from ..utils import numpy_utils, polars_utils, python_utils
from ..utils.rng import RNG
from . import _fitter_utils

ResponseMethod = Literal["predict", "predict_proba"]


class TabularFitter:
    """Tabular fitter.

    Args:
        verbosity: how detailed the logs outputted to console should be.
            - 0: No logging (silent mode)
            - 1: Displays warnings
            - 2: Displays info such as validation metrics when fitting models.
            - 3: Extra information for debugging.
        caching_level: how much should be stored in RAM.
            - 0: No caching - models and transformers are only loaded to RAM when used and then immediately unloaded,
                this can be slow so should only be used if you run into out of memory.
            - 1: Smart caching - Cache is cleared after every fit and predict (default).
            - 2: Greedy caching - All models and transformers are cached in RAM and are never unloaded.
        max_cache_mb: maximum size of cache folder in MB.
    """
    problem_type: _fitter_utils.ProblemType
    """One of 'binary', 'multiclass', 'multilabel', 'regression', 'multioutput', 'multitask'"""

    per_fold_info = False
    """if this is true, per-fold validation metrics are logged under INFO verbosity, otherwise DEBUG"""


    def __init__(
        self,
        verbosity: Literal[0, 1, 2, 3] = 2,
        caching_level: Literal[0, 1, 2] = 1,
        max_cache_mb: float = 10240,
    ):
        # Logging stuff
        self.logger = logging.getLogger("myautoml.core.fitter.TabularFitter")
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel([logging.CRITICAL, logging.WARNING, logging.INFO, logging.DEBUG][verbosity])
        self.logger.addHandler(console_handler)

        self._logging_file_handler = None

        self.caching_level = caching_level
        self.max_cache_mb = max_cache_mb

        self._temp_load_cache: dict[str, Any] = {}
        self._temp_predict_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._temp_transform_cache: dict[str, str] = {}
        self._temp_caching_enabled: bool = False
        self._tmpdir: str | None = None
        """Used internally"""

    def initialize(
        self,
        X: pl.DataFrame | Any,
        y: str | pl.Series | Any,
        X_unlabeled=None,
        problem_type: _fitter_utils.ProblemType | None = None,
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

    ):
        """Initialize or load this TabularFitter.

        Args:
            X: Training DataFrame.
            y: Either name of label column in ``X``, or a Series.
            X_unlabeled: unlabeled data for semi-supervised learning. Defaults to None.
            problem_type: 'binary', 'multiclass', or 'regression'. Inferred from label by default. Defaults to None.
            eval_metric: evaluation metric to use. By default uses a metric based on problem type.
            dir: directory to store everything in. Default directory is ``f"myautoml-{datetime}"``,
                and if such directory exists, it is loaded.
            n_folds: number of folds. Defaults to 8.
            n_fold_sets: number of fold sets. Total number of models fitted is ``n_folds * n_fold_sets``. Defaults to 1.
            seed: random seed for generating folds. Defaults to 0.
            load_if_exists: whether to load ``dir`` if it exists. Defaults to True.
            convert_categorical: whether to convert string columns to categorical. Defaults to True.
            drop_constant: whether to drop constant columns. Defaults to True.
            binary_to_bool: whether to convert binary features to bool. Defaults to True.
            encode_target: whether to encode target - ordinally for classification or to float64 for regression.
                Defaults to True.
            drop_cols: columns to ignore (like id)
        """
        # create dir
        if dir is None:

            if load_if_exists:
                # check if myautoml already exists
                for d in sorted(os.listdir()):
                    if d.startswith("myautoml-"):
                        if dir is not None: # another dir starting with myautoml was already assigned
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

        if self._logging_file_handler is not None: self.logger.removeHandler(self._logging_file_handler)
        file_handler = logging.FileHandler(root / "myautoml.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self._logging_file_handler = file_handler

        # create the dir structure
        (root / "models").mkdir(exist_ok=True)
        (root / "transformers").mkdir(exist_ok=True)
        # (root / "results").mkdir(exist_ok=True)

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

        X.write_parquet(root / "X.parquet")
        y.to_frame().write_parquet(root / "y.parquet")

        if X_unlabeled is not None:
            X_unlabeled = auto_encoder.transform_X(X_unlabeled)
            X_unlabeled.write_parquet(root / "X_unlabeled.parquet")

        # save config
        problem_type = cast(_fitter_utils.ProblemType, auto_encoder.problem_type_)
        config = {"problem_type": problem_type}
        with open(root / "config.json", "w", encoding='utf-8') as f:
            json.dump(config, f)

        # infer and save scorer
        if eval_metric is None: eval_metric = scoring.DEFAULT_SCORERS[problem_type]
        scorer = scoring.make_scorer(eval_metric)
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
            f.write("done")

        # load from initialized dir
        self.load(dir=dir)

    def load(self, dir: str | os.PathLike):
        self.root = Path(dir)

        # add log file handler
        if self._logging_file_handler is not None: self.logger.removeHandler(self._logging_file_handler)
        file_handler = logging.FileHandler(self.root / "myautoml.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self._logging_file_handler = file_handler

        # load encoder
        self.auto_encoder: _AutoEncoderWrapper = joblib.load(self.root / "auto_encoder.joblib")

        # load datas
        self.X = pl.read_parquet(self.root / "X.parquet")
        self.y = pl.read_parquet(self.root / "y.parquet").to_series()

        if (self.root / "X_unlabeled.parquet").exists():
            self.X_unlabeled = pl.read_parquet(self.root / "X_unlabeled.parquet")
        else:
            self.X_unlabeled = None

        # load other stuffs
        with open(self.root / "config.json", "r", encoding='utf-8') as f:
            config = json.load(f)
            self.problem_type: _fitter_utils.ProblemType = config["problem_type"]

        self.scorer: scoring.Scorer = joblib.load(self.root / "scorer.joblib")
        self.fold_set = _fitter_utils._FoldSet.from_file(self.root / "fold_indexes.npz")
        self._n_classes = None

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
    def n_features(self):
        return self.X.shape[1]

    @property
    def n_samples(self):
        return self.X.shape[0]

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

    def _get_fitted_configs(self):
        return _fitter_utils._get_fitted_configs(self)

    def list_fitted_models(self, sort="start_time"):
        models_d,_ = self._get_fitted_configs()
        models = pl.from_dicts(list(models_d.values()))
        cols = ("name", "transformer", "stack_level", "score_train_mean", "score_test_mean")
        models = models.sort(sort).select(*cols, pl.all().exclude(cols))
        return models

    def select_models(
        self,
        name_expr: str | None = None,
        stack_level: int | None = None,
        min_stack_level: int | None = None,
        max_stack_level: int | None = None,
    ):
        models = self.list_fitted_models()
        if name_expr is not None:
            models = models.filter(pl.col("name").str.contains(name_expr))

        if stack_level is not None:
            models = models.filter(pl.col("stack_level") == stack_level)

        if min_stack_level is not None:
            models = models.filter(pl.col("stack_level") >= min_stack_level)

        if max_stack_level is not None:
            models = models.filter(pl.col("stack_level") <= max_stack_level)

        return models["name"].to_list()

    def rename_model(self, current_name: str, new_name: str):
        _fitter_utils.rename_model(self, current_name, new_name)

    def rename_transformer(self, current_name: str, new_name: str):
        _fitter_utils.rename_transformer(self, current_name, new_name)

    def preview_transformed(
        self,
        transformer: str | None = None,
        stack_models: str | Sequence[str] | None = None,
        passthrough: bool = True,
        response_method: ResponseMethod = 'predict_proba',
        set_i: int = 0,
    ):
        return self.get_stacked_X(
            set_i=set_i, stack_models=stack_models, transformer=transformer,
            passthrough=passthrough, response_method=response_method,
        )

    def list_fitted_transformers(self, sort="start_time"):
        _,transformers_d = self._get_fitted_configs()
        transformers = pl.from_dicts(list(transformers_d.values()))
        transformers = transformers.sort(sort).select("name", pl.all().exclude("name"))
        return transformers

    def delete_unfitted(self):

        for model in (self.root / "models").iterdir():
            if "done.txt" not in os.listdir(model): shutil.rmtree(model)

        for transformer in (self.root / "transformers").iterdir():
            if "done.txt" not in os.listdir(transformer): shutil.rmtree(transformer)

    def _cached_load(self, path: str | os.PathLike, loader: Callable) -> Any:
        if self.caching_level == 0:
            return loader(path) # caching disabled

        path = os.path.normpath(path)
        if path in self._temp_load_cache:
            self.logger.debug("Loading cached %s", path)
            return self._temp_load_cache[path]

        obj = loader(path)

        if self._temp_caching_enabled or self.caching_level == 2:
            self.logger.debug("Saving %s to cache", path)
            self._temp_load_cache[path] = obj

        return obj

    @contextmanager
    def _temp_caching_context(self):

        assert self._temp_caching_enabled is False
        assert len(self._temp_predict_cache) == 0
        assert len(self._temp_transform_cache) == 0
        if self.caching_level != 2: assert len(self._temp_load_cache) == 0

        if self.caching_level != 0:
            self.logger.debug("Enabling temporary caching")
            self._temp_caching_enabled = True

        with (tempfile.TemporaryDirectory() if self._temp_caching_enabled else nullcontext()) as tmpdir:
            self._tmpdir = tmpdir
            try:
                yield

            finally:
                self.logger.debug("Disabling temporary caching")
                self._tmpdir = None
                self._temp_predict_cache.clear()
                self._temp_transform_cache.clear()
                if self.caching_level != 2: self._temp_load_cache.clear()
                self._temp_caching_enabled = False

    def transform_oof(self, set_i: int, transformer: str) -> pl.DataFrame:
        """(Internal method) Returns ``self.X`` transformed by out-of-fold (if applicable) ``transformer`` for fold set ``set_i``."""

        transformer_dir = self.root / "transformers" / transformer

        with open(transformer_dir / "config.json", "r", encoding='utf-8') as f:
            config = json.load(f)

        X = self.get_stacked_X(
            set_i,
            stack_models=config["stack_models"],
            transformer=config["pre_transformer"],
            passthrough=config["passthrough"],
            response_method=config["response_method"],
        )

        if config["use_folds"]:

            # transformer is fitted to folds, can transform out-of-folds samples
            transformed_list: list[pl.DataFrame] = []
            test_indexes = _fitter_utils._SavedPreds(transformer_dir / "test_indexes")

            for fold_i in range(test_indexes.n_folds):

                # load transformer
                fitted_transformer = self._cached_load(
                    transformer_dir / f"transformer-{set_i}-{fold_i}.joblib", joblib.load)

                test_index = test_indexes.load("test_index", set_i=set_i, fold_i=fold_i)
                assert np.issubdtype(test_index.dtype, np.integer)

                # transform
                oof = fitted_transformer.transform(X[test_index])

                # add transformed dataframe
                col = pl.Series("__myautoml_col_id", test_index)
                transformed_list.append(polars_utils.to_dataframe(oof).with_columns(col))

            transformed = pl.concat(transformed_list).sort("__myautoml_col_id")
            _fitter_utils._validate_test_indexes(transformed["__myautoml_col_id"].to_list(), self.n_samples)
            return transformed.drop("__myautoml_col_id")


        # config["use_folds"] = False, transformer is fitted to all data
        fitted_transformer = self._cached_load(transformer_dir / f"transformer-{set_i}-ALL.joblib", joblib.load)

        return polars_utils.to_dataframe(fitted_transformer.transform(X))


    def get_oof_preds(self, set_i: int, model: str, response_method: ResponseMethod) -> np.ndarray:
        """(Internal method) Returns cached out-of-fold predictions for all samples.

        Args:
            set_i: fold set index.
            model: model name.
            response_method: Specifies whether to use predict_proba or predict.
        """
        model_dir = self.root / "models" / model
        preds = _fitter_utils._SavedPreds(model_dir / "predictions")

        if response_method == "predict": k = "test_preds"
        elif response_method == "predict_proba": k = "test_proba"
        else: raise ValueError(response_method)

        one_hot = False
        if k not in preds.types:
            self.logger.debug("%s doesn't support predict_proba, falling back to predict", model)
            if response_method == 'predict_proba':
                one_hot = True
                shape = [preds.load("test_preds", 0, 0).shape[0], self.n_classes]
            else:
                raise KeyError(f"Predictions for {model} do not contain {k}")

        else:
            shape = list(preds.load(k, 0, 0).shape)

        shape[0] = self.n_samples
        oof_preds = np.zeros(shape)

        cat_test_indexes = []

        for fold_i in range(preds.n_folds):

            if one_hot:
                assert k == "test_proba"
                test_preds = numpy_utils.one_hot(preds.load("test_preds", set_i, fold_i), self.n_classes)
            else:
                test_preds = preds.load(k, set_i, fold_i)

            test_index = preds.load("test_index", set_i, fold_i)
            assert np.issubdtype(test_index.dtype, np.integer)
            oof_preds[test_index] = test_preds

            cat_test_indexes.extend(test_index.tolist())

        _fitter_utils._validate_test_indexes(cat_test_indexes, self.n_samples)
        return oof_preds

    def _stack_preds(
        self,
        models: Sequence[str],
        pred_fn: Callable[[str], Any],
        passthrough: bool,
        X_passthrough: pl.DataFrame,
        response_method: ResponseMethod,
    ):
        """Used in ``get_stacked_X``, ``get_stacked_X_unlabeled`` and ``_get_stacked_X_new``."""
        stack_preds = {}
        for model in sorted(set(models)):
            preds = np.asarray(pred_fn(model))

            if preds.ndim == 1: stack_preds[model] = preds
            else: python_utils.safe_dict_update_(
                stack_preds,
                {f"{model}_{i}": arr for i, arr in enumerate(preds.T)}
            )

        stack_df = pl.from_dict(stack_preds)

        if response_method == "predict" and self.is_classification() and self.n_classes > 2:
            # convert category predictions to categorical
            with pl.StringCache():
                stack_df = stack_df.cast(pl.String()).cast(pl.Categorical)

        if passthrough: stack_df = pl.concat([X_passthrough, stack_df], how='horizontal')
        return stack_df

    def get_stacked_X(
        self,
        set_i: int,
        stack_models: str | Sequence[str] | None,
        transformer: str | None,
        passthrough: bool,
        response_method: ResponseMethod,
    ) -> pl.DataFrame:
        """(Internal method) Returns ``X`` or ``transformer(X)``
        optionally with out-of-fold predictions of ``stack_models``.

        Args:
            set_i: fold set index.
            stack_models: list of models whose predictions are used for stacking.
            transformer: name of transformer to apply to passthrough X.
            passthrough: When False, only the predictions of ``stack_models`` will be included.
                When True, data includes predictions as well as the original training data.
            response_method: Specifies whether to use predict_proba or predict for stacking.
        """
        if (passthrough is False) and (transformer is not None):
            raise RuntimeError(f"Passthrough is False but {transformer = }")

        X = self.X
        if transformer is not None: X = self.transform_oof(set_i, transformer)

        if isinstance(stack_models, str): stack_models = (stack_models, )
        if stack_models is None or len(stack_models) == 0:
            if passthrough is False: raise RuntimeError("passthrough=False but no stack_models specified")
            return X

        pred_fn = lambda model: self.get_oof_preds(set_i, model, response_method=response_method)
        return self._stack_preds(
            models=stack_models,
            pred_fn=pred_fn,
            passthrough=passthrough,
            X_passthrough=X,
            response_method=response_method,
        )

    def get_stacked_X_unlabeled(
        self,
        set_i: int,
        fold_i: int,
        stack_models: str | Sequence[str] | None,
        transformer: str | None,
        passthrough: bool,
        response_method: ResponseMethod,
    ) -> pl.DataFrame | None:
        if (passthrough is False) and (transformer is not None):
            raise RuntimeError(f"Passthrough is False but {transformer = }")

        if self.X_unlabeled is None: return None

        X_unlabeled = self.X_unlabeled
        if transformer is not None:
            X_unlabeled = self._transform_new(X=self.X_unlabeled, transformer=transformer, set_i=set_i, fold_i=fold_i)

        if isinstance(stack_models, str): stack_models = (stack_models, )
        if (stack_models is None) or (len(stack_models) == 0):
            if passthrough is False: raise RuntimeError("passthrough=False but no stack_models specified")
            return X_unlabeled

        pred_fn = lambda model: self._predict_new(
            self.X_unlabeled, model, set_i, fold_i, response_method) # pyright:ignore[reportArgumentType]

        return self._stack_preds(
            models=stack_models,
            pred_fn=pred_fn,
            passthrough=passthrough,
            X_passthrough=X_unlabeled,
            response_method=response_method,
        )


    def fit_model(
        self,
        name: str,
        model,
        transformer: str | None = None,
        max_folds: int | None = None,

        stack_models: str | Sequence[str] | None = None,
        passthrough: bool = True,
        response_method: ResponseMethod | Literal['auto'] = "auto",
        use_unlabeled: bool = True,

        fit_fn: Callable[
            [Any, pl.DataFrame, pl.Series, pl.DataFrame | None], python_utils.HasPredict
        ] = lambda model, X, y, X_unlabeled: model.fit(X, y),
    ) -> np.ndarray:
        """Fit model to each fold.

        if both ``transformer`` and ``stack_models`` are specified, the input to ``model`` is
        ``concatenate(transformer(X), stack_models.predict(X))``

        Returns numpy array with per-fold errors (for hyperparameter tuning).

        Args:
            name: unique name for this combination of model, transformer and stacking configuration.
            model: the model to fit.
            transformer: fitted transformer name. Defaults to None.
            max_folds: limits number of folds by merging folds for models that are very slow to fit. Defaults to None.
            stack_models: list of fitted model names to use for stacking. Defaults to None.
            passthrough: When False, only the predictions of ``stack_models`` will be used as training data.
                When True, training data includes predictions as well as the original training data.
            response_method: Specifies whether to call predict_proba or predict on ``stack_models`` for stacking.
                By default uses predict_proba for classification, otherwise predict.
            use_unlabeled: if ``model`` doesn't use unlabeled data, setting this to False skips potentially expensive
                operation of getting ``transformer`` and ``stack_models`` predictions on the unlabeled data.
            fit_fn: Function which fits model to X, y and X_unlabeled dataframes.
                Defaults to ``model.fit(X, y)``.
        """
        start_time = time.time()

        if isinstance(stack_models, str): stack_models = (stack_models, )
        if stack_models is not None: stack_models = tuple(sorted(set(stack_models)))
        if response_method == "auto":
            if self.is_classification(): response_method = 'predict_proba'
            else: response_method = 'predict'

        # create model dir
        model_dir = self.root / "models" / name
        if (model_dir / "done.txt").exists():
            raise RuntimeError(f"{model_dir} already exists, set a different name for a different model")

        model_dir.mkdir(exist_ok=True)
        (model_dir / "predictions").mkdir(exist_ok=True)

        # fit to all folds
        scores = defaultdict(list)
        supports_proba = False
        obj_qualname = None
        obj_repr = None

        fold_set, fold_map = self.fold_set.merge_folds(n_folds=max_folds)
        self.logger.info('Fitting %i models "%s"', fold_set.n_models, name)

        with self._temp_caching_context():

            for set_i, folds in fold_set.items():

                X = self.get_stacked_X(
                    set_i = set_i,
                    stack_models = stack_models,
                    transformer = transformer,
                    passthrough = passthrough,
                    response_method = response_method,
                )

                for fold_i, (train_index, test_index) in folds.items():

                    # get fold samples
                    X_train = X[train_index]
                    y_train = self.y[train_index]
                    X_test = X[test_index]
                    y_test = self.y[test_index]

                    # fit
                    fitted_model_dir = model_dir / f"model-{set_i}-{fold_i}.joblib"
                    if fitted_model_dir.exists():
                        self.logger.warning("%s already exists, it will be loaded", str(fitted_model_dir))
                        fitted_model = self._cached_load(fitted_model_dir, joblib.load)
                    else:

                        if use_unlabeled and self.X_unlabeled is not None:
                            X_unlabeled_fold = self.get_stacked_X_unlabeled(
                                set_i=set_i,
                                fold_i=fold_i,
                                stack_models=stack_models,
                                transformer=transformer,
                                passthrough=passthrough,
                                response_method=response_method
                            )
                        else:
                            X_unlabeled_fold = None

                        fitted_model = fit_fn(model, X_train, y_train, X_unlabeled_fold)

                        if fitted_model is None: # pyright:ignore[reportUnnecessaryComparison]
                            raise RuntimeError(f"fit_fn for {name} returned None. Make sure model.fit returns self.")

                        joblib.dump(fitted_model, fitted_model_dir, compress=3)

                    obj_qualname = python_utils.get_qualname(fitted_model)
                    obj_repr = repr(fitted_model)[:10_000]

                    # score
                    preds_train = np.asarray(fitted_model.predict(X_train))
                    preds_test = np.asarray(fitted_model.predict(X_test))

                    preds_train = _fitter_utils._validate_preds(preds_train, X_train.shape[0], self.n_targets)
                    preds_test = _fitter_utils._validate_preds(preds_test, X_test.shape[0], self.n_targets)

                    proba_train = proba_test = None
                    if self.is_classification() and hasattr(fitted_model, "predict_proba"):
                        try:
                            proba_train = np.asarray(getattr(fitted_model, "predict_proba")(X_train))
                            proba_test = np.asarray(getattr(fitted_model, "predict_proba")(X_test))

                            proba_train = _fitter_utils._validate_probas(proba_train, X_train.shape[0], self.n_classes)
                            proba_test = _fitter_utils._validate_probas(proba_test, X_test.shape[0], self.n_classes)

                            supports_proba = True
                        except (NotImplementedError, AttributeError):
                            pass

                    score_train, error_train = self.scorer.score_and_error(
                        targets=y_train.to_numpy(), preds=preds_train, proba=proba_train)

                    score_test, error_test = self.scorer.score_and_error(
                        targets=y_test.to_numpy(), preds=preds_test, proba=proba_test)

                    scores["score_train"].append(float(score_train))
                    scores["score_test"].append(float(score_test))
                    scores["error_train"].append(float(error_train))
                    scores["error_test"].append(float(error_test))

                    np.savez_compressed(model_dir / "predictions" / f"test_index-{set_i}-{fold_i}.npz", data=test_index)
                    np.savez_compressed(model_dir / "predictions" / f"test_preds-{set_i}-{fold_i}.npz", data=preds_test)

                    if (proba_train is not None) and (proba_test is not None):
                        np.savez_compressed(model_dir / "predictions" / f"test_proba-{set_i}-{fold_i}.npz", data=proba_test)

                    (self.logger.info if self.per_fold_info else self.logger.debug)(
                        "Set %i fold %i - %s: train = %.8f, test = %.8f",
                        set_i, fold_i, self.scorer.name, float(score_train), float(score_test))

        fit_sec = time.time() - start_time
        config = {
            "transformer": transformer,
            "stack_models": stack_models,
            "passthrough": passthrough,
            "response_method": response_method,
            "fold_map": fold_map,
            "supports_proba": supports_proba,
            "n_models": fold_set.n_models,
            "start_time": start_time,
            "fit_sec": fit_sec,
            "obj_qualname": obj_qualname,
            **scores,
            **{f"{k}_mean": np.mean(v) for k,v in scores.items()},
        }

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        assert obj_repr is not None
        with open(model_dir / "repr.txt", "w", encoding='utf-8') as f:
            f.write(obj_repr)

        with open(model_dir / "done.txt", "w", encoding='utf-8') as f:
            f.write("done")

        self.logger.info(
            "Mean %s: train = %.8f; test = %.8f; Took %.2f seconds",
            self.scorer.name, np.mean(scores["score_train"]), np.mean(scores["score_test"]), fit_sec)

        return np.array(scores["error_test"])

    def fit_transformer(
        self,
        name: str,
        transformer,
        use_folds: bool,
        max_folds: int | None = None,

        pre_transformer: str | None = None,
        stack_models: str | Sequence[str] | None = None,
        passthrough: bool = True,
        response_method: ResponseMethod | Literal['auto'] = "auto",
        use_unlabeled: bool = True,

        fit_fn: Callable[
            [Any, pl.DataFrame, pl.Series, pl.DataFrame | None], python_utils.HasPredict
        ] = lambda transformer, X, y, X_unlabeled: transformer.fit(X, y),
    ) -> None:
        """Fits a feature transformer.

        if both ``pre_transformer`` and ``stack_models`` are specified, the input to ``transformer`` is
        ``concatenate(pre_transformer(X), stack_models.predict(X))``

        Args:
            name: unique name for this combination of transformer, pre-transformer and stacking configuration.
            transformer: the transformer to fit.
            use_folds: whether to use folds, if False, fits transformer to all data in each fold set.
                This should be set to true for transformers that use the label to avoid leakage.
            max_folds: limits number of folds by merging folds for transformers that are very slow to fit. Defaults to None.
            pre_transformer: fitted transformer name, whose output will be passed to this transformer. Defaults to None.
            stack_models: list of fitted model names to use for stacking. Defaults to None.
            passthrough: When False, only the predictions of ``stack_models`` will be used as training data.
                When True, training data includes predictions as well as the original training data.
            response_method: Specifies whether to call predict_proba or predict on ``stack_models`` for stacking.
                By default uses predict_proba for classification, otherwise predict.
            use_unlabeled: if ``model`` doesn't use unlabeled data, setting this to False skips potentially expensive
                operation of getting ``pre_transformer`` and ``stack_models`` predictions on the unlabeled data.
            fit_fn: Function which fits transformer to X, y and X_unlabeled dataframes.
                For transformers that don't use labels it may be beneficial to fit ``stack(X, X_unlabeled)``.
                Defaults to ``transformer.fit(X, y)``.
        """
        start_time = time.time()

        if isinstance(stack_models, str): stack_models = (stack_models, )
        if stack_models is not None: stack_models = tuple(sorted(set(stack_models)))
        if response_method == "auto":
            if self.is_classification(): response_method = 'predict_proba'
            else: response_method = 'predict'

        # create transformer dir
        transformer_dir = self.root / "transformers" / name
        if (transformer_dir / "done.txt").exists():
            raise RuntimeError(f"{transformer_dir} already exists, set a different name for a different transformer")

        transformer_dir.mkdir(exist_ok=True)

        obj_qualname = None
        obj_repr = None
        with self._temp_caching_context():

            fold_map = None
            if use_folds:

                # fit with folds
                fold_set, fold_map = self.fold_set.merge_folds(n_folds=max_folds)
                n_transformers = fold_set.n_models
                self.logger.info('Fitting %i transformers "%s"', n_transformers, name)
                (transformer_dir / "test_indexes").mkdir()

                for set_i, folds in fold_set.items():

                    X = self.get_stacked_X(
                        set_i = set_i,
                        stack_models = stack_models,
                        transformer = pre_transformer,
                        passthrough = passthrough,
                        response_method = response_method,
                    )

                    for fold_i, (train_index, test_index) in folds.items():

                        fitted_dir = transformer_dir / f"transformer-{set_i}-{fold_i}.joblib"
                        if fitted_dir.exists():
                            self.logger.warning("%s already exists, skipping", str(fitted_dir))
                            continue

                        X_train = X[train_index]
                        y_train = self.y[train_index]

                        if use_unlabeled and self.X_unlabeled is not None:
                            X_unlabeled_fold = self.get_stacked_X_unlabeled(
                                set_i=set_i,
                                fold_i=fold_i,
                                stack_models=stack_models,
                                transformer=pre_transformer,
                                passthrough=passthrough,
                                response_method=response_method
                            )
                        else:
                            X_unlabeled_fold = None


                        fitted_transformer = fit_fn(transformer, X_train, y_train, X_unlabeled_fold)
                        obj_qualname = python_utils.get_qualname(fitted_transformer)
                        obj_repr = repr(fitted_transformer)[:10_000]

                        joblib.dump(fitted_transformer, fitted_dir, compress=3)
                        np.savez_compressed(
                            transformer_dir / "test_indexes" / f"test_index-{set_i}-{fold_i}.npz", data=test_index)

            else:
                # fit without folds
                n_transformers = self.fold_set.n_fold_sets
                self.logger.info('Fitting %i transformers "%s" without folds', n_transformers, name)
                for set_i, _ in self.fold_set.items():

                    X = self.get_stacked_X(
                        set_i = set_i,
                        stack_models = stack_models,
                        transformer = pre_transformer,
                        passthrough = passthrough,
                        response_method = response_method,
                    )

                    fitted_dir = transformer_dir / f"transformer-{set_i}-ALL.joblib"
                    if fitted_dir.exists():
                        self.logger.warning("%s already exists, skipping", str(fitted_dir))

                    else:

                        if use_unlabeled and self.X_unlabeled is not None:
                            X_unlabeled_fold = self.get_stacked_X_unlabeled(
                                set_i=set_i,
                                fold_i=0,  # we can't meaningfully average dataframes, so just take 1st fold
                                stack_models=stack_models,
                                transformer=pre_transformer,
                                passthrough=passthrough,
                                response_method=response_method
                            )
                        else:
                            X_unlabeled_fold = None


                        fitted_transformer = fit_fn(transformer, X, self.y, X_unlabeled_fold)
                        obj_qualname = python_utils.get_qualname(fitted_transformer)
                        obj_repr = repr(fitted_transformer)[:10_000]
                        joblib.dump(fitted_transformer, fitted_dir, compress=3)


        if obj_qualname is None:
            obj_qualname = python_utils.get_qualname(transformer)
            obj_repr = repr(transformer)[:10_000]

        fit_sec = time.time() - start_time

        config = {
            "pre_transformer": pre_transformer,
            "stack_models": stack_models,
            "passthrough": passthrough,
            "response_method": response_method,
            "use_folds": use_folds,
            "fold_map": fold_map,
            "n_transformers": n_transformers,
            "start_time": start_time,
            "fit_sec": fit_sec,
            "obj_qualname": obj_qualname,
        }

        with open(transformer_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        assert obj_repr is not None
        with open(transformer_dir / "repr.txt", "w", encoding='utf-8') as f:
            f.write(obj_repr)

        with open(transformer_dir / "done.txt", "w", encoding='utf-8') as f:
            f.write("done")

        self.logger.info("Took %.2f seconds", fit_sec)

    # --------------------------------- inference -------------------------------- #

    def _transform_new(
        self,
        X: pl.DataFrame,
        transformer: str,
        set_i: int,
        fold_i: int,
    ) -> pl.DataFrame:
        """Transform new data using models and transformers from one fold.

        Args:
            X: new data
            transformer: tranformer name.
            set_i: index of fold set.
            fold_i: index of fold.
        """
        self.logger.debug("_transform_new: transformer=%s, set_i=%i, fold_i=%i", transformer, set_i, fold_i)

        transformer_dir = self.root / "transformers" / transformer
        if not transformer_dir.exists():
            raise FileNotFoundError(f'Transformer "{transformer}" doesn\'t exist in "{transformer_dir.parent}"')

        with open(transformer_dir / "config.json", "r", encoding='utf-8') as f:
            config = json.load(f)

        if config["use_folds"]: mapped_fold_i = str(config["fold_map"][str(fold_i)])
        else: mapped_fold_i = "ALL"

        cache_key = f"{transformer}--{set_i}--{mapped_fold_i}"
        if self._temp_caching_enabled:
            if cache_key in self._temp_transform_cache:
                self.logger.debug("Loading cached transformed dataframe from %s", self._temp_transform_cache[cache_key])
                assert self._tmpdir is not None
                try:
                    return pl.read_parquet(self._temp_transform_cache[cache_key])
                except Exception as e:
                    self.logger.warning("Failed to load %s", self._temp_transform_cache[cache_key])
                    self.logger.warning("%r", e)

        fitted_transformer = self._cached_load(transformer_dir / f"transformer-{set_i}-{mapped_fold_i}.joblib", joblib.load)

        stack_models = config["stack_models"]
        if hasattr(fitted_transformer, "__myautoml_used_models__"):
            used_models = getattr(fitted_transformer, "__myautoml_used_models__")()

            if used_models is not None:
                used_models = [str(m) for m in used_models] # makes sure its not np.str
                self.logger.debug('Overriding stack_models of transformer "%s" with __myautoml_used_models__', transformer)
                self.logger.debug("Old value: %r", stack_models)
                self.logger.debug("New value: %r", used_models)

            if used_models is not None: stack_models = used_models

        X = self._get_stacked_X_new(
            X=X,
            transformer=config["pre_transformer"],
            set_i=set_i,
            fold_i=fold_i,
            stack_models=stack_models,
            passthrough=config["passthrough"],
            response_method=config["response_method"],
        )

        start = time.perf_counter()
        transformed = polars_utils.to_dataframe(fitted_transformer.transform(X))

        if self._temp_caching_enabled:
            sec = time.perf_counter() - start
            min_sec = max(_fitter_utils._min_fit_sec_for_caching(X) / 500, 1)

            if sec > min_sec:
                assert self._tmpdir is not None

                assert cache_key not in self._temp_transform_cache
                save_dir = os.path.join(self._tmpdir, f'{cache_key}.parquet')

                if python_utils.get_folder_size_bytes(self._tmpdir) / 1e6 < self.max_cache_mb:
                    try:
                        self.logger.debug("Saving cached transformed dataframe to %s: %.5f > %.5f", save_dir, sec, min_sec)
                        transformed.write_parquet(save_dir)
                        self._temp_transform_cache[cache_key] = save_dir

                    except Exception as e:
                        self.logger.warning("Failed to save %s", save_dir)
                        self.logger.warning("%r", e)
                        self._temp_transform_cache.pop(cache_key, None)

                else:
                    self.logger.debug("Skipped caching dataframe %s: max cache size exceeded", save_dir)

            else:
                self.logger.debug(
                    "Skipped caching dataframe with %i elements: %.5f <= %.5f", math.prod(X.shape), sec, min_sec)

        return transformed

    def _predict_new(
        self,
        X: pl.DataFrame,
        model: str,
        set_i: int,
        fold_i: int,
        response_method: ResponseMethod,
    ) -> np.ndarray:
        """Predict on new data using models and transformers from one fold.

        Args:
            X: new data
            model: model name.
            set_i: index of fold set.
            fold_i: index of fold.
            response_method: what to call on ``model``.
        """
        self.logger.debug("_predict_new: model=%s, set_i=%i, fold_i=%i, response_method=%s",
                          model, set_i, fold_i,  response_method)

        model_dir = self.root / "models" / model
        if not model_dir.exists():
            raise FileNotFoundError(f'Model "{model}" doesn\'t exist in "{model_dir.parent}"')

        with open(model_dir / "config.json", "r", encoding='utf-8') as f:
            config = json.load(f)

        mapped_fold_i = config["fold_map"][str(fold_i)]

        cache_key = (model, set_i, mapped_fold_i, response_method)
        if self._temp_caching_enabled:
            if cache_key in self._temp_predict_cache:
                self.logger.debug("Loading cached predictions under key %r", cache_key)
                return self._temp_predict_cache[cache_key]

        fitted_model = self._cached_load(model_dir / f"model-{set_i}-{mapped_fold_i}.joblib", joblib.load)

        stack_models = config["stack_models"]
        if hasattr(fitted_model, "__myautoml_used_models__"):
            used_models = getattr(fitted_model, "__myautoml_used_models__")()

            if used_models is not None:
                used_models = [str(m) for m in used_models] # makes sure its not np.str
                self.logger.debug('Overriding stack_models of model "%s" with __myautoml_used_models__', model)
                self.logger.debug("Old value: %r", stack_models)
                self.logger.debug("New value: %r", used_models)

            if used_models is not None: stack_models = used_models

        X = self._get_stacked_X_new(
            X=X,
            transformer=config["transformer"],
            set_i=set_i,
            fold_i=fold_i,
            stack_models=stack_models,
            passthrough=config["passthrough"],
            response_method=config["response_method"], # this refers to response method of stack models.
        )

        start = time.perf_counter()
        if response_method == "predict_proba":
            if config["supports_proba"]:
                y = np.asarray(fitted_model.predict_proba(X))
            else:
                y = np.asarray(fitted_model.predict(X))
                y = numpy_utils.one_hot(y, self.n_classes)

        else:
            y = np.asarray(getattr(fitted_model, response_method)(X))

        if self._temp_caching_enabled:
            sec = time.perf_counter() - start
            min_sec = max(_fitter_utils._min_fit_sec_for_caching(y) / 1000, 1)

            if sec > min_sec:
                assert cache_key not in self._temp_predict_cache
                self.logger.debug("Storing cached predictions under key %r: %.5f > %.5f", cache_key, sec, min_sec)
                self._temp_predict_cache[cache_key] = y

            else:
                self.logger.debug(
                    "Skipped caching preds with %i elements: %.5f <= %.5f", math.prod(y.shape), sec, min_sec)

        return y

    def _get_stacked_X_new(
        self,
        X: pl.DataFrame,
        set_i: int,
        fold_i: int,
        stack_models: str | Sequence[str] | None,
        transformer: str | None,
        passthrough: bool,
        response_method: ResponseMethod,
    ) -> pl.DataFrame:
        """Creates stacked X from new data.

        Args:
            X: new data
            set_i: index of fold set.
            fold_i: index of fold.
            stack_models: list of fitted model names to use for stacking. Defaults to None.
            transformer: name of transformer to apply to passthrough X.
            passthrough: When False, only the predictions of ``stack_models`` will be used as training data.
                When True, training data includes predictions as well as the original training data.
            response_method: Specifies whether to call ``predict_proba`` or ``predict`` on ``stack_models`` for stacking.
                By default uses ``predict_proba`` for classification, otherwise ``predict``.
        """
        self.logger.debug(
            "_get_stacked_X_new: set_i=%i, fold_i=%i, stack_models=%r, transformer=%r, passthrough=%r, response_method=%s",
            set_i, fold_i, stack_models, transformer, passthrough, response_method)

        if (passthrough is False) and (transformer is not None):
            raise RuntimeError(f"Passthrough is False but {transformer = }")

        X_tfm = X
        if transformer is not None:
            X_tfm = self._transform_new(X=X, transformer=transformer, set_i=set_i, fold_i=fold_i)

        if isinstance(stack_models, str): stack_models = (stack_models, )
        if stack_models is None or len(stack_models) == 0: return X_tfm

        pred_fn = lambda model: self._predict_new(
            X=X, set_i=set_i, fold_i=fold_i, model=model, response_method=response_method)

        return self._stack_preds(
            models=stack_models,
            pred_fn=pred_fn,
            passthrough=passthrough,
            X_passthrough=X_tfm,
            response_method=response_method,
        )


    def _predict_numpy(self, X: pl.DataFrame | Any, model: str, response_method: ResponseMethod):
        """
        Args:
            X: data to predict labels for.
            model: name of the model to use for prediction.
            response_method: specifies method to call on ``model``,
                by default uses ``predict_proba`` or ``predict`` if it is not available for classification,
                and ``predict`` for regression. Defaults to 'auto'.
        """
        self.auto_encoder.validate_data(X)
        X = self.auto_encoder.transform_X(X)

        if self.is_classification():
            assert response_method != "predict"

        with self._temp_caching_context():
            preds = None
            n = 0
            for set_i in range(self.fold_set.n_fold_sets):
                for fold_i in range(self.fold_set.n_folds):

                    if preds is None: preds = self._predict_new(X, model, set_i, fold_i, response_method)
                    else: preds = preds + self._predict_new(X, model, set_i, fold_i, response_method)
                    n += 1

        assert preds is not None
        preds = preds / n

        return preds

    def predict_proba(self, X: pl.DataFrame | Any, model: str):
        """Predict probabilities (n_samples by n_classes) on specified data.

        Args:
            X: data to predict labels for.
            model: name of the model to use for prediction.

        """
        return self._predict_numpy(X, model, 'predict_proba')

    def predict(self, X: pl.DataFrame | Any, model: str):
        """Predicts the label for specified data and returns as a Series.

        Args:
            X: data to predict labels for.
            model: name of the model to use for prediction.
        """
        if self.is_classification():
            probas = self._predict_numpy(X, model, response_method='predict_proba')
            preds = np.argmax(probas, -1)
        else:
            preds = self._predict_numpy(X, model, response_method='predict')

        return self.auto_encoder.inverse_transform_y(preds)
