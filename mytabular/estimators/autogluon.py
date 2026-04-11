from autogluon.tabular import TabularPredictor
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
import copy
from .utility import ToPandas

class _BaseAutoGluon(BaseEstimator):
    def __init__(self, init_kwargs: dict, fit_kwargs: dict, cls: Any=TabularPredictor):
        self.cls = cls
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs

    def fit(self, X, y):
        _, y = validate_data(self, X=X, y=y)
        self.to_pandas_ = ToPandas().fit(X)
        X = self.to_pandas_.transform(X)

        label_name = "label"
        while label_name in list(X.columns): label_name = f"{label_name}_"
        df = X.assign(**{label_name: y})

        self.predictor_ = self.cls(label=label_name, **self.init_kwargs)
        self.predictor_.fit(df, **self.fit_kwargs)
        return self

    def predict(self, X):
        validate_data(self, X=X, reset=False)

        X = self.to_pandas_.transform(X)
        return self.predictor_.predict(X)

    def predict_proba(self, X):
        validate_data(self, X=X, reset=False)

        X = self.to_pandas_.transform(X)
        return self.predictor_.predict_proba(X)

class AutoGluonClassifier(ClassifierMixin, _BaseAutoGluon):
    """
    useful kwargs

    Init kwargs:
        eval_metric: evaluation metric

    Fit kwargs:
        time_limit: time limit in seconds
        presets: presets like "high_quality"
        hyperparameters: default, zeroshot, zeroshot_2025_tabfm, light, very_light, toy, multimodal.
            or dict with models to include.
        ag_args_ensemble: dict with num_folds, set save_bag_folds, fold_fitting_strategy.
        fit_strategy: parallel or sequential
        dynamic_stacking: enables dynamic stacking
        auto_stack: Automatically sets num_bag_folds and num_stack_levels
        num_bag_folds: number of folds
        num_bag_sets: number of fold sets
        num_stack_levels: number of stack levels
        included_model_types: models to train
        excluded_model_types: model to not train
        refit_full: refits on full data at the end making estimator smaller but slightly worse
        save_bag_folds: increases disk usage and decreases RAM usage
        save_space: deletes models unused in final ensemble after fit (doesn't affect accuracy)
        verbosity: verbosity level
    """
    def __init__(self, init_kwargs: dict, fit_kwargs: dict, cls: Any=TabularPredictor):
        super().__init__(init_kwargs, fit_kwargs, cls)

class AutoGluonRegressor(RegressorMixin, _BaseAutoGluon):
    """
    useful kwargs

    Init kwargs:
        eval_metric: evaluation metric

    Fit kwargs:
        time_limit: time limit in seconds
        presets: presets like "high_quality"
        hyperparameters: default, zeroshot, zeroshot_2025_tabfm, light, very_light, toy, multimodal.
            or dict with models to include.
        ag_args_ensemble: dict with num_folds, set save_bag_folds, fold_fitting_strategy.
        fit_strategy: parallel or sequential
        dynamic_stacking: enables dynamic stacking
        auto_stack: Automatically sets num_bag_folds and num_stack_levels
        num_bag_folds: number of folds
        num_bag_sets: number of fold sets
        num_stack_levels: number of stack levels
        included_model_types: models to train
        excluded_model_types: model to not train
        refit_full: refits on full data at the end making estimator smaller but slightly worse
        save_bag_folds: increases disk usage and decreases RAM usage
        save_space: deletes models unused in final ensemble after fit (doesn't affect accuracy)
        verbosity: verbosity level
    """
    def __init__(self, init_kwargs: dict, fit_kwargs: dict, cls: Any=TabularPredictor):
        super().__init__(init_kwargs, fit_kwargs, cls)
