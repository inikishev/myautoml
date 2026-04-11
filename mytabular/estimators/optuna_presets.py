
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    import optuna

def suggest_xgb_params(
    trial: "optuna.Trial",
):
    """Also use ``tree_method="hist", device="cuda", seed=0}``"""


    # Hyperparameter search space
    params = {
        # Tree structure parameters
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 0, 256),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),

        # Learning parameters
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10, log=True),

        # Regularization parameters
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),

        # Subsampling parameters
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        # Other parameters
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
    }

    # XGBoost training parameters with CUDA
    return {
        **params,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # GPU-accelerated with xgboost-cu12
        "enable_categorical": True,
        "device": "cuda",
        "seed": 0,
        "verbosity": 0,
        "num_boost_round": 1000,
        # "early_stopping_rounds": 50,
    }
