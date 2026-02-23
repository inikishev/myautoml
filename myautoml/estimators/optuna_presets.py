
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    import optuna

def suggest_xgb_params(
    trial: "optuna.Trial",
    search_space_size: Literal[1,2] = 1
):
    """Also use ``{"tree_method": "hist", "device": "cuda", "seed": 0}``"""

    params = {
        "eta": trial.suggest_float("eta", 1e-4, 1.0, log=True), # Step size shrinkage, default 0.3 [0,1]
        "gamma": trial.suggest_float("gamma", 1e-3, 1e3, log=True) - 1e-3, # Minimum loss reduction, default 0 [0,∞]
        "max_depth": trial.suggest_int("max_depth", 1, 16), # default=6  [0,∞]
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 100, log=True) - 1e-2, # default=1 [0,∞]
        "subsample": trial.suggest_float("subsample", 1e-2, 1.0), # default=1 (0,1]
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 1e-2, 1.0), # default=1 (0, 1]
    }

    if search_space_size >= 2:
        params.update({
            "max_delta_step": trial.suggest_float("max_delta_step", 0, 10), # default=0 [0,∞]
            "sampling_method": trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1000, log=True) - 1e-8, # L2 regularization default=1
            "alpha": trial.suggest_float("alpha", 1e-8, 1000, log=True) - 1e-8, # L1 regularization default=0
            "max_leaves": trial.suggest_categorical("max_leaves", [0, 64, 128, 256]), # default=0
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 1e-2, 1.0),
        })

    if search_space_size >= 3:
        params.update({
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.1, 10, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1, 2000)
        })

    elif params["eta"] < 0.1:
        params["n_estimators"] = 2000

    return params