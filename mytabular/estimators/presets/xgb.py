
XGB_PRESETS = {
    # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2025.py,
    "default": {"n_estimators": 10000, "learning_rate": 0.1, "booster": "gbtree"},
    "AG_zeroshot_2025_1": {
        "n_estimators": 10000, "booster": "gbtree",
        "colsample_bylevel": 0.9213705632288,
        "colsample_bynode": 0.6443385965381,
        "enable_categorical": True,
        "grow_policy": "lossguide",
        "learning_rate": 0.0068171645251,
        "max_cat_to_onehot": 8,
        "max_depth": 6,
        "max_leaves": 10,
        "min_child_weight": 0.0507304250576,
        "reg_alpha": 4.2446346389037,
        "reg_lambda": 1.4800570021253,
        "subsample": 0.9656290596647,
    },
    "AG_zeroshot_2025_2": {
        "n_estimators": 10000, "booster": "gbtree",
        "colsample_bylevel": 0.6377491713202,
        "colsample_bynode": 0.9237625621103,
        "enable_categorical": True,
        "grow_policy": "lossguide",
        "learning_rate": 0.0112462621131,
        "max_cat_to_onehot": 33,
        "max_depth": 10,
        "max_leaves": 35,
        "min_child_weight": 0.1403464856034,
        "reg_alpha": 3.4960653958503,
        "reg_lambda": 1.3062320805235,
        "subsample": 0.6948898835178,
    },
    "S6E2": dict(n_estimators=1000, reg_lambda=10, max_depth=2, tree_method="hist", device="cuda")
}

