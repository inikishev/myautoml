# pylint:disable=wrong-import-order
"""my imports. More useful things
```python
pl.Config.set_fmt_str_lengths(200)
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```
"""
import os

import matplotlib.pyplot as plt

# from sklearnex import patch_sklearn, config_context
# patch_sklearn()
import numpy as np
import polars as pl
import seaborn as sns
import sklearn.calibration
import sklearn.cluster
import sklearn.compose
import sklearn.covariance
import sklearn.cross_decomposition
import sklearn.datasets
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.feature_selection
import sklearn.gaussian_process
import sklearn.impute
import sklearn.isotonic
import sklearn.kernel_approximation
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.manifold
import sklearn.metrics
import sklearn.mixture
import sklearn.model_selection
import sklearn.multiclass
import sklearn.multioutput
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.random_projection
import sklearn.semi_supervised
import sklearn.svm
import sklearn.tree
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from xgboost import (
    XGBClassifier,
    XGBRanker,
    XGBRegressor,
    XGBRFClassifier,
    XGBRFRegressor,
)
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

import myautoml as ma
