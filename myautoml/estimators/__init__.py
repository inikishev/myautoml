from importlib.util import find_spec
from typing import TYPE_CHECKING

from . import optuna_presets
from .features import UnsupervisedFeatures
from .hill_climbing import (
    HillClimbingEnsembleClassifier,
    HillClimbingEnsembleRegressor,
    HillClimbingEnsembleSelector,
)
from .kernel_approximation import LaplaceRFF
from .ridge_classifier import RidgeClassifierCV
from .utility import ToCUDA
from .weighted_ensemble import (
    GreedyWeightedEnsembleClassifier,
    GreedyWeightedEnsembleRegressor,
    GreedyWeightedEnsembleSelector,
)

if TYPE_CHECKING or find_spec("torch") is not None:
    from .df_linear import DFLinearClassifier, DFLinearRegressor
    from .learnable_elm import LearnableELMClassifier, LearnableELMRegressor
