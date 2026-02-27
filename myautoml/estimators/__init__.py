from importlib.util import find_spec
from typing import TYPE_CHECKING

from . import optuna_presets
from .hill_climbing import (
    HillClimbingEnsembleClassifier,
    HillClimbingEnsembleRegressor,
    HillClimbingEnsembleSelector,
)
from .interaction import (
    InteractionFeatures,
    interaction_copysign,
    interaction_divide,
    interaction_exp,
    interaction_floordiv,
    interaction_log,
    interaction_max_neg,
    interaction_min_neg,
    interaction_mod,
)
from .kernel_approximation import LaplaceRFF
from .ridge_proba import RidgeClassifierProba, RidgeClassifierProbaCV
from .utility import NanToNum, ToDtype, ToList, ToPandas
from .weighted_ensemble import (
    GreedyWeightedEnsembleClassifier,
    GreedyWeightedEnsembleRegressor,
    GreedyWeightedEnsembleSelector,
)

if TYPE_CHECKING or find_spec("torch") is not None:
    from .df_linear import DFLinearClassifier, DFLinearRegressor
    from .learnable_elm import LearnableELMClassifier, LearnableELMRegressor
    from .mimic import DiagMimic, EighMimic, IterativeMimic
    from .to_cuda import ToCUDA
    from .bisine import BisineClassifier, BisineRegressor
