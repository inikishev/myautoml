from importlib.util import find_spec
from typing import TYPE_CHECKING

from . import optuna_presets
from .hill_climbing import (
    HillClimbingEnsembleClassifier,
    HillClimbingEnsembleRegressor,
    HillClimbingEnsembleSelector,
)
from .kernel_approximation import LaplaceRFF
from .ridge_proba import RidgeClassifierProba, RidgeClassifierProbaCV
from .weighted_ensemble import (
    GreedyWeightedEnsembleClassifier,
    GreedyWeightedEnsembleRegressor,
    GreedyWeightedEnsembleSelector,
)

from .utility import ToDtype, ToPandas, ToList
from .interaction import InteractionFeatures, interaction_copysign, interaction_divide, interaction_exp, interaction_log, interaction_max_neg, interaction_min_neg

if TYPE_CHECKING or find_spec("torch") is not None:
    from .df_linear import DFLinearClassifier, DFLinearRegressor
    from .learnable_elm import LearnableELMClassifier, LearnableELMRegressor
    from .to_cuda import ToCUDA
    from .mimic import DiagMimic, IterativeMimic, EighMimic