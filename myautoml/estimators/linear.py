import time
import logging
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

class _BaseDFLinear(BaseEstimator):
    """Fit a linear model to optimize a score directly using gradient-free optimization, only use for small number of features and target classes."""
    def __init__(self, scorer): ...