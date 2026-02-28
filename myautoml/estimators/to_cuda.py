import copy
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)
from sklearn import config_context
import inspect
CUDA_IF_AVAILABLE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ToCUDA(TransformerMixin, BaseEstimator):
    """Moves inputs to CUDA for Array API-compatible estimators.
    Inputs must be numeric and will be converted to a single dtype.

    Before importing sklearn, run ``os.environ["SCIPY_ARRAY_API"] = 1``, and use
    ``with config_context(array_api_dispatch=True):``
    """
    def __init__(self, device=CUDA_IF_AVAILABLE, dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def fit(self, X, y=None):
        validate_data(self, X=X, y=y, ensure_all_finite=False)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, ensure_all_finite=False)
        return torch.as_tensor(X, device=self.device, dtype=self.dtype)

class CUDAEstimator(BaseEstimator):
    """Before importing sklearn, run ``os.environ["SCIPY_ARRAY_API"] = 1``

    Fits and transforms/predicts on CUDA, but outputs are moved back to CPU.
    This prevents some errors like with some sklearn functionality that doesnt yet support CUDA."""
    def __init__(self, estimator, device=CUDA_IF_AVAILABLE, dtype=torch.float32):
        self.estimator = estimator
        self.device = device
        self.dtype = dtype

    def __sklearn_tags__(self):
        return self.estimator.__sklearn_tags__()

    def fit(self, X, y=None):
        X, y = validate_data(self, X=X, y=y, ensure_all_finite=False)
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if y is not None: y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        with config_context(array_api_dispatch=True): # pyright:ignore[reportGeneralTypeIssues]
            self.estimator_ = copy.copy(self.estimator).fit(X, y)
        self.fitted_ = True
        return self

    def _get_output(self, method, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, ensure_all_finite=False)
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        with config_context(array_api_dispatch=True): # pyright:ignore[reportGeneralTypeIssues]
            out = getattr(self.estimator_, method)(X)
        return out.numpy(force=True)

    def predict(self, X): return self._get_output("predict", X)
    def predict_proba(self, X): return self._get_output("predict_proba", X)
    def decision_function(self, X): return self._get_output("decision_function", X)
    def transform(self, X): return self._get_output("transform", X)
    def apply(self, X): return self._get_output("apply", X)