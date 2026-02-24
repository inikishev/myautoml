import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

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
