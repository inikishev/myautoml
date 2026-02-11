from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright:ignore[reportAttributeAccessIssue]
)

from ..utils.torch_utils import TORCH_INSTALLED


class ToCUDA(BaseEstimator, TransformerMixin):
    """Moves inputs to CUDA for Array API-compatible estimators.
    Inputs must be numeric and will be converted to a single dtype.

    Before importing sklearn, run ``os.environ["SCIPY_ARRAY_API"] = 1``, and use
    ``with config_context(array_api_dispatch=True):``
    """

    def fit(self, X, y=None):
        if not TORCH_INSTALLED: raise RuntimeError("PyTorch needs to be installed to use ToCUDA")
        X, y = validate_data(self, X=X, y=y, ensure_all_finite=False)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, ensure_all_finite=False)

        import torch
        return torch.as_tensor(X, device='cuda')
