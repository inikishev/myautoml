import numpy as np

def one_hot(preds: np.ndarray, n_classes: int):
    if not np.issubdtype(preds.dtype, np.integer):
        if not np.allclose((preds % 1).sum(), 0):
            raise TypeError(f"Can't one-hot encode array of dtype {preds.dtype}")

    n_samples = preds.shape[0]
    onehot = np.zeros((preds.size, n_classes))
    onehot[np.arange(n_samples), preds.astype(np.uint64)] = 1
    return onehot.reshape((*preds.shape, n_classes))