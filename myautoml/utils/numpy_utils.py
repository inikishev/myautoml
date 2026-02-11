import numpy as np

def one_hot(preds: np.ndarray):
    num_classes = np.max(preds) + 1
    num_samples = preds.shape[0]
    one_hot = np.zeros((preds.size, num_classes))
    one_hot[np.arange(num_samples), preds.astype(np.uint64)] = 1
    return one_hot.reshape((*preds.shape, num_classes))