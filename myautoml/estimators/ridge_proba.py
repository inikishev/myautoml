import numpy as np
from scipy.special import expit, softmax # pylint:disable=no-name-in-module
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV


def _predict_proba(scores, n_classes):
    if scores.ndim == 1: # scores is (n_samples, )
        pos = expit(scores)
        return np.stack([1-pos, pos], -1)

    # scores is (n_samples, n_classes)
    return softmax(scores, -1)


class RidgeClassifierProba(RidgeClassifier):
    def predict_proba(self, X):
        scores = self.decision_function(X) # returns (n_samples, n_classes) or (n_samples, ) for binary
        return _predict_proba(scores, len(self.classes_))

class RidgeClassifierProbaCV(RidgeClassifierCV):
    def predict_proba(self, X):
        scores = self.decision_function(X) # returns (n_samples, n_classes) or (n_samples, ) for binary
        return _predict_proba(scores, len(self.classes_))