import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
        l = -(np.sum((y_true * np.log(y_pred) + ((1 - y_true) * np.log(1 - y_pred)))) /len(y_true))
        return round(float(l), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:

        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
        l = -(np.sum(y_true * np.log(y_pred)) / len(y_true))
        return round(float(l), 4)
        pass
