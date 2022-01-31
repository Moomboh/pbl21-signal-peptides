from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from sklearn.metrics import precision_score, matthews_corrcoef, accuracy_score, recall_score
import tensorflow as tf
from tensorflow import keras

class Baseclass(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self._function(y_true.flatten(), y_pred.flatten())


class WeightedAverage(Baseclass):
    def __init__(self, weight: Literal["binary", "macro", "micro"]):
        self.weight = weight

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self._function(y_true.flatten(), y_pred.flatten(), average=self.weight)


class Accuracy(Baseclass):
    def __init__(self):
        self._function = accuracy_score
        self.name = "accuracy"


class Precision(WeightedAverage):
    def __init__(self, weight: Literal["binary", "macro", "micro"]):
        super().__init__(weight)
        self._function = precision_score
        self.name = "precision"


class Recall(WeightedAverage):
    def __init__(self, weight: Literal["binary", "macro", "micro"]):
        super().__init__(weight)
        self._function = recall_score
        self.name = "recall"


class MCC(Baseclass):
    def __init__(self):
        self._function = matthews_corrcoef
        self.name = "mcc"
