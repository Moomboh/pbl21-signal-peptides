from abc import ABC, abstractmethod
from typing import List

from tensorflow import keras
import pandas as pd
import pickle

from .metrics import Accuracy, Precision, Recall, MCC

allowed_metrics = ["accuracy", "precision", "recall", "mcc"]

metric_dict = {
    "accuracy": Accuracy(),
    "precision": Precision("micro"),
    "recall": Recall("micro"),
    "mcc": MCC()
}

class MultiMetric(ABC):

    def __init__(self, metrics: List[str]):
        for metric in metrics:
            if metric not in allowed_metrics:
                raise ValueError(f"Metric '{metric}' not supported")
        self._metrics = {name: metric_dict[name] for name in metrics}
        self._values = None


class Serializable:

    @classmethod
    def fromFile(cls, filepath: str):
        m = cls()
        with open(filepath, "rb") as f:
            attributes = pickle.load(f)
            for key, value in attributes.items():
                m.__dict__[key] = value
        return m

    def serialize(self, filepath: str):
        with open(filepath, "wb+") as f:
            pickle.dump(self.__dict__, f)


class DataframeOutput(ABC):
    @abstractmethod
    def toDataframe(self) -> pd.DataFrame:
        raise NotImplementedError


