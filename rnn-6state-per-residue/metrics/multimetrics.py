from typing import List

import pandas as pd

from .baseclasses import MultiMetric, Serializable, DataframeOutput
from constants import METRIC_KINGDOMS, TRAINING_PARTITIONS, METRIC_TYPES

class MetricsBundle(MultiMetric, Serializable, DataframeOutput):

    def __init__(self, epochs: int, metrics: List[str]):
        super().__init__(metrics)
        self._initialize(epochs)

    def _initialize(self, epochs: int) -> None:
        self._values = pd.DataFrame(
            [
                (metric, type, kingdom, i, None)
                for metric in self._metrics.keys()
                for type in METRIC_TYPES
                for kingdom in METRIC_KINGDOMS
                for i in range(1, epochs+1)
            ]
        )
        self._values.columns = ["metric", "type", "kingdom", "epoch", "value"]
        self._values.set_index(["metric", "type", "kingdom", "epoch"], inplace=True)

    def addEpoch(self, fold: int, type: str, kingdom: str, y_pred, y_true) -> None:
        for name, metric in self._metrics.items():
            self._values[name][fold][type][kingdom].append(metric.compute(y_pred, y_true))

            self.epochs = len(self._values[name][fold][type][kingdom])  # TODO make more efficient

    def addNullEpoch(self, fold: int, type: str, kingdom: str) -> None:
        for name, metric in self._metrics.items():
            self._values[name][fold][type][kingdom].append(None)

            self.epochs = len(self._values[name][fold][type][kingdom])  # TODO make more efficient

    def toDataframe(self) -> pd.DataFrame:
        return self._values.copy(deep=True)
