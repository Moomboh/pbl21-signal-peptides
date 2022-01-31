from typing import List, Tuple
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from constants import TYPES, METRIC_TYPES, KINGDOMS, METRIC_KINGDOMS, annotation_mapping, amino_acid_mapping
from utils.encoding import oneHotToCategorical, sequenceToCategorical, categoricalToOneHot
from metrics.metrics import Accuracy, Recall, Precision, MCC

allowed_metrics = {
    "accuracy": Accuracy(),
    "recall": Recall("micro"),
    "Precision": Precision("micro"),
    "mcc": MCC()
}

class MetricsAfterEpoch(keras.callbacks.Callback):
    def __init__(self, filepath: str, data: Tuple[np.ndarray, np.ndarray], metrics: List[str]):
        super().__init__()
        self.filepath = filepath
        self._metrics = {name: allowed_metrics[name] for name in metrics}
        self._values = {name: [] for name in metrics}
        self.x, self.y = data
        self._epoch = 1

        with open(self.filepath, "w+") as f:
            f.write(','.join(["epoch", *metrics]))
            f.write("\n")

    def on_epoch_end(self, batch, logs=None):
        predictions = self.model.predict(self.x)
        for metric in self._metrics.keys():
            self._values[metric].append(self._metrics[metric](
                oneHotToCategorical(self.y), oneHotToCategorical(predictions)
            ))

        with open(self.filepath, "a") as f:
            f.write(','.join([
                                 str(self._epoch),
                                 *["{:.8f}".format(self._values[metric][self._epoch-1]) for metric in self._metrics.keys()]
            ]))
            f.write("\n")

        self._epoch += 1


class SaveConfusionMatrixAfterEpoch(keras.callbacks.Callback):
    def __init__(self, filepath: str, data: Tuple[List[str], List[str]]):
        super().__init__()
        self.filepath = filepath
        self.x = np.array([sequenceToCategorical(seq, amino_acid_mapping) for seq in data[0]])
        self.y = np.array([sequenceToCategorical(ann, annotation_mapping) for ann in data[1]])
        self._values = []

    def on_epoch_end(self, batch, logs=None):
        y_pred = np.array([oneHotToCategorical(pred) for pred in self.model.predict(self.x)])
        cm = confusion_matrix(self.y.flatten(), y_pred.flatten())
        self._values.append(cm)

    def on_train_end(self, logs=None):
        with open(self.filepath, "wb+") as f:
            pickle.dump(self._values, f)


class StratifiedMetricsAfterTraining(keras.callbacks.Callback):
    def __init__(self, metrics: List[str], data: pd.DataFrame):
        self._data = data
        self._metrics = [allowed_metrics[metric] for metric in metrics]

    def on_train_end(self, logs=None):
        # TODO append prediction column to data DF
        stratified = pd.DataFrame([
            (metric.name, type, kingdom, metric((relevant_data := self._data.query(f"type == {type} and kingdom == {kingdom}"))["prediction"], relevant_data["annotation"]))
            for metric in self.metrics
            for type in TYPES
            for kingdom in KINGDOMS
        ])
        self._values.columns = ["metric", "type", "kingdom", "value"]
        self._values.set_index(["metric", "type", "kingdom"], inplace=True)

        # "overall" values
        aggregated_kingdom = pd.DataFrame([
            (metric.name, type, "overall", metric((relevant_data := self._data.query(f"type == {type}"))["prediction"], relevant_data["annotation"]))
            for metric in self._metrics
            for type in TYPES
        ])

        aggregated_type = pd.DataFrame([
            (metric.name, "overall", kingdom, metric((relevant_data := self._data.query(f"kingdom == {kingdom}"))["prediction"], relevant_data["annotation"]))
            for metric in self.metrics
            for kingdom in KINGDOMS
        ])

        aggregated_kingdom_and_type = pd.DataFrame([
            (metric.name, "overall", "overall", metric(self._data["prediction"], self._data["annotation"]))
            for metric in self.metrics
        ])


class PrintHeaderForEachModel(keras.callbacks.Callback):
    def __init__(self, holdout_fold: int):
        super().__init__()
        self.holdout_fold = holdout_fold

    def on_train_begin(self, logs=None):
        print("\n\n")
        print('------------------------------------------------------------------------')
        print(f'FOLD {self.holdout_fold}/4')
        print('------------------------------------------------------------------------')


class SaveFinalModel(keras.callbacks.Callback):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def on_train_end(self, logs=None):
        self.model.save(self.filename)
