import itertools
from typing import List

import numpy as np
import pandas as pd
from tensorflow import keras

from constants import TYPES, METRIC_KINGDOMS, KINGDOMS, TRAINING_PARTITIONS
from metrics import MetricsBundle

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model: keras.Sequential, data: pd.DataFrame, metrics: List[str]):
        self.model = model
        self.metrics = MetricsBundle(*metrics)
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.x)
        for fold in TRAINING_PARTITIONS:
            for type in TYPES:
                for kingdom in METRIC_KINGDOMS:
                    if type == "overall" and kingdom == "overall":
                        test_x = np.array(
                            list(itertools.chain.from_iterable([self.x[fold][type][kingdom] for type in TYPES for kingdom in KINGDOMS])))
                        test_y = np.array(
                            list(itertools.chain.from_iterable([self.y[fold][type][kingdom] for type in TYPES for kingdom in KINGDOMS])))

                    elif kingdom == "overall":
                        test_x = np.array(
                            list(itertools.chain.from_iterable([self.x[fold][type][kingdom] for kingdom in KINGDOMS])))
                        test_y = np.array(
                            list(itertools.chain.from_iterable([self.y[fold][type][kingdom] for kingdom in KINGDOMS])))

                    elif type == "overall":
                        test_x = np.array(
                            list(itertools.chain.from_iterable([self.x[fold][type][kingdom] for type in TYPES])))
                        test_y = np.array(
                            list(itertools.chain.from_iterable([self.y[fold][type][kingdom] for type in TYPES])))

                    else:
                        test_x = np.array(self.x[fold][type][kingdom])
                        test_y = np.array(self.y[fold][type][kingdom])

                    if len(test_x) > 0:
                        scores = self.model.evaluate(test_x, test_y, verbose=0)

                        self.loss[fold][type][kingdom].append(scores[0])

                        # TODO evaluate model-side accuracy (scores[1]) against eval-side accuracy

                        predictions = self.model.predict(test_x, verbose=0)
                        decoded_annotations = decodeAnnotations(test_y)
                        decoded_predictions = decodeAnnotations(predictions)

                        self.metrics.addEpoch(fold, type, kingdom, decoded_predictions, decoded_annotations)

                    else:
                        self.metrics.addNullEpoch(fold, type, kingdom)
                        self.loss[fold][type][kingdom].append(None)

    def on_train_begin(self, logs=None):
        print('\n------------------------------------------------------------------------')
        print(f'FOLD {self.holdout_fold}/4')
        print('------------------------------------------------------------------------')


    def on_train_end(self, logs=None):
        metrics_filename = f"/projects/University/2021S/PBL/Code/results/holdout-fold_{self.holdout_fold}_{self.metrics.epochs}.metrics"
        self.metrics.serialize(metrics_filename)
        print("Serialized metrics to " + metrics_filename)
        model_filename = f"/projects/University/2021S/PBL/Code/results/holdout-fold_{self.holdout_fold}_{self.metrics.epochs}.model"
