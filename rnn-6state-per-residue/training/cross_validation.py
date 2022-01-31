from datetime import datetime
import math
import os
from pathlib import Path

import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras

from constants import TRAINING_PARTITIONS, TYPES, KINGDOMS
from utils.Dataset import Dataset
from metrics import MajorityClassBaseline, MetricsBundle
from training.ModelBuilder import ModelBuilder
from utils.encoders import ProteinEncoder, AnnotationEncoder
from utils.helpers import getDatasetPath
from custom_callbacks import PrintHeaderForEachModel, MetricsAfterEpoch, SaveFinalModel

# Configure parameters
NUM_EPOCHS = 200
METRICS = ["accuracy", "precision", "recall", "mcc"]
BATCH_SIZE = 64

# Configure CPU usage
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

data = Dataset(getDatasetPath())

# Set up utilities
model_builder = ModelBuilder()

protein_encoder = ProteinEncoder()
annotation_encoder = AnnotationEncoder()

# Create folders for results and metrics
base_folder = f"results/{datetime.now().strftime('%Y%m%d-%H%M')}/"
Path(base_folder).mkdir()
Path(base_folder + "metrics").mkdir()
Path(base_folder + "models").mkdir()

for holdout_fold in TRAINING_PARTITIONS:
    model = model_builder.build(
        keras.losses.CategoricalCrossentropy(),
        keras.optimizers.Adam(
            learning_rate=0.0001,
            clipvalue=0.25)
    )

    training_data = data.getFolds(list(TRAINING_PARTITIONS.difference({holdout_fold})))
    test_data = data.getFolds([holdout_fold])

    history = model.fit(
        protein_encoder.encodeMultiple(list(training_data["sequence"])),
        annotation_encoder.encodeMultiple(list(training_data["annotation"])),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=0,
        validation_data=(
            protein_encoder.encodeMultiple(list(test_data["sequence"])),
            annotation_encoder.encodeMultiple(list(test_data["annotation"]))
        ),
        callbacks=[
            tfa.callbacks.TQDMProgressBar(
                leave_epoch_progress=False
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                verbose=1,
                min_delta=0.005,
                cooldown=0,
                min_lr=1e-9
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00001,
                patience=50,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                filename=base_folder + f"metrics/holdout-fold_{holdout_fold}.csv",
                separator=',',
                append=False
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=base_folder + f"models/holdout-fold_{holdout_fold}_epoch_" + "{epoch}.h5",
                verbose=1,
                save_freq=math.ceil(training_data.shape[0] / BATCH_SIZE) * 25
            ),
            PrintHeaderForEachModel(holdout_fold),
            SaveFinalModel(base_folder + f"models/holdout-fold_{holdout_fold}_final.h5")
        ]
    )
