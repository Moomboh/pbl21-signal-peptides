from datetime import datetime
import math
import os
from pathlib import Path

import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras

import metrics
from constants import TRAINING_PARTITIONS, TYPES, KINGDOMS
from utils.Dataset import Dataset
from metrics import MajorityClassBaseline, MetricsBundle
from training.ModelBuilder import ModelBuilder
from utils.encoders import ProteinEncoder, AnnotationEncoder
from utils.helpers import getDatasetPath
from custom_callbacks import PrintHeaderForEachModel, MetricsAfterEpoch, SaveFinalModel, SaveConfusionMatrixAfterEpoch

# Configure parameters
NUM_EPOCHS = 10
METRICS = ["accuracy", "precision", "recall", "mcc"]
BATCH_SIZE = 64

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

step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])

model = model_builder.build(
    keras.losses.CategoricalCrossentropy(),
    tfa.optimizers.AdamW(
        learning_rate=1e-1 * schedule(step),
        weight_decay=lambda: 1e-4 * schedule(step)
    )
)

training_data = data.getFolds(list(TRAINING_PARTITIONS))
test_data = data.getFolds([0])

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
        keras.callbacks.CSVLogger(
            filename=base_folder + f"metrics/final_model.csv",
            separator=',',
            append=False
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=base_folder + "models/final_model_epoch_{epoch}.h5",
            verbose=1,
            save_freq=math.ceil(training_data.shape[0] / BATCH_SIZE)
        ),
        SaveFinalModel(base_folder + f"models/final_model.h5")
    ]
)
