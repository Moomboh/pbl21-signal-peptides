from tensorflow import keras
import tensorflow_addons as tfa

from metrics import MCC

class ModelBuilder:
  def build(self, loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.CategoricalAccuracy()]) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.Input(shape=(70,20)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(125, return_sequences=True, input_shape=(70, 20))))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(75, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(6, activation="softmax"))

    model.build()

    model.compile(
      loss=loss,
      optimizer=optimizer,
      metrics=metrics
    )

    return model
