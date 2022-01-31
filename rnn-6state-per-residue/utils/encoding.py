from typing import List, Dict

import numpy as np
from tensorflow import keras

def oneHotToCategorical(array: np.ndarray) -> np.ndarray:
    return np.array([np.argmax(pos) for pos in array])

def categoricalToSequence(array: np.ndarray, reverse_map: Dict[int, str]) -> str:
    return "".join([reverse_map[pos] for pos in array])

def sequenceToCategorical(sequence: str, map: Dict[str, int]) -> np.ndarray:
    return np.array([map[c] for c in sequence])

def categoricalToOneHot(array: np.ndarray, map: Dict[str, int]) -> np.ndarray:
    return keras.utils.to_categorical(array, len(map))
