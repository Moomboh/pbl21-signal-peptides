from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from tensorflow import keras

from constants import amino_acid_mapping, annotation_mapping

# TODO convert into functions and merge file with decoders

class Encoder(ABC):
    @abstractmethod
    def __init__(self):
        """
        Set map for conversion
        """
        self._map = None

    def __call__(self, sequence: str) -> np.ndarray:
        return keras.utils.to_categorical(
            [self._map[char] for char in sequence],
            len(self._map)
        )

    def encodeMultiple(self, sequences: List[str]) -> np.ndarray:
        return np.array([self(seq) for seq in sequences])


class ProteinEncoder(Encoder):
    def __init__(self):
        self._map = amino_acid_mapping


class AnnotationEncoder(Encoder):
    def __init__(self):
        self._map = annotation_mapping


