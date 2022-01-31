from typing import Tuple
from typing_extensions import Literal

import numpy as np
from tensorflow import keras

from constants import LOCATIONS, PATHWAY_ANNOTATIONS, amino_acid_mapping, annotation_mapping, AMINO_ACID_CODES

class SignalPeptide:
    def __init__(self, metadata, sequence, annotation):
        [self.uniprot_ac, self.kingdom, self.type, partition] = metadata.split('|')
        self.partition = int(partition)
        self.sequence = sequence
        self.annotation = annotation

    @classmethod
    def fromLine(cls, lines: Tuple[str, str, str]):
        metadata = lines[0].strip().replace('>', '')
        sequence = lines[1].strip()
        annotation = lines[2].strip()
        return SignalPeptide(metadata, sequence, annotation)

    def getLocation(self) -> str:
        locations = set(self.annotation).intersection(LOCATIONS.keys())
        if len(locations) == 3:
            return "TRANSMEMBRANE"
        elif len(locations) == 1:
            return LOCATIONS[next(iter(locations))].upper()
        else:
            if 'I' in locations:
                return "CYTOPLASM+MEMBRANE"
            else:
                return "EXTRACELLULAR+MEMBRANE"

    def getSignalPeptidePosition(self) -> Tuple[int, int]:
        if self.type == "NO_SP":
            return (-1, -1)
        else:
            a = PATHWAY_ANNOTATIONS[self.type]
            start = self.annotation.index(a)
            end = len(self.annotation) - self.annotation[::-1].index(a) - 1
            return (start, end)

    def toDict(self):
        return {
            "uniprot_ac": self.uniprot_ac,
            "partition_no": self.partition,
            "type": self.type,
            "pathway": TYPES[self.type],
            "kingdom": self.kingdom,
            "sequence": self.sequence,
            "annotation": self.annotation,
            "location": self.getLocation(),
            "sp_start": self.getSignalPeptidePosition()[0],
            "sp_end": self.getSignalPeptidePosition()[1]
        }

    @staticmethod
    def _pad(sequence: str) -> np.array:
        categorized = [amino_acid_mapping[c] for c in sequence]
        if len(sequence) < 70:
            padding = [0] * (70 - len(sequence))
            categorized = [*padding, *categorized]
        return keras.utils.to_categorical(categorized, len(AMINO_ACID_CODES) + 1)

    def padSequence(self) -> np.array:
        return self._pad(self.sequence)

    def padAnnotation(self) -> np.array:
        return self._pad(self.annotation)

    """
    def encode(self, type: Literal["sequence", "annotation"]) -> np.array:
        if type == "sequence":
            return self.encodeSequence()
        else:
            return self.encodeAnnotation()

    def encodeSequence(self) -> np.array:
        categorized = [amino_acid_mapping[c] for c in self.sequence]
        return keras.utils.to_categorical(categorized, len(AMINO_ACID_CODES))

    def encodeAnnotation(self) -> np.array:
        categorized = [annotation_mapping[a] for a in self.annotation]
        return keras.utils.to_categorical(categorized, len(annotation_mapping))
    """

    def __repr__(self) -> str:
        return f"SP (len {len(self.sequence)})"

    def __len__(self) -> int:
        return len(self.sequence)
