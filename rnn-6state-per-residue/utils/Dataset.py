from typing import Dict, List, Tuple

import numpy as np
import pandas
import pandas as pd

from .SignalPeptide import SignalPeptide
from constants import KINGDOMS, TYPES, TRAINING_PARTITIONS, ALL_PARTITIONS

class Dataset:

    def __init__(self, filepath: str, keep_shorter_sequences: bool=False):

        # Read FASTA file
        # TODO filepath validation (unnecessary for this project)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            self._all_peptides = [SignalPeptide.fromLine(tuple(lines[i:i+3])) for i in range(0, len(lines)-2, 3)]

        # Filter out shorter sequences
        if not keep_shorter_sequences:
            self._all_peptides = [p for p in self._all_peptides if len(p) == 70]

        # Generate dataframe
        self._values = pd.DataFrame(
            [
                (partition, type, kingdom, i, p.sequence, p.annotation)
                for partition in ALL_PARTITIONS
                for type in TYPES
                for kingdom in KINGDOMS
                for i, p in enumerate([p for p in self._all_peptides
                                       if p.partition == partition
                                       and p.kingdom == kingdom
                                       and p.type == type
            ])
        ]
        )

        self._values.columns = ["partition", "type", "kingdom", "number", "sequence", "annotation"]
        self._values.set_index(["partition", "type", "kingdom", "number"], inplace=True)

    def getFolds(self, folds: List[int]) -> pandas.DataFrame:
        return self._values.query("partition == @folds").copy(deep=True)
