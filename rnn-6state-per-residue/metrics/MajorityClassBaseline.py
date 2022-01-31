from typing import List

import numpy as np
import pandas as pd

from .baseclasses import MultiMetric, Serializable, DataframeOutput
from constants import KINGDOMS, TYPES, METRIC_KINGDOMS
from utils.SignalPeptide import SignalPeptide

class MajorityClassBaseline(MultiMetric, Serializable, DataframeOutput):

    def __init__(self, peptides: List[SignalPeptide], metrics: List[str]):
        super().__init__(metrics)
        self._peptides = peptides
        self._initialize()
        self._compute()

    def _initialize(self) -> None:
        self._values = {
            metric: {
                type: {
                    kingdom: None for kingdom in METRIC_KINGDOMS
                } for type in TYPES
            } for metric in self._metrics.keys()
        }

    def _compute(self) -> None:
        for type in TYPES:
            self._values[type] = {}
            for kingdom in [*KINGDOMS, "overall"]:
                if kingdom == "overall":
                    relevant_peptides = [peptide for peptide in self._peptides
                                         if peptide.type == type and len(peptide) == 70]
                else:
                    relevant_peptides = [peptide for peptide in self._peptides if
                                         peptide.kingdom == kingdom and peptide.type == type
                                         and len(peptide) == 70]

                if len(relevant_peptides) == 0:
                    for name, metric in self._metrics.items():
                        self._values[name][type][kingdom] = np.NaN

                else:
                    decoded_annotations = np.array(
                        [np.array([np.argmax(pos) + 1 for pos in p.encodeAnnotation()]) for p in relevant_peptides]
                    )
                    majority_class = np.argmax(np.sum(
                        np.concatenate([p.encodeAnnotation() for p in relevant_peptides]),
                        axis=0
                    ))
                    decoded_predictions = np.array([
                      np.repeat(majority_class + 1, 70)
                      for _ in range(len(relevant_peptides))
                    ])

                    for name, metric in self._metrics.items():
                        self._values[name][type][kingdom] = metric.compute(decoded_annotations, decoded_predictions)

    def toDataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                (metric, type, kingdom, self._values[metric][type][kingdom])
                for metric in self._metrics.keys()
                for type in TYPES
                for kingdom in [*KINGDOMS, "overall"]
            ]
        )

        df.columns = ["metric", "type", "kingdom", "value"]
        df.set_index(["metric", "type", "kingdom"], inplace=True)
        return df
