# why don't we use the same OneHotEncoder directly from utils?

from __future__ import annotations

import numpy as np


class OneHotEncoder:

    def __init__(self, categories_amount: int, encoding: [[float]],  index_encoding: {str: int}):
        self.__categories_amount: int = categories_amount
        self.__encoding: [[float]] = encoding
        self.__index_encoding: {str: int} = index_encoding

#    def get_index_encoding(self) -> {str: int}: return self.__index_encoding
    def get_categories_amount(self) -> int: return self.__categories_amount
    def get_corresponding_index(self, category: str) -> int: return self.__index_encoding[category]
    def __encode(self, category: str) -> [float]: return self.__encoding[self.get_corresponding_index(category)]

    @staticmethod
    def from_categories(categories: [str]) -> OneHotEncoder:
        categories_amount: int = len(categories)
        return OneHotEncoder(categories_amount,
                             [dummy.tolist() for dummy in np.eye(categories_amount)],
                             {category: index for index, category in enumerate(categories)})

    @staticmethod  # using to transform [a, b, c, c, e,] -([a,b], [c,d], [e])-> [[1,0,0][1,0,0][0,1,0][0,1,0][0,0,1]]
    def from_multiple_categories(*categories_partitions: [[str]]) -> OneHotEncoder:
        categories_amount: int = len(categories_partitions)

        index_encoding: {str: int} = {}
        for partition_index, partition in enumerate(categories_partitions):
            for element in partition:
                index_encoding[element] = partition_index

        return OneHotEncoder(categories_amount,
                             [dummy.tolist() for dummy in np.eye(categories_amount)],
                             index_encoding)

    def transform(self, x: str):
        one_hot: np.ndarray = np.ndarray(shape=(len(x), self.__categories_amount), dtype=float)

        for index, element in enumerate(x):
            one_hot[index] = self.__encode(element)

        return one_hot
