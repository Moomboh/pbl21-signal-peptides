import numpy as np

class OneHotEncoder():
    def __init__(self, categories):
        self.categories = categories

        dummies = np.eye(len(categories))
        self.encoding = {cat: dummies[i].tolist() for i, cat in enumerate(categories)}

    def transform(self, x):
        one_hot = np.ndarray(
            shape=(len(x), len(self.categories)),
            dtype=float
        )

        for i, v in enumerate(x):
            one_hot[i] = self.encoding[v]

        return one_hot