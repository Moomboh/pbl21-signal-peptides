from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def train_batch(self, X, y, context):
        pass

    @abstractmethod
    def validate_batch(self, X, y, context):
        pass

    @abstractmethod
    def before_metrics(self, pred, y, context):
        pass

    @staticmethod
    @abstractmethod
    def class_labels():
        pass

    @staticmethod
    @abstractmethod
    def context_labels():
        pass

    @abstractmethod
    def transform_input(self, x):
        pass

    @abstractmethod
    def transform_target(self, y):
        pass

    @abstractmethod
    def transform_context(self, c):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    @classmethod
    @abstractmethod
    def get_background(cls, dataset):
        pass
