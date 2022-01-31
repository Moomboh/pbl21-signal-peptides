# maxim

import abc


# This interface needs to be implemented by the models.
class Model(abc.ABC):

    # Gives a prediction for the signal peptides based on the amino_acid_string given.
    # return a tuple with a prediction and a confidence score for how accurate this prediction might be.
    @abc.abstractmethod
    def annotate(self, observation: [str]) -> ([str], float):
        pass

    @abc.abstractmethod
    def get_train_data(self) -> ([[str]], [[str]]):
        pass

    @abc.abstractmethod
    def get_test_data(self) -> ([[str]], [[str]]):
        pass


# This interface needs to be implemented by the classes that builds models.
class ModelBuilder(abc.ABC):

    # Builds the model with default values. Both train and test data is provided in case substitution is necessary.
    @abc.abstractmethod
    def build(self, data_train: ([[str]], [[str]]), data_test: ([[str]], [[str]])) -> Model:
        pass
