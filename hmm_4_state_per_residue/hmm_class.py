# maxim

import sys
import hashlib

sys.path.append('../../consensus_model')
from consensus_model.model_interface import Model, ModelBuilder

from os.path import exists
import pickle
from SimpleHOHMM import HiddenMarkovModelBuilder, HiddenMarkovModel
# pip install git+https://github.com/jacobkrantz/Simple-HOHMM.git


_substitution_dict = {
 'I': 'X',
 'M': 'X',
 'O': 'X',
 'S': 'S',
 'L': 'L',
 'T': 'T',
 'X': 'X'
}


class HMMBuilder(ModelBuilder):

    def build(self, data_train: ([[str]], [[str]]), data_test: ([[str]], [[str]])) -> Model:
        return self.build_extended(data_train, data_test)

    def build_extended(self, data_train: ([[str]], [[str]]), data_test: ([[str]], [[str]]), k_smoothing: float = 0.01,
                       highest_order: int = 5, substitute_non_sp_before: bool = False, substitute_non_sp_after: bool = True) -> Model:
        model_hash = HMMBuilder.__hash(data_train, data_test, k_smoothing, highest_order, substitute_non_sp_before, substitute_non_sp_after)
        if HMMBuilder.__exists(model_hash):
            return HMMBuilder.load(model_hash)
        else:
            model: Model = HMM(data_train, data_test, k_smoothing, highest_order, substitute_non_sp_before, substitute_non_sp_after)
            HMMBuilder.save(model, model_hash)
            # print("Model saved under the hash: ", model_hash)
            return model

    def build_hmm(self, data_train: ([[str]], [[str]]), data_test: ([[str]], [[str]]), k_smoothing: float = 0.01,
                       highest_order: int = 5, substitute_non_sp_before: bool = False, substitute_non_sp_after: bool = True):
        model_hash = HMMBuilder.__hash(data_train, data_test, k_smoothing, highest_order, substitute_non_sp_before, substitute_non_sp_after)
        if HMMBuilder.__exists(model_hash):
            return HMMBuilder.load(model_hash)
        else:
            model: HMM = HMM(data_train, data_test, k_smoothing, highest_order, substitute_non_sp_before, substitute_non_sp_after)
            HMMBuilder.save(model, model_hash)
            # print("Model saved under the hash: ", model_hash)
            return model

    @staticmethod
    def __get_path(name: str):
        return "saved_models/" + name + "obj"

    @staticmethod
    def save(model: Model, name: str):
        with open(HMMBuilder.__get_path(name), 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load(name: str) -> Model:  # cant set return type to HMM (Must retype with 'as HMM')
        with open(HMMBuilder.__get_path(name), 'rb') as file:
            model: HMM = pickle.load(file)
        return model

    @staticmethod
    def __exists(name: str) -> bool:
        return exists(HMMBuilder.__get_path(name))

    @staticmethod  # needed in order to identify if a given model has been trained yet (concatenated with separator in order to get better hashes)
    def __hash(data_train: ([[str]], [[str]]), data_test: ([[str]], [[str]]), k_smoothing: float, highest_order: int,
               substitute_non_sp_before: bool, substitute_non_sp_after: bool) -> str:
        return str(hashlib.sha256((str(data_train) + "|" + str(data_test) + "|" + str(k_smoothing) + "|"
                                   + str(highest_order) + "|" + str(substitute_non_sp_before) + "|"
                                   + str(substitute_non_sp_after)).encode('utf-8')).hexdigest())


class HMM(Model):

    def __init__(self, data_train: ([[str]], [[str]]), data_test: ([[str]], [[str]]), k_smoothing: float = 0.01,
                 highest_order: int = 5, substitute_non_sp_before: bool = False, substitute_non_sp_after: bool = True):
        self.__k_smoothing: float = k_smoothing
        self.__highest_order: int = highest_order
        self.__substitute_non_sp_before: bool = substitute_non_sp_before
        self.__substitute_non_sp_after: bool = substitute_non_sp_after
        self.__observations_train_list, self.__states_train_list = data_train
        self.__observations_test_list, self.__states_test_list = data_test

        if substitute_non_sp_before:
            self.__states_train_list = [HMM.__4_states_substitution(states) for states in self.__states_train_list]

        if substitute_non_sp_before or substitute_non_sp_after:
            self.__states_test_list = [HMM.__4_states_substitution(states) for states in self.__states_test_list]

        builder: HiddenMarkovModelBuilder = HiddenMarkovModelBuilder()
        builder.add_batch_training_examples(self.__observations_train_list, self.__states_train_list)

        self.__model: HiddenMarkovModel = builder.build(k_smoothing=k_smoothing, highest_order=highest_order)

    def get_train_data(self) -> ([[str]], [[str]]):
        return self.__observations_train_list, self.__states_train_list

    def get_test_data(self) -> ([[str]], [[str]]):
        return self.__observations_test_list, self.__states_test_list

    def annotate(self, observation: [str]) -> ([str], float):
        return HMM.__4_states_substitution(self.__model.decode(observation)) \
                   if self.__substitute_non_sp_after or self.__substitute_non_sp_before \
                   else self.__model.decode(observation), \
               1.0  # TODO fix after analysis

    @staticmethod
    def __4_states_substitution(states: [str]) -> [str]:
        return [_substitution_dict[state] for state in states]

    def get_model(self) -> HiddenMarkovModel:
        return self.__model
