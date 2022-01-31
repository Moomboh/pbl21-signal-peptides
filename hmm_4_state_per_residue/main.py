#!/usr/local/bin/python3
# maxim

import pandas
import uuid

from hmm_class import HMMBuilder
from analysis import Analysis

import sys
sys.path.append('../../consensus_model')
from consensus_model.model_interface import Model
from hmm_class import HMM


# substitute_non_sp_before = False
# substitute_non_sp_after = True

iterate_highest_order = True
highest_order_range = range(1, 7)
default_highest_order = 5  # 5 is the best after evaluation

iterate_k_smoothing = False
k_smoothing_range = [x / 100.0 for x in range(0, 12, 2)]
default_k_smoothing = 0.01

builder: HMMBuilder = HMMBuilder()


# def main():

    # print(Analysis.get_header(), "K_Fold", "HighestOrder", "K_Smoothing", "Substitution_non_SP_before", "Substitution_non_SP_after", sep=',')
    #
    # for substitute_non_sp_before, substitute_non_sp_after in [(True, False), (False, True), (False, False)]:
    #     for highest_order in highest_order_range if iterate_highest_order else [default_highest_order]:
    #         for k_smoothing in k_smoothing_range if iterate_k_smoothing else [default_k_smoothing]:
    #
    #             # Training model with different folds
    #             for k_fold in range(1, 5):
    #                 train_partition = [1, 2, 3, 4]
    #                 train_partition.remove(k_fold)
    #
    #                 dataframe_raw: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')
    #
    #                 train_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"].isin(train_partition)]
    #                 test_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"] == k_fold]
    #
    #                 data_train: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in train_dataframe["sequence"].iteritems()], \
    #                                                  [list(sequence) for (_, sequence) in train_dataframe["annotation"].iteritems()]
    #                 data_test: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in test_dataframe["sequence"].iteritems()], \
    #                                                 [list(sequence) for (_, sequence) in test_dataframe["annotation"].iteritems()]
    #
    #                 # Training model with different values
    #                 model: Model = builder.build_extended(data_train=data_train, data_test=data_test, k_smoothing=k_smoothing,
    #                                                       highest_order=highest_order, substitute_non_sp_before=substitute_non_sp_before,
    #                                                       substitute_non_sp_after=substitute_non_sp_after)
    #
    #                 analysis: Analysis = Analysis(model=model)
    #
    #                 print(analysis, k_fold, highest_order, k_smoothing, substitute_non_sp_before, substitute_non_sp_after, sep=',')
    #
    #         sys.stdout.flush()

    # print(Analysis.get_header(), "HighestOrder", "K_Smoothing", "Substitution_non_SP_before", "Substitution_non_SP_after", sep=',')

    # for substitute_non_sp_before, substitute_non_sp_after in [(True, False), (False, True), (False, False)]:
    #     for highest_order in highest_order_range if iterate_highest_order else [default_highest_order]:
    #         for k_smoothing in k_smoothing_range if iterate_k_smoothing else [default_k_smoothing]:
    #
    #             dataframe_raw: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')
    #
    #             train_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"].isin([1, 2, 3, 4])]
    #             test_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"] == 0]
    #
    #             data_train: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in train_dataframe["sequence"].iteritems()], \
    #                                              [list(sequence) for (_, sequence) in train_dataframe["annotation"].iteritems()]
    #             data_test: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in test_dataframe["sequence"].iteritems()], \
    #                                             [list(sequence) for (_, sequence) in test_dataframe["annotation"].iteritems()]
    #
    #             # Training model with different values
    #             model: Model = builder.build_extended(data_train=data_train, data_test=data_test, k_smoothing=k_smoothing,
    #                                                   highest_order=highest_order, substitute_non_sp_before=substitute_non_sp_before,
    #                                                   substitute_non_sp_after=substitute_non_sp_after)
    #
    #             analysis: Analysis = Analysis(model=model)
    #
    #             print(analysis, highest_order, k_smoothing, substitute_non_sp_before, substitute_non_sp_after, sep=',')
    #
    #         sys.stdout.flush()

# if __name__ == '__main__':
#     filepath = "saved_output_and_evaluation/full_analysis/" + str(uuid.uuid4()) + ".csv"
#
#     with open(filepath, "w") as file:
#         commandline_stdout = sys.stdout
#         sys.stdout = file
#
#         main()
#
#         sys.stdout.flush()
#         sys.stdout.close()
#         sys.stdout = commandline_stdout
#
#     dataframe_raw: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')
#
#     train_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"].isin([1, 2, 3, 4])]
#     test_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"] == 0]
#
#     data_train: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in train_dataframe["sequence"].iteritems()], \
#                                      [list(sequence) for (_, sequence) in train_dataframe["annotation"].iteritems()]
#     data_test: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in test_dataframe["sequence"].iteritems()], \
#                                     [list(sequence) for (_, sequence) in test_dataframe["annotation"].iteritems()]
#
#     # Best Model
#     model: HMM = builder.build_hmm(data_train=data_train, data_test=data_test, k_smoothing=0.01,
#                                    highest_order=3, substitute_non_sp_before=False, substitute_non_sp_after=True)
#     analysis: Analysis = Analysis(model=model)
#     print(analysis.confusion_matrix_types())
#
#     print("finished")

# def main():
#     dataframe_raw: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')
#
#     train_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"].isin([1, 2, 3, 4])]
#     test_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"] == 0]
#
#     data_train: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in train_dataframe["sequence"].iteritems()], \
#                                      [list(sequence) for (_, sequence) in train_dataframe["annotation"].iteritems()]
#     data_test: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in test_dataframe["sequence"].iteritems()], \
#                                     [list(sequence) for (_, sequence) in test_dataframe["annotation"].iteritems()]
#
#     # Best Model
#     model: HMM = builder.build_hmm(data_train=data_train, data_test=data_test, k_smoothing=0.01,
#                                    highest_order=3, substitute_non_sp_before=False, substitute_non_sp_after=False)
#
#     model.get_model().display_parameters()
#     print("sequence;annotation;annotation_pred")
#
#     data_test_observation, data_test_states = data_test
#     data_prediction_scored: [([str], float)] = [model.annotate(observation) for observation in data_test_observation]
#
#     for test_observation, test_states, test_prediction_scored in zip(data_test_observation, data_test_states, data_prediction_scored):
#         test_prediction, _ = test_prediction_scored
#         print("".join(test_observation), "".join(test_states), "".join(test_prediction), sep=";")

def main():

    print(Analysis.get_header(), "K_Fold", "HighestOrder", "K_Smoothing", "Substitution_non_SP_before", "Substitution_non_SP_after", sep=',')

    for highest_order in highest_order_range if iterate_highest_order else [default_highest_order]:

        # Training model with different folds
        for k_fold in range(1, 5):
            train_partition = [1, 2, 3, 4]
            train_partition.remove(k_fold)

            dataframe_raw: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')

            train_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"].isin(train_partition)]
            test_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"] == k_fold]

            data_train: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in train_dataframe["sequence"].iteritems()], \
                                             [list(sequence) for (_, sequence) in train_dataframe["annotation"].iteritems()]
            data_test: ([[str]], [[str]]) = [list(sequence) for (_, sequence) in test_dataframe["sequence"].iteritems()], \
                                            [list(sequence) for (_, sequence) in test_dataframe["annotation"].iteritems()]

            # Training model with different values
            model: Model = builder.build_extended(data_train=data_train, data_test=data_test, k_smoothing=0.01,
                                                  highest_order=highest_order, substitute_non_sp_before=False,
                                                  substitute_non_sp_after=True)

            analysis: Analysis = Analysis(model=model)

            print(analysis, k_fold, highest_order, 0.01, False, True, sep=',')
            sys.stdout.flush()


if __name__ == '__main__':
    filepath = "saved_output_and_evaluation/full_analysis/" + str(uuid.uuid4()) + ".csv"
    print(filepath)

    with open(filepath, "w") as file:
        commandline_stdout = sys.stdout
        sys.stdout = file

        main()

        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = commandline_stdout

    print("finished")
