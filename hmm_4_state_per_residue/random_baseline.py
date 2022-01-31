#!/usr/local/bin/python3
# maxim

import re
import random
import pandas
from SimpleHOHMM import HiddenMarkovModelBuilder, HiddenMarkovModel
from sklearn import metrics
from os.path import exists

import warnings
warnings.filterwarnings("ignore")

##### Settings #####
substitute_no_sp_before_iteration = [False, True]
substitute_no_sp_after_iteration = [False, True]

highest_order_range = range(1, 9)
k_smoothing_range = [0.01]  # [x / 100.0 for x in range(0, 20, 2)]

print("highest_order,k_smoothing,mcc,substitute_no_sp_before,substitute_no_sp_after")
###################


def substitute_states(string_list: [str], pattern: str, substitute: str):
    for index in range(len(string_list)):
        string_list[index] = re.sub(pattern, substitute, string_list[index])


sp_dataframe: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')
sp_dataframe = sp_dataframe[sp_dataframe["partition_no"] != 0]  # leaving partition 0 out
sp_dataframe["annotation"] = sp_dataframe["annotation"].apply(lambda x: x.lower())

# sp_dataframe = sp_dataframe[1:10]

annotations: [str] = ["i", "m", "o", "s", "t", "l", "x"]
amino_acids: [str] = ['M', 'A', 'P', 'T', 'L', 'F', 'Q', 'K', 'S', 'R', 'G', 'D', 'C', 'W', 'E', 'I', 'V', 'Y', 'N', 'H']

frequencies_position_dataframe: pandas.DataFrame

if exists("./../data/frequencies.pkl"):
    frequencies_position_dataframe = pandas.read_pickle("./../data/frequencies.pkl")
else:
    __data: {str: [float]} = {annotation: [0.0] * 70 for annotation in annotations}
    __data.update({amino_acid: [0.0] * 70 for amino_acid in amino_acids})
    __data["n"] = [0] * 70  # used for keeping track of amount of amino acids per position
    frequencies_position_dataframe: pandas.DataFrame = pandas.DataFrame(__data)
    for sequence_index, row in sp_dataframe.iterrows():
        for index, (amino_acid, annotation) in enumerate(zip(row["sequence"], row["annotation"])):
            frequencies_position_dataframe.loc[index, annotation] += 1
            frequencies_position_dataframe.loc[index, amino_acid] += 1
            if annotation in ["i", "m", "o"]:
                frequencies_position_dataframe.loc[index, "x"] += 1
            frequencies_position_dataframe.loc[index, "n"] += 1
    frequencies_position_dataframe.to_pickle("./../data/frequencies.pkl")
    frequencies_position_dataframe.to_string("./../data/frequencies.txt")
    print(frequencies_position_dataframe)

    # for position in range(len(frequencies_position_dataframe)):
    #     if not frequencies_position_dataframe.at[position, "n"] == 0:  # otherwise we would get nan values (and who needs those rows anyway?) or skip them?
    #         for annotation, amino_acid in zip(annotations, amino_acids):
    #             frequencies_position_dataframe.loc[position, annotation] /= frequencies_position_dataframe.at[position, "n"]
    #             frequencies_position_dataframe.loc[position, amino_acid] /= frequencies_position_dataframe.at[position, "n"]

# Dataframes should look like this:  (with amino_acids and annotations)
#            S         T         L         I     ...
# pos
# 0    0.183642  0.155864  0.094136  0.066358    ...
# 1    0.290123  0.277778  0.111111  0.098765    ...
# 2    0.276235  0.316358  0.168210  0.208333    ...
# ...       ...       ...       ...       ...    ...


for substitute_no_sp_before in substitute_no_sp_before_iteration:

    if substitute_no_sp_before:
        sp_dataframe["annotation"] = sp_dataframe["annotation"].apply(lambda annotation: re.sub(r"[imo]", "x", annotation))
        annotations = ["x", "s", "t", "l"]  # removing non sp_types

    builder: HiddenMarkovModelBuilder = HiddenMarkovModelBuilder()
    builder.add_batch_training_examples([list(sequence) for (_, sequence) in sp_dataframe["sequence"].iteritems()],
                                        [list(sequence) for (_, sequence) in sp_dataframe["annotation"].iteritems()])  # so nice this library does the encoding :)

    for substitute_no_sp_after in substitute_no_sp_after_iteration:
        for highest_order in highest_order_range:
            for k_smoothing in k_smoothing_range:
                model: HiddenMarkovModel = builder.build(k_smoothing=k_smoothing, highest_order=highest_order)

                predicted_flatten: [str] = []
                annotations_flatten: [str] = []

                # generate a sequence and annotations randomly according to weight
                # simulate a lot of predictions and calculate MCC to get an average random baseline
                for i in range(2000):
                    sequence_random: [str] = []
                    annotations_random: [str] = []
                    for pos in range(70):
                        sequence_random.append(random.choices(amino_acids, weights=[frequencies_position_dataframe.at[pos, amino_acid] for amino_acid in amino_acids], k=1)[0])
                        annotations_random.append(random.choices(annotations, weights=[frequencies_position_dataframe.at[pos, annotation] for annotation in annotations], k=1)[0])

                    predicted: [str] = model.decode(sequence_random)

                    if substitute_no_sp_after and not substitute_no_sp_before:
                        substitute_states(predicted, r"[imo]", "x")
                        substitute_states(annotations_random, r"[imo]", "x")

                    predicted_flatten.extend(predicted)
                    annotations_flatten.extend(annotations_random)

                print(highest_order, k_smoothing, metrics.matthews_corrcoef(annotations_flatten, predicted_flatten), substitute_no_sp_before, substitute_no_sp_after, sep=",")
