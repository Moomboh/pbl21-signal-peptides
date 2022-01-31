#!/usr/local/bin/python3
# maxim

import sys
import re
import pandas
import difflib
import uuid
from sklearn import metrics
from SimpleHOHMM import HiddenMarkovModelBuilder, HiddenMarkovModel
# pip install git+https://github.com/jacobkrantz/Simple-HOHMM.git


#### Variables and Settings ####

debug = False

dataframe_raw: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')

substitution_non_sp_before = False
substitution_non_sp_after = True

reverse_data = False

iterate_highest_order = False
highest_order_range = range(1, 9)
default_highest_order = 5  # 5 is the best after evaluation

iterate_k_smoothing = False
k_smoothing_range = [x / 100.0 for x in range(0, 20, 2)]
default_k_smoothing = 0.01

# printing settings
print("substitution_non_sp_before=", substitution_non_sp_before)
print("substitution_non_sp_after=", substitution_non_sp_after)
print("reverse_data=", reverse_data)
print("iterate_highest_order=", iterate_highest_order)
print("iterate_k_smoothing=", iterate_k_smoothing)
print()

# Header
print("highest_order,k_smoothing,mcc,test_partition,results_path")

################################


#### Helper functions ####
def substitute_states(states_list: [[str]], pattern: str, substitute: str):
    for states in states_list:
        if debug:
            print("before: ", "".join(states))
        for index in range(len(states)):
            states[index] = re.sub(pattern, substitute, states[index])
        if debug:
            print("after:  ", "".join(states))
##########################


# changing all non SP-Types to a common Type X
# performance got worse after substitution: MCC: 0.65 -> 0.47
if substitution_non_sp_before:
    dataframe_raw["annotation"] = dataframe_raw["annotation"].apply(lambda annotation: re.sub(r"[IMO]", "X", annotation))

if reverse_data:
    dataframe_raw["sequence"] = dataframe_raw["sequence"].apply(lambda sequence: sequence[::-1])
    dataframe_raw["annotation"] = dataframe_raw["annotation"].apply(lambda annotation: annotation[::-1])

for k_fold in range(1, 5):
    train_partition = [1, 2, 3, 4]
    train_partition.remove(k_fold)

    train_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"].isin(train_partition)]
    test_dataframe: pandas.DataFrame = dataframe_raw[dataframe_raw["partition_no"] == k_fold]

    observations_train_list: [[str]] = [list(sequence) for (_, sequence) in train_dataframe["sequence"].iteritems()]
    states_train_list: [[str]] = [list(sequence) for (_, sequence) in train_dataframe["annotation"].iteritems()]

    observations_test_list: [[str]] = [list(sequence) for (_, sequence) in test_dataframe["sequence"].iteritems()]
    states_test_list: [[str]] = [list(sequence) for (_, sequence) in test_dataframe["annotation"].iteritems()]

    builder: HiddenMarkovModelBuilder = HiddenMarkovModelBuilder()
    builder.add_batch_training_examples(observations_train_list, states_train_list)  # so nice this library does the encoding :)

    if not iterate_highest_order:
        highest_order_range = [default_highest_order]
    if not iterate_k_smoothing:
        k_smoothing_range = [default_k_smoothing]

    for highest_order in highest_order_range:
        for k_smoothing in k_smoothing_range:

            model: HiddenMarkovModel = builder.build(k_smoothing=k_smoothing, highest_order=highest_order)

            states_predicted_list: [[str]] = [model.decode(observations_test) for observations_test in observations_test_list]

            if substitution_non_sp_after and not substitution_non_sp_before:
                # reducing from 6 state prediction to 4 by substituting I,M,O to X
                substitute_states(states_predicted_list, r"[IMO]", "x")
                # also doing this for the test set in order to compare
                substitute_states(states_test_list, r"[IMO]", "x")

            states_predicted_flattened: [str] = sum(states_predicted_list, [])
            states_test_flattened: [str] = sum(states_test_list, [])

            filepath = "saved_output_and_evaluation/" + str(uuid.uuid4()) + ".txt"

            print(metrics.precision_score(y_true=states_test_flattened, y_pred=states_predicted_flattened, average='macro'))
            print(highest_order, k_smoothing, metrics.matthews_corrcoef(states_test_flattened, states_predicted_flattened), k_fold, filepath, sep=",")

            # saving all output to a file
            with open(filepath, "w") as file:
                commandline_stdin = sys.stdout
                sys.stdout = file

                print("highest_order=", highest_order)
                print("k_smoothing=", k_smoothing)
                print("MCC: ", metrics.matthews_corrcoef(states_test_flattened, states_predicted_flattened))
                print("Training partitions: ", train_partition)

                parameters = model.get_parameters()

                print("Starting probabilities (pi):")
                for element in parameters.get("pi"):
                    print(element)

                print("Transition probabilities (A):")
                for element in parameters.get("A"):
                    print(element)

                print("Emission probabilities (B):")
                for element in parameters.get("B"):
                    print(element)

                print("Observations: ", end="")
                for element in parameters.get("all_obs"):
                    print(element, end="")

                print("\nAll States: ", end="")
                for element in parameters.get("all_states"):
                    print(element, end="")

                print("\nSingle: ", end="")
                for element in parameters.get("single_states"):
                    print(element, end="")

                print()
                for states_predicted, states_test in zip(states_predicted_list, states_test_list):
                    # print("MCC: " + str(metrics.matthews_corrcoef(target_state, predicted_state)))
                    states_predicted_string: str = "".join(states_predicted)
                    states_test_string: str = "".join(states_test)
                    sequence_matcher: difflib.SequenceMatcher = difflib.SequenceMatcher(None, states_predicted_string,
                                                                                        states_test_string)
                    print("Predicted: " + states_predicted_string)
                    print("Target:    " + states_test_string)
                    print(difflib.ndiff(states_predicted, states_test))
                    print("Ratio: " + str(sequence_matcher.ratio()))
                    print("-" * 50)

                print("#" * 50)
                print("#" * 50)
                sys.stdout.close()
                sys.stdout = commandline_stdin
