#!/usr/local/bin/python3
# maxim
import pandas
import pycm

frequencies_dataframe: pandas.DataFrame = pandas.read_pickle("./../data/frequencies.pkl")


columns: [str] = ["i", "m", "o", "s", "t", "l", "n"]
# columns = [column for column in columns if column not in ["i", "m", "o"]].append("x")

__data: {str: float} = {column: frequencies_dataframe[column].sum() for column in columns}
columns.remove("n")
rel_frequencies_dataframe = pandas.DataFrame({column: [__data[column] / __data["n"]] for column in columns})

print(rel_frequencies_dataframe)

cm = (rel_frequencies_dataframe.transpose().dot(rel_frequencies_dataframe) * 10000000).round().astype(int).to_dict()
cm = pycm.ConfusionMatrix(matrix=cm)

print(cm.stat(overall_param=['Overall MCC', 'ACC Macro'], class_param=['MCC', 'ACC']))