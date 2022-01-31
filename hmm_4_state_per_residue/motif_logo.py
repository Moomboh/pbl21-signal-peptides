#!/usr/local/bin/python3
# maxim

import pandas
import uuid
import matplotlib.pyplot as plt
import logomaker
# https://academic.oup.com/bioinformatics/article/36/7/2272/5671693
# https://github.com/jbkinney/logomaker

plt.ion()

debug = False
rel_frequencies = False

sp_dataframe: pandas.DataFrame = pandas.read_csv("./../data/dataset.tsv", sep='\t')
sp_dataframe = sp_dataframe[sp_dataframe["partition_no"] != 0]  # leaving partition 0 out

annotations: [str] = ["I", "M", "O", "S", "T", "L"]
amino_acids: [str] = ['M', 'A', 'P', 'T', 'L', 'F', 'Q', 'K', 'S', 'R', 'G', 'D', 'C', 'W', 'E', 'I', 'V', 'Y', 'N', 'H']

# needs a pandas table in the form like:
#            A         C         G         T
# pos
# 0   -0.183642  0.155864  0.094136 -0.066358
# 1   -0.290123  0.277778  0.111111 -0.098765
# 2   -0.276235  0.316358  0.168210 -0.208333
# ...

__data: {str: [float]} = {amino_acid: [0.0] * 70 for amino_acid in amino_acids}
__data["n"] = [0] * 70  # used for keeping track of amount of amino acids per position

sp_frequencies_dict: {str: pandas.DataFrame} = {state: pandas.DataFrame(__data) for state in annotations}

# Cumulative Moving Average (CMA) is hard to implement for this kind of frequencies
for sequence_index, row in sp_dataframe.iterrows():
    for index, (amino_acid, annotation) in enumerate(zip(row["sequence"], row["annotation"])):
        sp_frequencies_dict[annotation].loc[index, amino_acid] += 1
        sp_frequencies_dict[annotation].loc[index, "n"] += 1
        if debug:
            print("index=", index, " amino_acid=", amino_acid, " annotation=", annotation)
            print(sp_frequencies_dict[annotation].iloc[index])

frequency_dataframe: pandas.DataFrame
for annotation, frequency_dataframe in sp_frequencies_dict.items():

    filepath = "motif_logos/" + annotation + "_" + str(uuid.uuid4())

    frequency_dataframe.to_pickle(filepath + ".pkl")
    frequency_dataframe.to_string(filepath + ".txt")

    if rel_frequencies:
        # sp_frequencies_dict[annotation] = frequency_dataframe[frequency_dataframe.n != 0]
        for position in range(len(frequency_dataframe)):
            if not frequency_dataframe.at[position, "n"] == 0:  # otherwise we would get nan values (and who needs those rows anyway?) or skip them?
                for amino_acid in amino_acids:
                    frequency_dataframe.loc[position, amino_acid] /= frequency_dataframe.at[position, "n"]

    logo: logomaker.Logo = logomaker.Logo(frequency_dataframe[amino_acids], font_name='Arial Rounded MT Bold', color_scheme="chemistry")

    logo.style_spines(visible=False)
    logo.style_spines(spines=('left', 'bottom'), visible=True)
    logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    logo.ax.set_ylabel("Amino acid count", labelpad=-1)
    logo.ax.xaxis.set_ticks_position('none')
    logo.ax.xaxis.set_tick_params(pad=-1)

    logo.fig.show()
    logo.fig.savefig(filepath)

    print(filepath)
