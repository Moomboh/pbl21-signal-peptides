#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Outputs random baseline performance measures for 4-state per protein prediction"""

import argparse
import pandas as pd
import pycm

#read data and create dataframe
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str,
                    help='The training dataset file. The file must be in the project-specific tsv format.')
args = parser.parse_args()
df = pd.read_csv(args.input_file, sep='\t')

#calculate absolute and relative frequency of sp types
sp_type_absolute = df['sp_type'].value_counts()
rel = pd.DataFrame(sp_type_absolute / df['sp_type'].size)

#transpose relative frequency for confusion matrix
rel_t = rel.transpose()

#confusion matrix
cm = (rel.dot(rel_t) * 10000000).round().astype(int).to_dict()
cm = pycm.ConfusionMatrix(matrix=cm)

print(sp_type_absolute)
print(rel_t)
print(cm.stat(overall_param=['Overall MCC', 'ACC Macro', 'PPV Macro', 'TPR Macro'], class_param=['MCC', 'ACC', 'PPV', 'TPR']))