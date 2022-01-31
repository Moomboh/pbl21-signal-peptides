#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn

from ..constants import *
from . import styles

seaborn.set_theme()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    help='The training dataset file. The file must be in the project-specific tsv format.')
parser.add_argument('--output', type=str)
args = parser.parse_args()


def prepend_output_path(path):
    return os.path.join(args.output, path)

Path(args.output).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.dataset, sep='\t')


counts = defaultdict(dict)

for sp_type in ANNOTATION_4STATE_LABELS:
    for kingdom in KINGDOMS:
        counts[sp_type][kingdom] = len(df[
            (df['sp_type'] == sp_type)
             & (df['kingdom'] == kingdom)]
        )

count_df = pd.DataFrame.from_dict(counts)


type_percent = {}

for sp_type in ANNOTATION_4STATE_LABELS:
    percent = float(count_df[sp_type].sum() / count_df.to_numpy().sum()) * 100
    type_percent[sp_type] = f'{percent:>0.1f}%'

count_df_t = count_df.transpose()

kingdom_percent = {}

for kingdom in KINGDOMS:
    percent = float(count_df_t[kingdom].sum() / count_df_t.to_numpy().sum()) * 100
    kingdom_percent[kingdom] = f'{percent:>0.1f}%'


def replace_kingdom_ylabels(kingdom_prct):
    ks = styles.kingdom_styles
    ylocs, ylabels = plt.yticks()
    ylabels = [f"{ks[l.get_text()]['label']} ({kingdom_prct[l.get_text()]})" for l in ylabels]
    plt.yticks(ylocs, ylabels)

def replace_sp_type_legend(type_prct):
    cs = styles.class_styles
    legend = plt.gca().get_legend()
    labels = [f"{cs[l.get_text()]['label']} ({type_prct[l.get_text()]})" for l in legend.texts]
    plt.legend(labels)

def replace_kingdom_legend(kingdom_prct):
    ks = styles.kingdom_styles
    legend = plt.gca().get_legend()
    labels = [f"{ks[l.get_text()]['label']} ({kingdom_prct[l.get_text()]})" for l in legend.texts]
    plt.legend(labels)

def replace_sp_type_ylabels(type_prct):
    cs = styles.class_styles
    ylocs, ylabels = plt.yticks()
    ylabels = [f"{cs[l.get_text()]['label']} ({type_prct[l.get_text()]})" for l in ylabels]
    plt.yticks(ylocs, ylabels)


count_df.plot.barh(stacked=True)
plt.gca().invert_yaxis()
replace_kingdom_ylabels(kingdom_percent)
replace_sp_type_legend(type_percent)
plt.tight_layout()
plt.savefig(prepend_output_path('dataset_distribution.png'))
plt.close()

eukarya_df = count_df.drop(index=['ARCHAEA', 'POSITIVE', 'NEGATIVE'])
eukarya_df.plot.barh(stacked=True)
plt.gca().invert_yaxis()
replace_kingdom_ylabels(kingdom_percent)
replace_sp_type_legend(type_percent)
plt.tight_layout()
plt.savefig(prepend_output_path('eukarya_dataset_distribution.png'))
plt.close()

rest_df = count_df.drop(index=['EUKARYA'])
rest_df.plot.barh(stacked=True)
plt.gca().invert_yaxis()
replace_kingdom_ylabels(kingdom_percent)
replace_sp_type_legend(type_percent)
plt.tight_layout()
plt.savefig(prepend_output_path('rest_dataset_distribution.png'))
plt.close()



no_sp_kingdom_df =count_df_t.drop(index=['SP', 'TAT', 'LIPO'])

nosp_kgd_prct = {}

for kingdom in KINGDOMS:
    percent = float(no_sp_kingdom_df[kingdom].sum() / no_sp_kingdom_df.to_numpy().sum()) * 100
    nosp_kgd_prct[kingdom] = f'{percent:>0.1f}%'

no_sp_kingdom_df.plot.barh(stacked=True)
plt.gca().invert_yaxis()
replace_kingdom_legend(nosp_kgd_prct)
replace_sp_type_ylabels(type_percent)
plt.tight_layout()
plt.savefig(prepend_output_path('no_sp_kingdom_distribution.png'))
plt.close()