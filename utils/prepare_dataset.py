#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Downloads and prepares the dataset

1. Downloads the SignalP5.0 dataset from their server
2. Converts the 3-line fasta format to a tsv format
3. Cleans the dataset (currently only removes sequences with length other than 70)
4. Splits the dataset into training and test sets
"""

from tqdm import tqdm
import requests
import pandas as pd
import argparse
from pathlib import Path
from itertools import zip_longest


parser = argparse.ArgumentParser()
parser.add_argument('--output-path', type=str)
parser.add_argument('--dataset-url', type=str, default='https://services.healthtech.dtu.dk/services/SignalP-5.0/train_set.fasta')
args = parser.parse_args()

Path(args.output_path).mkdir(parents=True, exist_ok=True)


###
# Constants
###
dataset_download_url = args.dataset_url
raw_fasta_path = f'{args.output_path}/raw_dataset.fasta'

raw_tsv_path = f'{args.output_path}/raw_dataset.tsv'

clean_tsv_path = f'{args.output_path}/dataset.tsv'

test_partition_no = 0
test_tsv_path = f'{args.output_path}/test_set.tsv'
train_tsv_path = f'{args.output_path}/train_set.tsv'


###
# 1. Download the dataset
###
print('⏳ Downloading dataset from: ' + dataset_download_url)

request = requests.get(dataset_download_url, stream=True)

with open(raw_fasta_path, 'wb') as file:
    total_length = int(request.headers.get('content-length'))

    for chunk in tqdm(request.iter_content(chunk_size=1024), total=(total_length/1024) + 1, unit='KB'):
        if chunk:
            file.write(chunk)
            file.flush()

print('✔️ Downloaded dataset!\n')

###
# 2. Convert the dataset to tsv
###
print('⏳ Converting data to tsv...')


class Datapoint:
    def __init__(self, metadata, sequence, annotation):
        self.sequence = sequence.strip('\n')
        self.annotation = annotation.strip('\n')

        metadata = metadata.split('|')

        self.uniprot_ac = metadata[0].strip('>')
        self.kingdom = metadata[1]
        self.sp_type = metadata[2]
        self.partition_no = metadata[3].strip('\n')

    def to_dict(self):
        return {
            'sequence': self.sequence,
            'annotation': self.annotation,
            'uniprot_ac': self.uniprot_ac,
            'kingdom': self.kingdom,
            'sp_type': self.sp_type,
            'partition_no': self.partition_no,
        }


raw_data = []

num_of_lines = sum(1 for line in open(raw_fasta_path))

with open(raw_fasta_path) as file:
    for lines in zip_longest(*[file] * 3):
        raw_data.append(Datapoint(lines[0], lines[1], lines[2]))

raw_df = pd.DataFrame(
    [datapoint.to_dict() for datapoint in raw_data]
).astype({
    'sequence': str,
    'annotation': str,
    'uniprot_ac': str,
    'kingdom': str,
    'sp_type': str,
    'partition_no': int,
})

raw_df.to_csv(raw_tsv_path, sep='\t', index=False)

print('✔️ Converted dataset to tsv!\n')


###
# 3. Clean the dataset (currently only removes sequences with length other than 70)
###
print('⏳ Cleaning dataset...')

clean_df = raw_df[raw_df.sequence.str.len() == 70]

clean_df.to_csv(clean_tsv_path, sep='\t', index=False)

print('✔️ Cleaned dataset!\n')


###
# 4. Split dataset into training and test set
###
print('⏳ Splitting training and test data...')

test_df = clean_df[clean_df.partition_no == test_partition_no]
train_df = clean_df[clean_df.partition_no != test_partition_no]

test_df.to_csv(test_tsv_path, sep='\t', index=False)
train_df.to_csv(train_tsv_path, sep='\t', index=False)

print('✔️ Splitted training and test sets!\n')


print('✅ Done!')
