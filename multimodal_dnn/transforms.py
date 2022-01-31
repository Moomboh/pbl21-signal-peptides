import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import normalize

from .constants import *
from .utils.OneHotEncoder import OneHotEncoder
from .utils.aaindex import get_aaindex

aa_encoder = OneHotEncoder(categories=AMINO_ACIDS)


def aa_seq_to_one_hot(sequence):
    return aa_encoder.transform(sequence)


annotation_4state_encoder = OneHotEncoder(categories=ANNOTATION_4STATE_CHARS)


def annotation_4state_to_one_hot(annotation):
    return annotation_4state_encoder.transform(annotation)


annotation_6state_encoder = OneHotEncoder(categories=ANNOTATION_6STATE_CHARS)


def annotation_6state_to_one_hot(annotation):
    return annotation_6state_encoder.transform(annotation)


annotation_9state_encoder = OneHotEncoder(categories=ANNOTATION_9STATE_CHARS)


def annotation_9state_to_one_hot(annotation):
    return annotation_9state_encoder.transform(annotation)


sp_type_encoder = OneHotEncoder(categories=ANNOTATION_4STATE_LABELS)


def sp_type_to_one_hot(sp_type):
    return sp_type_encoder.transform(sp_type)


kingdom_encoder = OneHotEncoder(categories=KINGDOMS)


def kingdom_to_one_hot(kingdom):
    return kingdom_encoder.transform(kingdom)


def reduce_no_sp_annotations(annotation):
    for char in ANNOTATION_NO_SP_CHARS:
        annotation = annotation.replace(char, NON_SP_ANNOTATION_SUB)

    return annotation


def expand_annotation_to_9state(annotation):
    last_char = ''
    transformed = ''

    for char in annotation:
        transformed_char = ''

        if char == 'M':
            if last_char == 'I':
                transformed_char = 'M'
            elif last_char == 'O':
                transformed_char = 'N'
            elif last_char == 'M':
                transformed_char = 'M'
            elif last_char == 'N':
                transformed_char = 'N'
        elif char == 'O':
            if last_char == 'S':
                transformed_char = 'C'
            elif last_char == 'L':
                transformed_char = 'D'
            else:
                transformed_char = char
        else:
            transformed_char = char

        last_char = transformed_char
        transformed += transformed_char

    return transformed


def normalize_matrix(matrix):
    norm = np.linalg.norm(matrix)
    return matrix/norm


NORMALIZED_BLOSUM62 = normalize_matrix(BLOSUM62)


def blosum62_encode(one_hot_seq):
    return np.matmul(one_hot_seq, NORMALIZED_BLOSUM62)


class FeatureEncoder:
    def __init__(self, aaindex_ids):
        self.aaindex = get_aaindex(aaindex_ids)

        self.aaindex = {
            id: OrderedDict(
                sorted(aamap.items(), key=lambda i: AMINO_ACIDS.index(i[0])))
            for id, aamap in self.aaindex.items()
        }

        self.aaindex = OrderedDict(sorted(self.aaindex.items()))

        self.aaindex = np.transpose(normalize([
            list(row.values())
            for _, row in self.aaindex.items()
        ]))

    def transform(self, one_hot_seq):
        return np.matmul(one_hot_seq, self.aaindex)

def get_label_indices(labels, include_labels):
    sp_indices = [labels.index(l) for l in include_labels]

    return sp_indices

def get_label_diff_indices(labels, exclude_labels):
    diff_labels = [l for l in labels if l not in exclude_labels]

    return get_label_indices(labels, diff_labels)

def get_label_index_map(from_labels, to_labels, label_map):
    index_map = {}

    for i, l in enumerate(from_labels):
        to_index = to_labels.index(label_map[l])
        index_map[i] = to_index
    
    return index_map

