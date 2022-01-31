# cache dirs
DATASET_CACHE_DIR = "./.cache/multimodal_dnn_dataset_cache"
METRIC_CACHE_DIR = "./.cache/multimodal_dnn_metric_cache"
ATTRIBUTION_CACHE_DIR = "./.cache/multimodal_dnn_attribution_cache"

# numerical constants
Z_SCORE_95_PERCENT = 1.95996


# default cross-validation partitions
PARTITIONS = [1, 2, 3, 4]

# sequence length
SEQ_LENGTH = 70

# amino acids in sequence
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")

### constants for dataset tsv-files
# column labels for dataset
COL_TYPE = "sp_type"
COL_SEQUENCE = "sequence"
COL_ANNOTATION = "annotation"
COL_KINGDOM = "kingdom"

COL_TYPE_PRED = "sp_type_pred"
COL_ANNOT_PRED = "annotation_pred"


# types column values
TYPE_SP = "SP"
TYPE_LIPO = "LIPO"
TYPE_TAT = "TAT"
TYPE_NO_SP = "NO_SP"

# type to annotation char mapping
TYPE_ANNOT_CHAR = {
    TYPE_SP: "S",
    TYPE_LIPO: "L",
    TYPE_TAT: "T",
    TYPE_NO_SP: "X",
}

ANNOT_TYPE_LABEL = {
    "X": "NO_SP",
    "S": "SP",
    "L": "LIPO",
    "T": "TAT",
}

# all types
TYPES = [TYPE_SP, TYPE_LIPO, TYPE_TAT, TYPE_NO_SP]

# sp types only
SP_TYPES = [TYPE_SP, TYPE_LIPO, TYPE_TAT]

# organism groups in dataset in kingdom column
KINGDOMS = ["EUKARYA", "ARCHAEA", "POSITIVE", "NEGATIVE"]

### constants for metrics tsv-files
METRICS_IDX_HEADER_TARGET = "target"
METRICS_IDX_HEADER_GROUP = "group"
METRICS_IDX_HEADER_LABEL = "label"
METRICS_IDX_HEADER_MODEL = "model"

COL_MCC = "mcc"
COL_MCC_CI = "mcc_ci"

ALL_LABEL = "all"
PROTEIN_4STATE_LABEL = "protein_4state"
RESIDUE_6STATE_LABEL = "residue_6state"

# annotation labels
ANNOTATION_4STATE_CHARS = ["X", "S", "L", "T"]
ANNOTATION_4STATE_LABELS = ["NO_SP", "SP", "LIPO", "TAT"]

ANNOTATION_6STATE_CHARS = ["I", "M", "O", "S", "L", "T"]
ANNOTATION_6STATE_LABELS = ["IN", "MEMBR", "OUT", "SP", "LIPO", "TAT"]

ANNOTATION_9STATE_CHARS = ["I", "M", "N", "O", "S", "L", "T", "C", "D"]
ANNOTATION_9STATE_LABELS = [
    "INNER",
    "TM_IN_OUT",
    "TM_OUT_IN",
    "OUTER",
    "SP",
    "LIPO",
    "TAT",
    "SP_CLEAVE",
    "LIPO_CLEAVE",
]

ANNOTATION_NO_SP_CHARS = ["I", "M", "O", "N", "C", "D"]
ANNOT_SP_CHARS = set(ANNOTATION_9STATE_CHARS) - set(ANNOTATION_NO_SP_CHARS)
NON_SP_ANNOTATION_SUB = "X"

ANNOT_9_TO_4CHARS = {
    "I": "X",
    "M": "X",
    "N": "X",
    "O": "X",
    "S": "S",
    "L": "L",
    "T": "T",
    "C": "X",
    "D": "X",
}

#####
# BLOSUM62 matrix
#
# BLOSUM62.tsv was generated form matrix file from
# https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
# with comments removed and converted to tsv.
# BLOSUM62 constant array was then created using following code:
#
# blosum62_matrix = pandas.read_csv('./BLOSUM62.tsv', sep='\t', index_col=0)
# blosum62_matrix = blosum62_matrix.drop(columns=['B', 'Z', 'X', '*'])
# blosum62_matrix = blosum62_matrix.drop(index=['B', 'Z', 'X', '*'])
# print(blosum62_matrix.to_numpy())
#
####

# fmt: off
BLOSUM62 = \
[[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0], # A
 [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3], # R
 [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3], # n
 [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3], # d
 [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1], # c
 [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2], # Q
 [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2], # E
 [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3], # G
 [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3], # H
 [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3], # I
 [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1], # L
 [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2], # K
 [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1], # M
 [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1], # F
 [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2], # P
 [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2], # S
 [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0], # T
 [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3], # W
 [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1], # Y
 [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]] # V
# fmt: on
