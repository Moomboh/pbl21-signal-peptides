PATHWAYS = {'S': "Sec/SPI", 'T': "Tat/SPI", 'L': "Sec/SPII", 'X': "None"}
LOCATIONS = {'I': "cytoplasm", 'M': "membrane", 'O': "extracellular"}
TYPE_CODES = {"LIPO": "Sec/SPII", "TAT": "Tat/SPI", "SP": "Sec/SPI", "NO_SP": "None"}
TYPES = {"LIPO", "TAT", "SP", "NO_SP"}
METRIC_TYPES = {*TYPES, "overall"}

PATHWAY_ANNOTATIONS = {"LIPO": 'L', "TAT": 'T', "SP": 'S'}
annotation_mapping = {
    'S': 0,
    'T': 1,
    'L': 2,
    'I': 3,
    'M': 4,
    'O': 5
}
reverse_annotation_mapping = {val: key for key, val in annotation_mapping.items()}
AMINO_ACID_CODES = "ACDEFGHIKLMNPQRSTVWY"
amino_acid_mapping = {c: AMINO_ACID_CODES.index(c) for c in AMINO_ACID_CODES}

KINGDOMS = ["EUKARYA", "ARCHAEA", "POSITIVE", "NEGATIVE"]
METRIC_KINGDOMS = [*KINGDOMS, "overall"]

TRAINING_PARTITIONS = {1, 2, 3, 4}
ALL_PARTITIONS = {0, 1, 2, 3, 4}
