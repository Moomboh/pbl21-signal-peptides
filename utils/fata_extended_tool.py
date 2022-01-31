#!/usr/bin/env python3

# maxim
# currently not in use and not tested properly

from utils.fasta_tool import *
from enum import Enum
from deprecated import deprecated


@deprecated  # somewhere is a bug !
class FastaExtended:

    class AminoAcidAnnotated:

        # according to http://www.cbs.dtu.dk/services/SignalP-5.0/data.php
        # S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular
        class Annotation(Enum):
            S = 0
            T = 1
            L = 2
            I = 3
            M = 4
            O = 5

            @staticmethod
            def parse_annotation(key: str):
                dic = {
                    "S": FastaExtended.AminoAcidAnnotated.Annotation.S,
                    "T": FastaExtended.AminoAcidAnnotated.Annotation.T,
                    "L": FastaExtended.AminoAcidAnnotated.Annotation.L,
                    "I": FastaExtended.AminoAcidAnnotated.Annotation.I,
                    "M": FastaExtended.AminoAcidAnnotated.Annotation.M,
                    "O": FastaExtended.AminoAcidAnnotated.Annotation.O,
                }

                value = dic[key]
                if value is None:
                    raise KeyError

                return value

        def __init__(self, amino_acid: chr, annotation: Annotation):
            self.__amino_acid = amino_acid
            self.__annotation = annotation

        def get_amino_acid(self): return self.__amino_acid
        def get_annotation(self): return self.__annotation

    class SpType(Enum):
        NONE = 0
        SP = 1
        TAT = 2
        LIPO = 3

        @staticmethod
        def parse_sp_type(key: str):
            dic = {
                "NONE": FastaExtended.SpType.NONE,
                "NO_SP": FastaExtended.SpType.NONE,
                "SP": FastaExtended.SpType.SP,
                "TAT": FastaExtended.SpType.TAT,
                "LIPO": FastaExtended.SpType.LIPO,
            }

            value = dic[key]
            if value is None:
                raise KeyError

            return value

    def __init__(self, fasta: Fasta, uniprot_ac: str, kingdom: str, sp_type: SpType, partition_no: int):
        self.__fasta = fasta
        self.__uniprot_ac = uniprot_ac
        self.__kingdom = kingdom
        self.__sp_type = sp_type
        self.__partition_no = partition_no
        self.__amino_acids_annotated = []
        for amino_acid, annotation in zip(fasta.get_content()[0], fasta.get_content()[1]):
            self.__amino_acids_annotated.append(FastaExtended.AminoAcidAnnotated(amino_acid, annotation))

    def get_fasta(self): return self.__fasta
    def get_uniprot_ac(self): return self.__uniprot_ac
    def get_kingdom(self): return self.__kingdom
    def get_sp_type(self): return self.__sp_type
    def get_partition_no(self): return self.__partition_no
    def get_amino_acid_annotated(self): return self.__amino_acids_annotated

    @staticmethod
    def __parse_fasta(fasta: Fasta):
        head_split = fasta.get_head().split("|")  # >Q1ENB6|EUKARYA|NO_SP|1
        return FastaExtended(fasta, uniprot_ac=head_split[0], kingdom=head_split[1],
                             sp_type=FastaExtended.SpType.parse_sp_type(head_split[2]), partition_no=int(head_split[3]))

    @staticmethod
    def get_fasta_extended_from_file(file_path: str):
        for fasta in Fasta.scan_fasta_file(file_path):
            yield FastaExtended.__parse_fasta(fasta)
