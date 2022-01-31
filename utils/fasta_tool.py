#!/usr/bin/env python3

# maxim
# currently not in use and not tested properly

import re
import os
import sys
import warnings
from deprecated import deprecated


@deprecated
class Fasta:

    def __init__(self, head: str, content: [str], comments: [str] = None):
        if comments is None: comments = []
        self.__head = head
        self.__content = content
        self.__comments = comments

    def get_head(self): return self.__head
    def get_content(self): return self.__content
    def get_content_length(self): return len(self.__content)
    def get_comments(self): return self.__comments

    # for encoding see https://en.wikipedia.org/wiki/FASTA_format
    @staticmethod
    def scan_string(fasta_raw: str):
        sequence_parts = re.findall(r">.+\n[A-Z\-*;a-z \n]+", fasta_raw)  # parsing every sequence part

        for sequence in sequence_parts:

            sequence_head_match = re.search(r">.+\n", sequence)
            if sequence_head_match is None:
                raise AssertionError(f"No Sequence head Matched:\n{sequence}")

            sequence_comments_matches = re.findall(r";.+\n?", sequence)

            content_raw = sequence.split('\n', 1)[1]
            content = []

            for line in content_raw:
                if not (line.startswith(";") or line.startswith(">")):
                    content.append(line.upper())

            yield Fasta(sequence_head_match.group(0).replace('\n', '').replace('>', '', 1),
                        content,
                        [comment.replace('\n', '').replace(';', '', 1)
                         for comment in sequence_comments_matches])

    @staticmethod
    def scan_fasta_file(file_path: str):
        if file_path == "-":
            content = "".join(sys.stdin.readlines())
            print(content)
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_path} is not a file.")
            if os.path.splitext(file_path)[1] != '.fasta':
                warnings.warn(f"{file_path} does not have .fasta extension")

            with open(file_path) as file:
                content = file.read()

        return Fasta.scan_string(content)

    def get_as_text(self):
        return '>' + self.__head + '\n' \
               + '\n'.join([';' + comment for comment in self.__comments]) + '\n' \
               + self.__content + '\n'

    def generate_fasta_file(self, directory_path, file_head):
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"{directory_path} does not exist or is not a directory.")
        file_path = os.path.join(directory_path, file_head + ".fasta")
        if os.path.exists(file_path):
            raise FileExistsError(f"{file_path} does already exist.")
        file = open(file_path, "w")
        file.write(self.get_as_text())
