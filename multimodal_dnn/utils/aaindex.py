import os
import re
import requests
from collections import defaultdict

AAINDEX_DOWNLOAD_URL = 'https://www.genome.jp/ftp/db/community/aaindex/aaindex1'
AAINDEX_FILE = '.aaindex1'


def download_aaindex():
    print(
        f"AAindex not found loacally. Downloading from: '{AAINDEX_DOWNLOAD_URL}'")

    request = requests.get(AAINDEX_DOWNLOAD_URL)

    with open(AAINDEX_FILE, 'w') as file:
        file.write(request.text)
        file.flush()

    print(f"Saved AAindex to: {AAINDEX_FILE}")


if not os.path.isfile(AAINDEX_FILE):
    download_aaindex()


def get_aaindex(accession_ids):
    aaindex = defaultdict(dict)

    with open(AAINDEX_FILE) as file:
        lines = file.readlines()

        current_id = ''
        current_index_first = []
        current_index_second = []

        for line in lines:
            if line.startswith('H'):
                for id in accession_ids:
                    if id in line:
                        current_id = id

            elif line.startswith('I') and current_id != '':
                index = line.lstrip('I').lstrip(' ').rstrip('\n')
                index = re.split(' +', index)
                index = list(map(lambda aas: aas.split('/'), index))
                current_index_first = [aas[0] for aas in index]
                current_index_second = [aas[1] for aas in index]

            elif current_index_first != []:
                values = line.lstrip(' ').rstrip('\n')
                values = re.split(' +', values)
                mapping = dict(zip(current_index_first, values))

                aaindex[current_id] = {
                    **aaindex[current_id],
                    **mapping
                }

                current_index_first = []

            elif current_index_second != []:
                values = line.lstrip(' ').rstrip('\n')
                values = re.split(' +', values)
                mapping = dict(zip(current_index_second, values))

                aaindex[current_id] = {
                    **aaindex[current_id],
                    **mapping
                }

                current_index_second = []
                current_id = ''

    if set(accession_ids) != set(aaindex.keys()):
        miss_ids = set(accession_ids).difference(set(aaindex.keys()))
        raise KeyError(
            f"Could not find accession ids {', '.join(miss_ids)} in AAindex")

    return dict(aaindex)
