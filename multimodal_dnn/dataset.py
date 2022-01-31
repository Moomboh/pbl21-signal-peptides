import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

from .constants import *
from .utils import helpers
class SignalPeptideDataset(Dataset):
    def __init__(self, dataset_file, partitions, model, device, filter_query=None):
        self.sp_data = pd.read_csv(
            dataset_file,
            sep='\t',
            usecols=['sequence', 'annotation', 'kingdom', 'sp_type', 'partition_no']
        ).astype({
            'sequence': str,
            'annotation': str,
            'kingdom': str,
            'sp_type': str,
            'partition_no': int,
        })

        self.sp_data = self.sp_data[self.sp_data.partition_no.isin(partitions)]
        self.model = model
        self.device = device

        if filter_query:
            self.sp_data = self.sp_data.query(filter_query)

        self.sp_data.reset_index(drop=True, inplace=True)

        self.cache = {}
        self.from_pickle = False
        self.cache_dir = Path(DATASET_CACHE_DIR, helpers.slugify(type(model)))
        self.pickle_path = Path(
            self.cache_dir,
            f"{helpers.slugify(dataset_file)}_partitions-{helpers.slugify(partitions)}.pth"
        )

    def __len__(self):
        return len(self.sp_data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]['sequence'], self.cache[idx]['annotation'], self.cache[idx]['context']

        sequence = self.sp_data.loc[idx].at['sequence']
        annotation = self.sp_data.loc[idx].at['annotation']
        context = (
            self.sp_data.loc[idx].at['kingdom'],
            self.sp_data.loc[idx].at['sp_type']
        )

        sequence = self.model.transform_input(sequence).to(self.device)

        annotation = self.model.transform_target(annotation).to(self.device)

        context = self.model.transform_context(context)
        context = (context[0].to(self.device), context[1].to(self.device))

        self.cache[idx] = {
            'sequence': sequence,
            'annotation': annotation,
            'context': context,
        }

        return sequence, annotation, context

    def unpickle(self, flush_cache=False, log=False):
        if flush_cache:
            self.pickle_path.unlink(missing_ok=True)
        elif self.pickle_path.exists():
            self.from_pickle = True
            if log:
                print(f"Loading dataset cache from: {str(self.pickle_path)}")
            self.cache = torch.load(self.pickle_path)
            self.to(self.device)

    def pickle(self, log=False):
        self.to('cpu')
        if not self.from_pickle:
            if log:
                print(f"Saving dataset cache to: {str(self.pickle_path)}")
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.cache, self.pickle_path)
    
    def to(self, device):
        self.device = device
        self.cache = {
            idx: {
                'sequence': item['sequence'].to(self.device),
                'annotation': item['annotation'].to(self.device),
                'context': (item['context'][0].to(self.device), item['context'][1].to(self.device)),
            }
            for idx, item in self.cache.items()
        }

