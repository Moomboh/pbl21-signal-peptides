import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.utilities.data import to_onehot
from pathlib import Path

from ..constants import *
from .. import transforms
from ..dataset import SignalPeptideDataset
from .Model import Model
from ..utils import helpers


class BaseModelNetwork(nn.Module):
    def __init__(self, hyperparams):
        super(BaseModelNetwork, self).__init__()

        self.hp = hyperparams

        self.linear = nn.Linear(
            in_features=SEQ_LENGTH *
            (len(AMINO_ACIDS) + len(self.hp['aaindex_ids'])),
            out_features=SEQ_LENGTH * len(ANNOTATION_6STATE_CHARS),
        )

        self.linear6to4 = nn.Linear(
            in_features=SEQ_LENGTH * len(ANNOTATION_6STATE_CHARS),
            out_features=len(ANNOTATION_4STATE_LABELS),
        )

    def forward(self, x: torch.Tensor, kingdom=None) -> torch.Tensor:
        shape = x.size()
        x = x.flatten(start_dim=1)

        x = self.linear(x)

        x_annot = x.view(shape[0], shape[1], len(ANNOTATION_6STATE_CHARS))

        x_type = self.linear6to4(x_annot.flatten(start_dim=1))

        x_annot = F.softmax(x_annot, dim=2)
        x_type = F.softmax(x_type, dim=1)

        return x_annot, x_type


class BaseModel(Model):
    def __init__(self, background, learning_rate, device, loss_weight={
        'annot': 1,
        'sp_type': 1,
    }, hyperparams={
        'aaindex_ids': sorted([
            # Normalized frequency of alpha-helix (Chou-Fasman, 1978b)
            'CHOP780201',
            # Normalized frequency of beta-sheet (Chou-Fasman, 1978b)
            'CHOP780202',
            # Normalized frequency of beta-turn (Chou-Fasman, 1978b)
            'CHOP780203',
            # Normalized van der Waals volume (Fauchere et al., 1988)
            'FAUJ880103',
            'KLEP840101',  # Net charge (Klein et al., 1984)
            'KYTJ820101',  # Hydropathy index (Kyte-Doolittle, 1982)
            'MITS020101',  # Amphiphilicity index (Mitaku et al., 2002)
            'RADA880108',  # Mean polarity (Radzicka-Wolfenden, 1988)
        ])
    }):
        self.device = device
        self.learning_rate = learning_rate
        self.hyperparams = hyperparams

        if not hasattr(self, 'feature_encoder') and 'aaindex_ids' in self.hyperparams:
            self.feature_encoder = transforms.FeatureEncoder(
                hyperparams['aaindex_ids'])

        if not hasattr(self, 'network'):
            self.network = BaseModelNetwork(hyperparams)
            self.network.to(self.device)

        self.loss_weight = loss_weight

        annot_background = background['annot']
        type_background = background['type']

        annot_weight = torch.ones_like(annot_background).div(annot_background)
        self.loss_module = nn.CrossEntropyLoss(weight=annot_weight)
        self.loss_module.to(self.device)

        type_weight = torch.ones_like(type_background).div(type_background)
        self.sp_type_loss_module = nn.CrossEntropyLoss(weight=type_weight)
        self.sp_type_loss_module.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=learning_rate)

    def loss(self, pred_annot, pred_type, y_annot, y_type):
        pred_annot = pred_annot.flatten(start_dim=0, end_dim=1).float()
        y_annot = y_annot.argmax(2).flatten()

        pred_type = pred_type.flatten(start_dim=0, end_dim=0).float()
        y_type = y_type.argmax(1)

        loss = (self.loss_weight['annot'] * self.loss_module(pred_annot, y_annot)
                + self.loss_weight['sp_type'] * self.sp_type_loss_module(pred_type, y_type)) \
            / (self.loss_weight['annot'] + self.loss_weight['sp_type'])

        return loss

    def train_batch(self, X, y, context):
        y_annot = y
        y_type = context[1].to(self.device)
        kingdom = context[0].to(self.device)

        pred_annot, pred_type = self.network(X, kingdom)
        loss = self.loss(pred_annot, pred_type, y_annot, y_type)

        # Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return pred_annot, y_annot, pred_type, y_type, kingdom, loss

    def validate_batch(self, X, y, context):
        y_annot = y
        y_type = context[1].to(self.device)
        kingdom = context[0].to(self.device)

        pred_annot, pred_type = self.network(X, kingdom)
        loss = self.loss(pred_annot, pred_type, y_annot, y_type)

        return pred_annot, y_annot, pred_type, y_type, kingdom, loss

    def before_metrics(self, pred_annot, y_annot, pred_type, y_type, kingdom):
        pred_annot_shape = pred_annot.size()
        pred_type_shape = pred_type.size()

        pred_annot = pred_annot.argmax(2)
        pred_annot = pred_annot.flatten()
        pred_annot = to_onehot(pred_annot, len(ANNOTATION_6STATE_CHARS))

        pred_annot = pred_annot.reshape(
            pred_annot_shape[0], pred_annot_shape[1], len(ANNOTATION_6STATE_CHARS))

        pred_type = pred_type.argmax(1)
        pred_type = pred_type.flatten()
        pred_type = to_onehot(pred_type, len(ANNOTATION_4STATE_LABELS))

        pred_type = pred_type.reshape(
            pred_type_shape[0], 1, pred_type_shape[1])
        y_type = y_type.reshape(
            pred_type_shape[0], 1, pred_type_shape[1])

        kingdom_shape = kingdom.size()
        kingdom_annot = kingdom.expand(
            kingdom_shape[0], pred_annot_shape[1], kingdom_shape[2])

        kingdom_type = kingdom.expand(
            pred_type_shape[0], 1, pred_type_shape[1])

        return pred_annot, y_annot, kingdom_annot, pred_type, y_type, kingdom_type

    def to(self, device):
        self.device = device
        self.network.to(device)
        self.loss_module.to(device)
        helpers.optimizer_to(self.optimizer, device)

    @staticmethod
    def class_labels():
        return ANNOTATION_6STATE_LABELS

    @staticmethod
    def context_labels():
        return KINGDOMS

    def transform_input(self, x) -> torch.Tensor:
        x = transforms.aa_seq_to_one_hot(list(x))

        x_blosum62 = transforms.blosum62_encode(x)
        x_blosum62 = torch.tensor(x_blosum62, dtype=torch.float)

        x_features = self.feature_encoder.transform(x)
        x_features = torch.tensor(x_features, dtype=torch.float)

        x = torch.cat((x_blosum62, x_features), dim=1)

        return x

    def transform_target(self, y) -> torch.Tensor:
        y = transforms.annotation_6state_to_one_hot(list(y))
        y = torch.tensor(y)

        return y

    def transform_context(self, c) -> torch.Tensor:
        kingdom = c[0]
        kingdom = transforms.kingdom_to_one_hot([kingdom])
        kingdom = torch.tensor(kingdom, dtype=torch.float)

        sp_type = c[1]
        sp_type = transforms.sp_type_to_one_hot([sp_type])[0]
        sp_type = torch.tensor(sp_type, dtype=torch.float)

        return (kingdom, sp_type)

    def get_state(self):
        self.to('cpu')

        state = {
            'model_state':{
                'state_dict': self.network.state_dict(),
                'hyperparams': self.hyperparams
            },
            'optimizer_state': self.optimizer.state_dict(),
        }

        self.to(self.device)
        return state

    def load_state(self, state):
        self.hyperparams = state['model_state']['hyperparams']

        self.network = type(self.network)(self.hyperparams)
        self.network.load_state_dict(state['model_state']['state_dict'])

        self.optimizer = type(self.optimizer)(self.network.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(state['optimizer_state'])

        self.to(self.device)

    @classmethod
    def get_background(cls, train_file, partitions, flush_cache=False, log=False):
        cache_dir = Path(DATASET_CACHE_DIR, helpers.slugify(cls.__name__))
        pickle_path = Path(
            cache_dir,
            f"{helpers.slugify(train_file)}_partitions-{helpers.slugify(partitions)}.pth"
        )

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        if flush_cache:
           pickle_path.unlink(missing_ok=True)
        elif pickle_path.exists():
            if log:
                print(f"Loading background freq cache from: {str(pickle_path)}")
            return torch.load(pickle_path)


        dummy_background = {
            'annot': torch.tensor(0.25).repeat(len(ANNOTATION_6STATE_CHARS)),
            'type': torch.tensor(0.25).repeat(len(ANNOTATION_4STATE_LABELS)),
        }
        dummy_model = cls(dummy_background, 1e-3, 'cpu')

        dataset = SignalPeptideDataset(
            train_file,
            partitions=partitions,
            model=dummy_model,
            device='cpu',
        )

        dataloader = DataLoader(dataset, batch_size=len(dataset))

        data = next(iter(dataloader))

        annot_target = data[1]
        type_target = data[2][1]

        annot_target = torch.argmax(annot_target, 2).flatten()
        type_target = torch.argmax(type_target, 1)

        annot_len = annot_target.size()[0]
        type_len = type_target.size()[0]

        _, annot_counts = annot_target.unique(return_counts=True)
        _, type_counts = type_target.unique(return_counts=True)

        del dataset, dummy_model

        background = {
            'annot': annot_counts.div(annot_len),
            'type': type_counts.div(type_len),
        }

        torch.save(background, pickle_path)

        if log:
            print(f"Saving background freq cache to: {str(pickle_path)}")

        return background
