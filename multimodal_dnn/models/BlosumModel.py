import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import *
from .BaseModel import BaseModel


class BlosumModelNetwork(nn.Module):
    def __init__(self, hyperparams):
        super(BlosumModelNetwork, self).__init__()

        self.hp = hyperparams

        self.conv1_blosum62 = nn.Conv1d(
            in_channels=len(AMINO_ACIDS),
            out_channels=self.hp['conv1_blosum62_out_channels'],
            kernel_size=self.hp['conv1_blosum62_kernel_size'],
            padding='same'
        )

        self.dropout1 = nn.Dropout(self.hp['dropout1_p'])

        self.lstm = nn.LSTM(
            input_size=self.hp['conv1_blosum62_out_channels'] + len(KINGDOMS),
            hidden_size=self.hp['lstm_hidden_size'],
            num_layers=self.hp['lstm_num_layers'],
            bidirectional=True,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(self.hp['dropout2_p'])

        self.conv2 = nn.Conv1d(
            in_channels=self.hp['lstm_hidden_size']*2,
            out_channels=self.hp['conv2_out_channels'],
            kernel_size=self.hp['conv2_kernel_size'],
            padding='same'
        )

        self.dropout3 = nn.Dropout(self.hp['dropout3_p'])

        self.conv3 = nn.Conv1d(
            in_channels=self.hp['conv2_out_channels'],
            out_channels=len(ANNOTATION_6STATE_CHARS),
            kernel_size=self.hp['conv3_kernel_size'],
            padding='same'
        )

        self.linear9to4 = nn.Linear(
            in_features=SEQ_LENGTH * len(ANNOTATION_6STATE_CHARS),
            out_features=len(ANNOTATION_4STATE_LABELS),
        )

    def forward(self, x: torch.Tensor, kingdom) -> torch.Tensor:
        shape = x.size()

        x = x.split(len(AMINO_ACIDS), dim=2)

        x_blosum62 = x[0]
        x_blosum62 = x_blosum62.transpose(1, 2)
        x_blosum62 = self.conv1_blosum62(x_blosum62)
        x_blosum62 = x_blosum62.transpose(2, 1)
        x_blosum62 = F.relu(x_blosum62)

        kingdom = kingdom.expand(shape[0], shape[1], len(KINGDOMS))

        x = torch.cat((x_blosum62, kingdom), 2)
        x = self.dropout1(x)

        x = self.lstm(x)[0]
        x = self.dropout2(x)

        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dropout3(x)

        x = F.relu(x)
        x = self.conv3(x)
        x_annot = x.transpose(2, 1)
        x_annot = F.softmax(x_annot, dim=2)

        x_type = self.linear9to4(x_annot.flatten(start_dim=1))
        x_type = F.softmax(x_type, dim=1)

        return x_annot, x_type


class BlosumModel(BaseModel):
    def __init__(self, background, learning_rate, device):
        self.hyperparams = {
            'conv1_blosum62_out_channels': 32,
            'conv1_blosum62_kernel_size': 3,
            'conv1_feature_out_channels': 32,
            'conv1_feature_kernel_size': 3,
            'dropout1_p': 0.2,
            'lstm_hidden_size': 64,
            'lstm_num_layers': 2,
            'dropout2_p': 0.2,
            'conv2_out_channels': 64,
            'conv2_kernel_size': 5,
            'dropout3_p': 0.2,
            'conv3_kernel_size': 1,
            'aaindex_ids': sorted([
                # Normalized frequency of alpha-helix (Chou-Fasman, 1978b)
                'CHOP780201',
                # Normalized frequency of beta-sheet (Chou-Fasman, 1978b)
                'CHOP780202',
                # Normalized frequency of beta-turn (Chou-Fasman, 1978b)
                'CHOP780203',
                # Normalized van der Waals volume (Fauchere et al., 1988)
                'FAUJ880103',
                'KLEP840101', # Net charge (Klein et al., 1984)
                'KYTJ820101', # Hydropathy index (Kyte-Doolittle, 1982)
                'MITS020101', # Amphiphilicity index (Mitaku et al., 2002)
                'RADA880108', # Mean polarity (Radzicka-Wolfenden, 1988)
                'CHAM810101', # Steric parameter (Charton, 1981)
                'CHAM830107', # A parameter of charge transfer capability (Charton-Charton, 1983)
                'JANJ780101', # Average accessible surface area (Janin et al., 1978)
                'MEIH800103', # Average side chain orientation angle (Meirovitch et al., 1980)
                'VELV850101', # Electron-ion interaction potential (Veljkovic et al., 1985)
                'WERD780101', # Propensity to be buried inside (Wertz-Scheraga, 1978)
                'ZIMJ680105', # RF rank (Zimmerman et al., 1968)
                'ZIMJ680104', # Isoelectric point (Zimmerman et al., 1968)
            ])
        }

        self.device = device

        self.network = BlosumModelNetwork(self.hyperparams)
        self.network.to(self.device)

        super(BlosumModel, self).__init__(
            background,
            learning_rate,
            device,
            loss_weight={
                'annot': 3,
                'sp_type': 1,
            },
            hyperparams=self.hyperparams
        )
