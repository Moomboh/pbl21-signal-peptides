
import os
import gc
import argparse
from pandas.core.frame import DataFrame
import torch
import torch.nn.functional as torchF
from pathlib import Path
from captum.attr import IntegratedGradients
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logomaker
import torchmetrics.utilities.data as tmutil

from ..constants import *
from . import plot_helpers
from ..utils import helpers
from .. import checkpoint
from .. import transforms
from . import styles



parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='+')
parser.add_argument('--dataset', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--partitions', type=int, nargs='+', default=[])
parser.add_argument('--background-partitions', type=int,
                    nargs='+', default=PARTITIONS)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--flush-cache',
                    dest='flush_cache', action='store_true')
parser.add_argument('--align-cs',
                    dest='align_cs', action='store_true')
parser.add_argument('--aaindex-max-4',
                    dest='aaindex_max_4', action='store_true')
parser.set_defaults(flush_cache=False)
parser.set_defaults(align_cs=False)
parser.set_defaults(aaindex_max_4=False)
args = parser.parse_args()


def prepend_output_path(path):
    return os.path.join(args.output, path)


Path(args.output).mkdir(parents=True, exist_ok=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def pickle_attributions(attrs, idstring):
    Path(ATTRIBUTION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    filename = idstring + '.pth'
    path = Path(ATTRIBUTION_CACHE_DIR, filename)

    print(f'Saving attributions to cache: {path}')
    torch.save(attrs, path)


def unpickle_attributions(idstring):
    path = Path(ATTRIBUTION_CACHE_DIR, idstring + '.pth')

    if Path.exists(path) and not args.flush_cache:
        print(f'Loading attributions from cache: {path}')
        return torch.load(path)
    
    return {}


def get_cache_idstring(model_filename, dataset):
    idstring = helpers.slugify(
        model_filename +
        f'_ds{dataset}_p{args.partitions}'
    )

    return idstring


class NetworkWrapper(torch.nn.Module):
    def __init__(self, network, default_output):
        super(NetworkWrapper, self).__init__()
        self.network = network
        self.default_output = default_output

    def forward(self, x, kingdom):
        self.result = self.network.forward(x, kingdom)

        if self.default_output == 'annotation':
            return self.result[0]

        if self.default_output == 'type':
            return self.result[1]
    
    def get_annot(self):
        if hasattr(self, 'result'):
            return self.result[0]
        else:
            raise RuntimeError('Cannot get annotation prediction. Call .forward() first.')

    def get_type(self):
        if hasattr(self, 'result'):
            return self.result[1]
        else:
            raise RuntimeError('Cannot get type prediction. Call .forward() first.')



for file in args.models:
    model_name = plot_helpers.get_model_name_from_filename(
        file, include_id=True)

    attributions = unpickle_attributions(
        get_cache_idstring(file, args.dataset)
    )

    if attributions == {}:
        fold_states = checkpoint.load(file)[0]

        for fold, fold_state in fold_states.items():
            model = plot_helpers.create_model(
                file, args.dataset, args.background_partitions, device)
            checkpoint.initialize(fold_state, model)

            partitions = args.partitions or [fold]

            dataloader = plot_helpers.get_dataloader(
                model,
                args.dataset,
                partitions,
                args.batch_size,
                device
            )

            network = NetworkWrapper(model.network, 'type')

            ig = IntegratedGradients(network)

            input_attrs = []
            kingdom_attrs = []
            kingdom_ctxs = []
            type_ctxs = []
            type_preds = []
            sp_lengths = []

            with torch.no_grad():
                for batch, (X, y, context) in enumerate(dataloader):
                    target = context[1].argmax(dim=1)

                    attr = ig.attribute((X, context[0]), target=target)

                    type_pred = network.forward(X, context[0])
                    type_pred = tmutil.to_onehot(
                        tmutil.to_categorical(type_pred, argmax_dim=1),
                        num_classes=len(ANNOTATION_4STATE_LABELS)
                    )

                    annot_target = tmutil.to_onehot(
                        tmutil.to_categorical(y, argmax_dim=2),
                        num_classes=len(ANNOTATION_6STATE_CHARS)
                    ).transpose(2, 1)

                    sp_indices = transforms.get_label_diff_indices(ANNOTATION_6STATE_CHARS, ANNOTATION_NO_SP_CHARS)

                    sp_length = torch.zeros(annot_target.size()[0], device=device)

                    for i in sp_indices:
                        sp_length += annot_target.select(2, i).sum(1)

                    input_attrs.append(attr[0])
                    kingdom_attrs.append(attr[1])
                    kingdom_ctxs.append(context[0])
                    type_ctxs.append(context[1])
                    type_preds.append(type_pred)
                    sp_lengths.append(sp_length)

            input_attrs = torch.squeeze(torch.cat(input_attrs))
            kingdom_attrs = torch.squeeze(torch.cat(kingdom_attrs))
            kingdom_ctxs = torch.squeeze(torch.cat(kingdom_ctxs))
            type_ctxs = torch.squeeze(torch.cat(type_ctxs))
            type_preds = torch.squeeze(torch.cat(type_preds))
            sp_lengths = torch.squeeze(torch.cat(sp_lengths))

            attributions[fold] = {
                'input_attrs': input_attrs.cpu(),
                'kingdom_attrs': kingdom_attrs.cpu(),
                'kingdom_ctxs': kingdom_ctxs.cpu(),
                'type_ctxs': type_ctxs.cpu(),
                'type_preds': type_preds.cpu(),
                'sp_lengths': sp_lengths.cpu(),
            }

            gc.collect()

        pickle_attributions(
            attributions,
            get_cache_idstring(file, args.dataset)
        )

    for fold in attributions.keys():
        attributions[fold]['input_attrs'] = attributions[fold]['input_attrs'].to(device)
        attributions[fold]['kingdom_attrs'] = attributions[fold]['kingdom_attrs'].to(device)
        attributions[fold]['kingdom_ctxs'] = attributions[fold]['kingdom_ctxs'].to(device)
        attributions[fold]['type_ctxs'] = attributions[fold]['type_ctxs'].to(device)
        attributions[fold]['type_preds'] = attributions[fold]['type_preds'].to(device)
        attributions[fold]['sp_lengths'] = attributions[fold]['sp_lengths'].to(device)

    for c in range(len(ANNOTATION_4STATE_LABELS)):
        fold_input_attrs = []
        fold_kingdom_attrs = []

        for fold, attr in attributions.items():
            input_attrs : torch.Tensor = attr['input_attrs']
            kingdom_attrs : torch.Tensor = attr['kingdom_attrs']
            kingdom_ctxs : torch.Tensor = attr['kingdom_ctxs']
            type_ctxs : torch.Tensor = attr['type_ctxs']
            type_preds : torch.Tensor = attr['type_preds']
            sp_lengths : torch.Tensor = attr['sp_lengths']

            class_indices = type_ctxs.select(1, c).nonzero().flatten()
            class_input_attrs = input_attrs.index_select(0, class_indices)
            class_kingdom_attrs = kingdom_attrs.index_select(0, class_indices)
            class_sp_lengths = sp_lengths.index_select(0, class_indices)

            class_input_attrs = torchF.pad(
                class_input_attrs,
                pad=(0, 0, 0, class_input_attrs.size()[1])
            )

            if args.align_cs:
                for i in range(len(class_input_attrs)):
                    class_input_attrs[i] = class_input_attrs[i].roll(SEQ_LENGTH - class_sp_lengths[i].int().item(), 0)

            fold_input_attrs.append(class_input_attrs)
            fold_kingdom_attrs.append(class_kingdom_attrs)

        input_attrs = torch.cat(fold_input_attrs)
        input_attrs = input_attrs.transpose(0, 2)

        input_attrs = input_attrs.mean(2).transpose(0, 1)

        aa_attrs = input_attrs[:,:20]
        feature_attrs = input_attrs[:,20:]

        Path(prepend_output_path(model_name)).mkdir(parents=True, exist_ok=True)

        # BLOSUM62 input sequence logos
        if torch.sum(torch.abs(aa_attrs)) > 0:
            aa_attrs_df = pd.DataFrame(aa_attrs.tolist(), columns=AMINO_ACIDS)

            logomaker.Logo(
                aa_attrs_df,
                flip_below=False,
            )

            plt.xlabel(styles.logo_styles[model_name]['xlabels'][c])
            plt.ylabel('Avg. attribution')

            xlocs, xlabels = plt.xticks()
            xlocs = np.arange(min(xlocs), max(xlocs)+1, step=5, dtype=int)
            xlabels = [l - SEQ_LENGTH for l in xlocs]
            plt.xticks(xlocs, xlabels)

            plt.xlim([lim + SEQ_LENGTH for lim in styles.logo_styles[model_name]['xlims'][c]])

            if args.align_cs:
                if c != 0:
                    plt.axvline(
                        x=SEQ_LENGTH-0.5,
                        color='grey',
                        linestyle='--',
                        linewidth=1,
                        label='cleavage_site'
                    )

            plt.tight_layout()

            plt.savefig(
                prepend_output_path(Path(
                    model_name,
                    f'{model_name}_{ANNOTATION_4STATE_LABELS[c]}_aa_attributions_logo.png'
                )),
                dpi=300
            )
            plt.close()

        # AAindex sequence logos
        if torch.sum(torch.abs(feature_attrs)) > 0:
            columns = [
                styles.aaindex_styles[model_name]['ylabels'][l]
                for l in styles.aaindex_styles[model_name]['aaindex_ids']
            ]

            plot_columns = [
                styles.aaindex_styles[model_name]['ylabels'][l]
                for l in styles.aaindex_styles['plot_features']
            ]

            feature_attrs_df = pd.DataFrame(feature_attrs.tolist(), columns=columns)

            if args.aaindex_max_4:
                column_sums = feature_attrs_df.abs().sum(axis=0)
                plot_columns = column_sums.nlargest(4).index

            feature_attrs_df[plot_columns].plot.line(
                linestyle='dotted',
                linewidth=1.5,
                marker='.',
                markersize=8,
            )

            plt.legend(
                loc='upper center',
                ncol=2,
                bbox_to_anchor=(0.5, 1.3),
                fancybox=True,
                shadow=False
            )

            plt.gcf().set_size_inches(10, 5)

            plt.xlabel(styles.aaindex_styles[model_name]['xlabels'][c])
            plt.ylabel('Avg. attribution')

            xlocs, xlabels = plt.xticks()
            xlocs = np.arange(min(xlocs), max(xlocs)+1, step=5, dtype=int)
            xlabels = [l - SEQ_LENGTH for l in xlocs]
            plt.xticks(xlocs, xlabels)

            plt.xlim([lim + SEQ_LENGTH for lim in styles.aaindex_styles[model_name]['xlims'][c]])

            if args.align_cs:
                if c != 0:
                    plt.axvline(
                        x=SEQ_LENGTH-0.5,
                        color='grey',
                        linestyle='--',
                        linewidth=1,
                        label='cleavage_site'
                    )

            plt.ylabel('Avg. attribution')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

            plt.tight_layout()

            plt.savefig(
                prepend_output_path(Path(
                    model_name,
                    f'{model_name}_{ANNOTATION_4STATE_LABELS[c]}_aaindex_attrs.png'
                )),
                dpi=300
            )
            plt.close()

        kingdom_attrs = torch.cat(fold_kingdom_attrs).transpose(0, 1)
        kingdom_attrs = kingdom_attrs.mean(1).view(1, len(KINGDOMS))

        kingdom_attrs_df = pd.DataFrame(kingdom_attrs.tolist(), columns=KINGDOMS)
        kingdom_attrs_df = kingdom_attrs_df.transpose()

        kingdom_attrs_df.plot.bar()

        colors = [prop['color'] for prop in iter(plt.rcParams['axes.prop_cycle'])]

        plt.bar(
            kingdom_attrs_df.index,
            kingdom_attrs_df[0],
            color=colors
        )

        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.ylabel('Avg. attribution')
        plt.legend().remove()

        plt.tight_layout()

        plt.savefig(
            prepend_output_path(Path(
                model_name,
                f'{model_name}_{ANNOTATION_4STATE_LABELS[c]}_kingdom_attrs.png'
            )),
            dpi=300
        )
        plt.close()
