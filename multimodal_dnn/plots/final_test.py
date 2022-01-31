
from cmath import sqrt
from collections import defaultdict
import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path
import torchmetrics.utilities.data as tmutil
import sklearn.metrics as sklm
import numpy as np
import hashlib

from . import plot_helpers
from .. import checkpoint
from ..constants import *
from . import styles
from .. import validate
from ..utils import helpers

seaborn.set_theme()

Z_SCORE_95_PERCENT = 1.95996


parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='+')
parser.add_argument('--dataset', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--random-baseline', type=str)
parser.add_argument('--background-partitions', type=int,
                    nargs='+', default=PARTITIONS)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--bootstraps', type=int, default=4)
parser.add_argument('--flush-cache',
                    dest='flush_cache', action='store_true')
parser.set_defaults(flush_cache=False)
args = parser.parse_args()

rndb_df = pd.read_csv(args.random_baseline, sep='\t', index_col=['target', 'group', 'label'])
rndb_df = rndb_df.replace('None', 'NaN')
rndb_df = rndb_df.astype(float)


def pickle_metrics(metrics, model_filename, idstring):
    Path(METRIC_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    filename = helpers.slugify(model_filename + idstring) + '.pth'
    path = Path(METRIC_CACHE_DIR, filename)

    print(f'Saving metrics to cache: {path}')
    torch.save(metrics, path)


def unpickle_metrics(model_filename, idstring):
    filename = helpers.slugify(model_filename + idstring) + '.pth'
    path = Path(METRIC_CACHE_DIR, filename)

    if Path.exists(path) and not args.flush_cache:
        print(f'Loading metrics from cache: {path}')
        return torch.load(path)

    return {}

def get_hash(str):
    sha = hashlib.sha256()
    sha.update(str.encode())
    return sha.hexdigest()


def prepend_output_path(path):
    return os.path.join(args.output, path)


Path(args.output).mkdir(parents=True, exist_ok=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))




metrics = {}

#####
# Evaluate models
#####
for file in args.models:
    cache_id = f'_ds{args.dataset}_bg{args.background_partitions}'
    model_metrics = unpickle_metrics(file, cache_id)
    model_name = plot_helpers.get_model_name_from_filename(
        file, include_id=True)

    if model_metrics == {}:
        fold_states = checkpoint.load(file)[0]

        for fold, fold_state in fold_states.items():
            model = plot_helpers.create_model(file, args.dataset, args.background_partitions, device)
            checkpoint.initialize(fold_state, model)

            dataloader = plot_helpers.get_dataloader(
                model,
                args.dataset,
                [fold],
                args.batch_size,
                device
            )

            model_metrics[fold] = validate.validate(
                dataloader,
                model,
                device,
                args.batch_size,
                log_interval=32,
                seq_length=SEQ_LENGTH,
                return_pred=True
            )

            dataloader.dataset.pickle()

        pickle_metrics(model_metrics, file, cache_id)

    metrics[model_name] = model_metrics


def calc_class_metrics(classes, model_metrics, mode, model_name, kingdoms):
    cache_id = get_hash(
        f'{model_name}{classes}{args.bootstraps}{args.models}{kingdoms}{[str(bgp) for bgp in args.background_partitions]}'
    )
    class_metrics = unpickle_metrics(f'{model_name}_{mode}', cache_id)

    if class_metrics == {}:
        num_classes = len(classes)
        num_folds = len(model_metrics)
        bootstraps_per_fold = int(np.ceil(args.bootstraps / num_folds))

        type_precision_scores = defaultdict(list)
        type_recall_scores = defaultdict(list)
        type_mcc_scores = defaultdict(list)

        for fold, fold_metrics in model_metrics.items():
            pred : torch.Tensor = fold_metrics[f'{mode}_pred']['pred']
            target : torch.Tensor = fold_metrics[f'{mode}_pred']['target']
            context : torch.Tensor = fold_metrics[f'{mode}_pred']['context']

            # filter by kingdom
            context = context[:,0,:]
            context_indices = []

            for kingdom in kingdoms:
                context_indices.append(context.select(1, kingdom).nonzero().squeeze())
            
            context_indices = torch.cat(context_indices, dim=0)

            pred = pred.index_select(0, context_indices)
            target = target.index_select(0, context_indices)

            for bs in range(bootstraps_per_fold):
                if (bootstraps_per_fold > 1):
                    bootstrap_indices = torch.multinomial(
                        torch.ones(pred.size(dim=0)),
                        pred.size(dim=0),
                        replacement=True
                    )
                else:
                    bootstrap_indices = torch.arange(0, pred.size(dim=0))

                bootstrap_pred = pred.index_select(0, bootstrap_indices)
                bootstrap_target = target.index_select(0, bootstrap_indices)

                pred_categorical = tmutil.to_categorical(bootstrap_pred, argmax_dim=2).flatten()
                pred_one_hot = tmutil.to_onehot(pred_categorical, num_classes)

                target_categorical = tmutil.to_categorical(
                    bootstrap_target, argmax_dim=2).flatten()
                target_one_hot = tmutil.to_onehot(target_categorical, num_classes)

                # Overall
                precision = sklm.precision_score(
                    pred_categorical,
                    target_categorical,
                    labels=range(num_classes),
                    average='macro',
                    zero_division=0,
                )

                recall = sklm.recall_score(
                    pred_categorical,
                    target_categorical,
                    labels=range(num_classes),
                    average='macro',
                    zero_division=0,
                )

                mcc = sklm.matthews_corrcoef(
                    pred_categorical,
                    target_categorical,
                )

                type_precision_scores['all'].append(precision)
                type_recall_scores['all'].append(recall)
                type_mcc_scores['all'].append(mcc)

                # Per class
                for i, clss in enumerate(classes):
                    class_pred = pred_one_hot.select(1, i)
                    class_target = target_one_hot.select(1, i)

                    precision = sklm.precision_score(
                        class_pred,
                        class_target,
                        zero_division=0,
                    )

                    recall = sklm.recall_score(
                        class_pred,
                        class_target,
                        zero_division=0,
                    )

                    mcc = sklm.matthews_corrcoef(
                        class_pred,
                        class_target,
                    )

                    type_precision_scores[clss].append(precision)
                    type_recall_scores[clss].append(recall)
                    type_mcc_scores[clss].append(mcc)

        for clss in ['all'] + classes:
            precision_scores = np.array(type_precision_scores[clss])
            recall_scores = np.array(type_recall_scores[clss])
            mcc_scores = np.array(type_mcc_scores[clss])

            class_metrics[clss] = [
                precision_scores.mean(),
                precision_scores.std(),
                recall_scores.mean(),
                recall_scores.std(),
                mcc_scores.mean(),
                mcc_scores.std(),
            ]
        
        pickle_metrics(class_metrics, f'{model_name}_{mode}', cache_id)

    return class_metrics

def ci95_from_std(std, n):
    return float(Z_SCORE_95_PERCENT * std / sqrt(n))

def get_class_metrics_df(metrics, mode, labels, kingdoms):
    class_metrics = {}

    for model_name, model_metrics in metrics.items():
        model_class_metrics = calc_class_metrics(
            labels,
            model_metrics,
            mode,
            model_name,
            kingdoms,
        )

        class_metrics = {
            **class_metrics,
            **{
                (model_name, clss): metrics
                for clss, metrics in model_class_metrics.items()
            }
        }

    class_metrics_df = pd.DataFrame.from_dict(class_metrics)
    class_metrics_df = class_metrics_df.transpose()
    class_metrics_df.columns = ['precision', 'precision_sd',
                            'recall', 'recall_sd',
                            'mcc', 'mcc_sd']
    class_metrics_df.index.set_names(['model', 'type'])

    for col in class_metrics_df.columns:
        if '_sd' in col:
            for index in class_metrics_df.index:
                n = 4 if 'crosstrain' in index[0] else 1

                class_metrics_df.loc[index, col.replace('_sd', '_ci')] = ci95_from_std(
                    class_metrics_df.loc[index, col],
                    n
                )

    return class_metrics_df

def prepare_model_styles(model_names):
    model_labels = model_names.map(
        lambda x: styles.model_styles[x]['label']
    )
    model_colors = model_names.map(
        lambda x: styles.model_styles[x]['color']
    )
    model_groups = model_names.map(
        lambda x: styles.model_styles[x]['group']
    )

    # TODO: make robust for different group and model orders
    # (currently assumes groups to be ascending)
    axis_values = []
    for i in range(len(model_groups)):
        axis_values.append(i + model_groups[i])
    
    return model_labels, model_colors, axis_values

def precision_recall_plot(df, filename, xlim=[0.0, 1.0], axis_scale=1.0, hide_labels=False, suptitle=None, 
                          pre_rndb=0.0, rec_rndb=0.0, pre_rndb_err=0.0, rec_rndb_err=0.0, rndb_xlim=False, show_rndb=True):
    return
    print(f'Creating precision-recall plot... \t\t {filename}')
    model_names = df.index.get_level_values(0)
    model_labels, model_colors, axis_values = prepare_model_styles(model_names)
    axis_values = [v*axis_scale for v in axis_values]

    if np.isnan(pre_rndb):
        pre_rndb = 0.0
    if np.isnan(rec_rndb):
        rec_rndb = 0.0

    if rndb_xlim:
        pre_rndb_floor = np.floor(pre_rndb*10 - 1) / 10
        rec_rndb_floor = np.floor(rec_rndb*10 - 1) / 10
        rndb_floor_min = np.min([pre_rndb_floor, rec_rndb_floor])
        rndb_floor_min = max(rndb_floor_min, 0.0)

        xlim[0] = rndb_floor_min

    fig, axes = plt.subplots(ncols=2, sharey=True)
    plt.tight_layout()

    axes[0].barh(
        axis_values,
        df['precision'],
        tick_label=model_labels,
        color=model_colors,
        height=1*axis_scale,
        xerr=df['precision_ci'],
        error_kw={
            'capsize': 10.0*axis_scale,
            'capthick': 2.0
        },
        align='center'
    )
    axes[0].axvline(
        x=pre_rndb,
        color='black',
        linestyle='--',
        linewidth=0.75,
    )

    if show_rndb:
        axes[0].axvspan(pre_rndb - pre_rndb_err, pre_rndb + pre_rndb_err, alpha=0.2, color='black')
        axes[0].text(pre_rndb + (0.05 * (xlim[1] - xlim[0])), 0, 'random baseline (shade ± 95% CI)', color='black', rotation=90, fontsize=12)

    axes[0].set_xlabel('Precision', fontsize=18)
    axes[0].set_xlim(xlim)
    axes[0].invert_xaxis()
    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='x', labelsize=16)


    for x, y, precision in zip(axis_values, df['precision'] - df['precision_ci'], df['precision']):
        x = x - 0.1 * axis_scale
        y = y - 0.005*y
        y = max(0.05, y)
        axes[0].text(y, x, f'{round(precision, 3):.3f}', color='white', fontsize=18, fontweight='bold')
    
    if hide_labels:
        axes[0].tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    axes[1].barh(
        axis_values,
        df['recall'],
        tick_label=model_labels,
        color=model_colors,
        height=1*axis_scale,
        xerr=df['recall_ci'],
        error_kw={
            'capsize': 10.0*axis_scale,
            'capthick': 2.0
        },
        align='center'
    )
    axes[1].axvline(
        x=rec_rndb,
        color='black',
        linestyle='--',
        linewidth=0.75,
    )
    #  
    if show_rndb:
        axes[1].axvspan(rec_rndb + rec_rndb_err, rec_rndb - rec_rndb_err, alpha=0.2, color='black')
        axes[1].text(rec_rndb + (0.015 * (xlim[1] - xlim[0])), 0, 'random baseline (shade ± 95% CI)', color='black', rotation=90, fontsize=12)
    axes[1].set_xlabel('Recall', fontsize=18)
    axes[1].set_xlim(xlim)
    axes[1].tick_params(axis='x', labelsize=16)

    for x, y, precision in zip(axis_values, df['recall'] - df['recall_ci'], df['recall']):
        leftshift_factor = 0.23 if hide_labels else 0.22
        leftshift_factor *= (xlim[1] - xlim[0])
        y = y - leftshift_factor - 0.005 * y
        y = max(0.05, y)
        x = x - 0.1 * axis_scale
        axes[1].text(y, x, f'{round(precision, 3):.3f}', color='white', fontsize=18, fontweight='bold')

    if suptitle:
        plt.suptitle(suptitle, fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    plt.savefig(
        prepend_output_path(filename),
        dpi=300
    )
    plt.close()

def mcc_plot(df, filename, ylim=[0.0, 1.0], axis_scale=1.0, hide_labels=False, suptitle=None):
    return
    print(f'Creating mcc plot...\t\t\t\t {filename}')
    model_names = df.index.get_level_values(0)
    model_labels, model_colors, axis_values = prepare_model_styles(model_names)
    axis_values = [v*axis_scale for v in axis_values]

    plt.bar(
        axis_values,
        df['mcc'],
        tick_label=model_labels,
        color=model_colors,
        width=1*axis_scale,
        yerr=df['mcc_ci'],
        error_kw={
            'capsize': 10.0*axis_scale,
            'capthick': 2.0
        },
        align='center'
    )

    for x, y, mcc in zip(axis_values, df['mcc'] - df['mcc_ci'], df['mcc']):
        downshift_factor = 0.041 * (ylim[1] - ylim[0])
        plt.gca().text(x + 0.4*axis_scale, y - downshift_factor - y*0.005, f'{round(mcc, 3):.3f}', color='white', fontsize=18, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tick_params(axis='y', labelsize=16)

    if hide_labels:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    if suptitle:
        plt.suptitle(suptitle, fontsize=20)

    axis = plt.gca()
    axis.invert_xaxis()
    axis.set_ylabel('MCC')
    axis.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(
        prepend_output_path(filename),
        dpi=300
    )
    plt.close()

############################
# Type metrics
############################
type_metrics_df = get_class_metrics_df(metrics, 'type', ANNOTATION_4STATE_LABELS, kingdoms=list(range(len(KINGDOMS))))

all_type_df = type_metrics_df[
    type_metrics_df.index.get_level_values(1) == 'all'
]


precision_recall_plot(
    all_type_df,
    'type_all_precision_recall.png',
    [0.6, 1.0],
    hide_labels=True,
    suptitle='4-state per-protein performance (precision, recall)',
    pre_rndb=rndb_df['precision'].loc['protein_4state', 'all', 'all'],
    rec_rndb=rndb_df['recall'].loc['protein_4state', 'all', 'all'],
    pre_rndb_err=rndb_df['precision_ci'].loc['protein_4state', 'all', 'all'],
    rec_rndb_err=rndb_df['recall_ci'].loc['protein_4state', 'all', 'all'],
    rndb_xlim=False,
    show_rndb=False,
)

precision_recall_plot(
    all_type_df,
    'type_all_precision_recall_rndb.png',
    [0.6, 1.0],
    hide_labels=True,
    suptitle='4-state per-protein performance (precision, recall)',
    pre_rndb=rndb_df['precision'].loc['protein_4state', 'all', 'all'],
    rec_rndb=rndb_df['recall'].loc['protein_4state', 'all', 'all'],
    pre_rndb_err=rndb_df['precision_ci'].loc['protein_4state', 'all', 'all'],
    rec_rndb_err=rndb_df['recall_ci'].loc['protein_4state', 'all', 'all'],
    rndb_xlim=True,
    show_rndb=False,
)

mcc_plot(
    all_type_df,
    'type_all_mcc.png',
    [0.8, 1.0],
    hide_labels=True,
    suptitle='4-state per-protein performance (MCC)',
)

for clss in ANNOTATION_4STATE_LABELS:
    class_type_df = type_metrics_df[
        type_metrics_df.index.get_level_values(1) == clss
    ]

    class_label = styles.class_styles[clss]['label']

    precision_recall_plot(
        class_type_df,
        f'type_{clss}_precision_recall.png',
        [0.8, 1.0],
        axis_scale=0.5,
        hide_labels=True,
        suptitle=class_label,
        pre_rndb=rndb_df['precision'].loc['protein_4state', 'all', clss],
        rec_rndb=rndb_df['recall'].loc['protein_4state', 'all', clss],
        pre_rndb_err=rndb_df['precision_ci'].loc['protein_4state', 'all', clss],
        rec_rndb_err=rndb_df['recall_ci'].loc['protein_4state', 'all', clss],
        rndb_xlim=False,
    )

    precision_recall_plot(
        class_type_df,
        f'type_{clss}_precision_recall_rndb.png',
        [0.8, 1.0],
        axis_scale=0.5,
        hide_labels=True,
        suptitle=class_label,
        pre_rndb=rndb_df['precision'].loc['protein_4state', 'all', clss],
        rec_rndb=rndb_df['recall'].loc['protein_4state', 'all', clss],
        pre_rndb_err=rndb_df['precision_ci'].loc['protein_4state', 'all', clss],
        rec_rndb_err=rndb_df['recall_ci'].loc['protein_4state', 'all', clss],
        rndb_xlim=True,
    )

    mcc_plot(
        class_type_df,
        f'type_{clss}_mcc.png',
        [0.8, 1.0],
        axis_scale=0.5,
        hide_labels=True,
        suptitle=class_label,
    )

for kingdom in range(len(KINGDOMS)):
    kingdom_type_metrics_df = get_class_metrics_df(metrics, 'type', ANNOTATION_4STATE_LABELS, [kingdom])

    kingdom_all_type_df = kingdom_type_metrics_df[
        kingdom_type_metrics_df.index.get_level_values(1) == 'all'
    ]

    Path(prepend_output_path(KINGDOMS[kingdom])).mkdir(parents=True, exist_ok=True)

    precision_recall_plot(
        kingdom_all_type_df,
        f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_type_all_precision_recall.png',
        [0.0, 1.0],
        hide_labels=True,
        suptitle=f'{KINGDOMS[kingdom]} all 4 classes',
        pre_rndb=rndb_df['precision'].loc['protein_4state', KINGDOMS[kingdom], 'all'],
        rec_rndb=rndb_df['recall'].loc['protein_4state', KINGDOMS[kingdom], 'all'],
        pre_rndb_err=rndb_df['precision_ci'].loc['protein_4state', KINGDOMS[kingdom], 'all'],
        rec_rndb_err=rndb_df['recall_ci'].loc['protein_4state', KINGDOMS[kingdom], 'all'],
        rndb_xlim=True,
    )

    mcc_plot(
        kingdom_all_type_df,
        f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_type_all_mcc.png',
        [0.0, 1.0],
        hide_labels=True,
        suptitle=f'{KINGDOMS[kingdom]} all 4 classes',
    )

    for clss in ANNOTATION_4STATE_LABELS:
        kingdom_class_type_df = kingdom_type_metrics_df[
            kingdom_type_metrics_df.index.get_level_values(1) == clss
        ]

        class_label = styles.class_styles[clss]['label']

        precision_recall_plot(
            kingdom_class_type_df,
            f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_type_{clss}_precision_recall.png',
            [0.0, 1.0],
            axis_scale=0.5,
            hide_labels=True,
            suptitle=f'{KINGDOMS[kingdom]} {class_label}',
            pre_rndb=rndb_df['precision'].loc['protein_4state', KINGDOMS[kingdom], clss],
            rec_rndb=rndb_df['recall'].loc['protein_4state', KINGDOMS[kingdom], clss],
            rndb_xlim=True,
        )

        mcc_plot(
            kingdom_class_type_df,
            f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_type_{clss}_mcc.png',
            [0.0, 1.0],
            axis_scale=0.5,
            hide_labels=True,
            suptitle=f'{KINGDOMS[kingdom]} {class_label}',
        )

############################
# Annotation metrics
############################
annot_metrics_df = get_class_metrics_df(metrics, 'annotation', ANNOTATION_6STATE_LABELS, kingdoms=list(range(len(KINGDOMS))))

all_annot_df = annot_metrics_df[
    annot_metrics_df.index.get_level_values(1) == 'all'
]

precision_recall_plot(
    all_annot_df,
    'annot_all_precision_recall.png',
    [0.6, 1.0],
    suptitle='6-state per-residue performance (precision, recall)',
    hide_labels=True,
    pre_rndb=rndb_df['precision'].loc['residue_6state', 'all', 'all'],
    rec_rndb=rndb_df['recall'].loc['residue_6state', 'all', 'all'],
    pre_rndb_err=rndb_df['precision_ci'].loc['residue_6state', 'all', 'all'],
    rec_rndb_err=rndb_df['recall_ci'].loc['residue_6state', 'all', 'all'],
    rndb_xlim=False,
    show_rndb=False,
)

precision_recall_plot(
    all_annot_df,
    'annot_all_precision_recall_rndb.png',
    [0.6, 1.0],
    suptitle='6-state per-residue performance (precision, recall)',
    hide_labels=True,
    pre_rndb=rndb_df['precision'].loc['residue_6state', 'all', 'all'],
    rec_rndb=rndb_df['recall'].loc['residue_6state', 'all', 'all'],
    pre_rndb_err=rndb_df['precision_ci'].loc['residue_6state', 'all', 'all'],
    rec_rndb_err=rndb_df['recall_ci'].loc['residue_6state', 'all', 'all'],
    rndb_xlim=True,
    show_rndb=False,
)

mcc_plot(
    all_annot_df,
    'annot_all_mcc.png',
    [0.8, 1.0],
    suptitle='6-state per-residue performance (MCC)',
    hide_labels=True,
)

for clss in ANNOTATION_6STATE_LABELS:
    class_annot_df = annot_metrics_df[
        annot_metrics_df.index.get_level_values(1) == clss
    ]

    class_label = styles.class_styles[clss]['label']
    class_char = ANNOTATION_6STATE_CHARS[ANNOTATION_6STATE_LABELS.index(clss)]

    precision_recall_plot(
        class_annot_df,
        f'annot_{clss}_precision_recall.png',
        [0.0, 1.0],
        axis_scale=0.5,
        hide_labels=True,
        suptitle=class_label,
        pre_rndb=rndb_df['precision'].loc['residue_6state', 'all', class_char],
        rec_rndb=rndb_df['recall'].loc['residue_6state', 'all', class_char],
        pre_rndb_err=rndb_df['precision_ci'].loc['residue_6state', 'all', class_char],
        rec_rndb_err=rndb_df['recall_ci'].loc['residue_6state', 'all', class_char],
        rndb_xlim=True,
    )

    mcc_plot(
        class_annot_df,
        f'annot_{clss}_mcc.png',
        [0.4, 1.0],
        axis_scale=0.5,
        hide_labels=True,
        suptitle=class_label,
    )

for kingdom in range(len(KINGDOMS)):
    kingdom_annot_metrics_df = get_class_metrics_df(metrics, 'annotation', ANNOTATION_6STATE_LABELS, [kingdom])

    kingdom_all_annot_df = kingdom_annot_metrics_df[
        kingdom_annot_metrics_df.index.get_level_values(1) == 'all'
    ]

    Path(prepend_output_path(KINGDOMS[kingdom])).mkdir(parents=True, exist_ok=True)

    precision_recall_plot(
        kingdom_all_annot_df,
        f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_annot_all_precision_recall.png',
        [0.0, 1.0],
        hide_labels=True,
        pre_rndb=rndb_df['precision'].loc['residue_6state', KINGDOMS[kingdom], 'all'],
        rec_rndb=rndb_df['recall'].loc['residue_6state', KINGDOMS[kingdom], 'all'],
        rndb_xlim=True,
    )

    mcc_plot(
        kingdom_all_annot_df,
        f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_annot_all_mcc.png',
        [0.0, 1.0],
        hide_labels=True,
    )

    for clss in ANNOTATION_6STATE_LABELS:
        kingdom_class_annot_df = kingdom_annot_metrics_df[
            kingdom_annot_metrics_df.index.get_level_values(1) == clss
        ]

        class_label = styles.class_styles[clss]['label']

        precision_recall_plot(
            kingdom_class_annot_df,
            f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_annot_{clss}_precision_recall.png',
            [0.0, 1.0],
            axis_scale=0.5,
            hide_labels=True,
            suptitle=class_label,
            pre_rndb=rndb_df['precision'].loc['residue_6state', KINGDOMS[kingdom], class_char],
            rec_rndb=rndb_df['recall'].loc['residue_6state', KINGDOMS[kingdom], class_char],
            pre_rndb_err=rndb_df['precision_ci'].loc['residue_6state', KINGDOMS[kingdom], class_char],
            rec_rndb_err=rndb_df['recall_ci'].loc['residue_6state', KINGDOMS[kingdom], class_char],
            rndb_xlim=True,
        )

        mcc_plot(
            kingdom_class_annot_df,
            f'{KINGDOMS[kingdom]}/{KINGDOMS[kingdom]}_annot_{clss}_mcc.png',
            [0.0, 1.0],
            axis_scale=0.5,
            hide_labels=True,
            suptitle=class_label,
        )


############################
# Save all metrics to TSV 
############################
dfs = [
    pd.concat(
        [pd.concat([type_metrics_df], keys=[ALL_LABEL])],
        keys=[PROTEIN_4STATE_LABEL]
    ),
    pd.concat(
        [pd.concat([annot_metrics_df], keys=[ALL_LABEL])],
        keys=[RESIDUE_6STATE_LABEL]
    ),
]


dfs.extend(
    pd.concat(
        [pd.concat([
            get_class_metrics_df(
                metrics,
                'type',
                ANNOTATION_4STATE_LABELS,
                [kingdom]
            )
        ], keys=[KINGDOMS[kingdom]])],
        keys=[PROTEIN_4STATE_LABEL]
    )
    for kingdom in range(len(KINGDOMS))
)

dfs.extend(
    pd.concat(
        [pd.concat([
           get_class_metrics_df(
                metrics,
                'annotation',
                ANNOTATION_6STATE_LABELS, # TODO: should be ..._CHARS 
                [kingdom]
            )
        ], keys=[KINGDOMS[kingdom]])],
        keys=[RESIDUE_6STATE_LABEL]
    )
    for kingdom in range(len(KINGDOMS))
)

for df in dfs:
    df.index.set_names(
        [
            METRICS_IDX_HEADER_TARGET,
            METRICS_IDX_HEADER_GROUP,
            METRICS_IDX_HEADER_MODEL,
            METRICS_IDX_HEADER_LABEL
        ],
        inplace=True
    )

df = pd.concat(dfs).reorder_levels([2, 0, 1, 3])

df.to_csv(prepend_output_path("metrics.tsv"), sep="\t")



