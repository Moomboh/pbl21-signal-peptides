import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..constants import *
from . import plot_helpers


parser = argparse.ArgumentParser()
parser.add_argument('--model-file', type=str,
                    help='The model chekpoint save in.pth format.')
parser.add_argument('--output-path', type=str,
                    help='The directory the plots are going to be saved in.')
parser.add_argument('--skip-epochs', type=int, default=1,
                    help='Skip first n epochs in plots (defaults to skipping one epoch).')
parser.add_argument('--exclude-folds', nargs="+", default=[],
                    help='Fold numbers to exclude (seperated by spaces).')
parser.add_argument('--exclude-classes', nargs="+", default=['overall'],
                    help='Exclude class from targets and contexts. Defaults to [\'overall\'].')
parser.add_argument('--print-df',
                    dest='print_df', action='store_true')
parser.add_argument('--max-text-pos', nargs=2, default=[0.9, 0.6], type=float,
                    help='Position of max value highlight textbox.')
parser.add_argument('--xticks-step', type=int, default=10,
                    help='Space between ticks on x-axis')
parser.add_argument('--rolling-window', type=int, default=1,
                    help='Rolling window size. Defaults to 1 i.e. equivalent to no window.')
parser.set_defaults(print_df=False)
args = parser.parse_args()


def prepend_output_path(path):
    return os.path.join(args.output_path, path)


Path(args.output_path).mkdir(parents=True, exist_ok=True)


df = plot_helpers.load_metric_df(args.model_file, args.exclude_folds)

if args.print_df:
    print(df)

df.to_html(prepend_output_path('datatable.html'))

folds = df.columns.get_level_values(1).unique()
classes = df.columns.get_level_values(3).unique()

df = df.reorder_levels([4, 3, 0, 2, 1], axis=1)

df = df[args.skip_epochs:]


def plt_dots_with_error_shade(x, y, error):
    plt.plot(
        x,
        y,
        linestyle='dotted',
        linewidth=1.5,
        marker='.',
        markersize=8,
    )

    plt.fill_between(
        x,
        y - error,
        y + error,
        alpha=0.15
    )


def metric_per_epoch_plots(target, metric_name, class_labels, context_labels):
    epochs = df.index
    train_average = df[metric_name]['overall']['train'][target].mean(axis=1, skipna=False).rolling(args.rolling_window).mean()
    train_average_sd = df[metric_name]['overall']['train'][target].std(axis=1, skipna=False).rolling(args.rolling_window).mean()
    valid_average = df[metric_name]['overall']['valid'][target].mean(axis=1, skipna=False).rolling(args.rolling_window).mean()
    valid_average_sd = df[metric_name]['overall']['valid'][target].std(axis=1, skipna=False).rolling(args.rolling_window).mean()

    plt_dots_with_error_shade(epochs, train_average, train_average_sd)
    plt_dots_with_error_shade(epochs, valid_average, valid_average_sd)

    plot_helpers.annotate_max(
        epochs,
        valid_average,
        plt.gca(),
        'max MCC',
        xtext=args.max_text_pos[0],
        ytext=args.max_text_pos[1],
    )

    plt.suptitle(f"{metric_name} average across folds", fontsize=16)
    plt.title(f'rolling mean window size: {args.rolling_window}', fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.xticks(df.index[df.index % args.xticks_step == 0])
    plt.legend(['training', '∓σ', 'validation', '∓σ'])

    plt.savefig(prepend_output_path(f"{target}_average_{metric_name}.png"))
    plt.clf()

    for fold in folds:
        metric_per_fold = df[metric_name]['overall']['valid'][target][fold].rolling(args.rolling_window).mean()

        plt.plot(
            epochs,
            metric_per_fold,
            linestyle='dotted',
            linewidth=1.5,
            marker='.',
            markersize=8,
        )

    plt.suptitle(f'{metric_name} per fold', fontsize=16)
    plt.title(f'rolling mean window size: {args.rolling_window}', fontsize=10)
    plt.savefig(prepend_output_path(f"{target}_{metric_name}_per_fold.png"))
    plt.clf()

    for cls in class_labels:
        metric_column = df[metric_name][cls]['valid'].mean(axis=1, skipna=False).rolling(args.rolling_window).mean()
        metric_column_sd = df[metric_name][cls]['valid'].std(axis=1, skipna=False).rolling(args.rolling_window).mean()

        plt_dots_with_error_shade(epochs, metric_column, metric_column_sd)


    plt.suptitle(f"{metric_name} per class", fontsize=16)
    plt.title(f'rolling mean window size: {args.rolling_window}', fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.xticks(df.index[df.index % args.xticks_step == 0])

    class_legend = sum([[label, '∓σ'] for label in class_labels], [])
    plt.legend(class_legend)

    plt.savefig(prepend_output_path(f"{target}_{metric_name}_per_type.png"))
    plt.clf()

    for ctx in context_labels:
        metric_column = df[metric_name][ctx]['valid'].mean(axis=1, skipna=False).rolling(args.rolling_window).mean()
        metric_column_sd = df[metric_name][ctx]['valid'].std(axis=1, skipna=False).rolling(args.rolling_window).mean()

        plt_dots_with_error_shade(epochs, metric_column, metric_column_sd)

    plt.suptitle(f"{metric_name} per kingdom", fontsize=16)
    plt.title(f'rolling mean window size: {args.rolling_window}', fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.xticks(df.index[df.index % args.xticks_step == 0])

    context_legend = sum([[label, '∓σ'] for label in context_labels], [])
    plt.legend(context_legend)

    plt.savefig(prepend_output_path(f"{target}_{metric_name}_per_kingdom.png"))
    plt.clf()


metric_per_epoch_plots('annotation', 'mcc', ANNOTATION_6STATE_LABELS, KINGDOMS)
metric_per_epoch_plots('type', 'mcc', ANNOTATION_4STATE_LABELS, KINGDOMS)
