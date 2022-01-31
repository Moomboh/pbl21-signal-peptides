import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from . import plot_helpers


parser = argparse.ArgumentParser()
parser.add_argument('--model-files', type=str, nargs='+',
                    help='The model chekpoint saves in.pth format (seperated by spaces).')
parser.add_argument('--output-path', type=str,
                    help='The directory the plots are going to be saved in.')
parser.add_argument('--skip-epochs', type=int, default=1,
                    help='Skip first n epochs in plots (defaults to skipping one epoch).')
parser.add_argument('--exclude-folds', nargs="+", default=[],
                    help='Fold numbers to exclude (seperated by spaces).')
parser.add_argument('--exclude-classes', nargs="+", default=['overall'],
                    help='Exclude class from targets and contexts. Defaults to [\'overall\'].')
parser.add_argument('--labels', nargs="+", default=[],
                    help='Labels for the models in plots.')
parser.add_argument('--title', default='Model comparison per epoch',
                    help='Plot title.')
parser.add_argument('--xticks-step', type=int, default=10,
                    help='Space between ticks on x-axis')
parser.add_argument('--rolling-window', type=int, default=1,
                    help='Rolling window size. Defaults to 1 i.e. equivalent to no window.')
args = parser.parse_args()


def prepend_output_path(path):
    return os.path.join(args.output_path, path)


Path(args.output_path).mkdir(parents=True, exist_ok=True)

def plot_compare_target_metric(target, metric, subtitle=''):
    model_dfs = {}

    for file in args.model_files:
        model_name = plot_helpers.get_model_name_from_filename(file)

        model_df= plot_helpers.load_metric_df(file, args.exclude_folds).reorder_levels([4, 3, 0, 2, 1], axis=1)

        valid_average_metric = model_df[
            metric]['overall']['valid'][target].mean(axis=1, skipna=False).rename('valid_metric').to_frame()
        valid_average_metric_sd = model_df[
            metric]['overall']['valid'][target].std(axis=1, skipna=False).rename('valid_metric_sd').to_frame()

        model_dfs[model_name] = pd.concat([valid_average_metric, valid_average_metric_sd], axis=1)

        model_dfs[model_name]['valid_metric_rolling'] = model_dfs[model_name]['valid_metric'].rolling(args.rolling_window).mean()
        model_dfs[model_name]['valid_metric_sd_rolling'] = model_dfs[model_name]['valid_metric_sd'].rolling(args.rolling_window).mean()

    df = pd.concat(model_dfs, names=['model'], axis=1)

    df = df.reorder_levels([1, 0], axis=1)
    df = df.loc[args.skip_epochs:]


    model_names = df.columns.get_level_values(1).unique()

    for model_name in model_names:
        # error shade
        plt.fill_between(
            df.index,
            df['valid_metric_rolling'][model_name] - df['valid_metric_sd_rolling'][model_name],
            df['valid_metric_rolling'][model_name] + df['valid_metric_sd_rolling'][model_name],
            alpha=0.15
        )

        # datapoints
        plt.plot(
            df.index,
            df['valid_metric_rolling'][model_name],
            linestyle='dotted',
            linewidth=1.5,
            marker='.',
            markersize=8,
        )

    labels = args.labels

    if len(labels) != len(model_names):
        labels = model_names

    plt.legend(labels, loc='lower right')
    plt.suptitle(args.title, fontsize=16)
    plt.title(f'{subtitle}\nrolling mean window size: {args.rolling_window}', fontsize=10)
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.xticks(df.index[df.index % args.xticks_step == 0])
    plt.subplots_adjust(top=0.85)


    plt.savefig(prepend_output_path(f'compare_{target}_{metric}.png'))
    plt.clf()

plot_compare_target_metric('annotation', 'mcc', subtitle='per-residue, all 9 classes')
plot_compare_target_metric('type', 'mcc', subtitle='per-protein, all 4 classes')
