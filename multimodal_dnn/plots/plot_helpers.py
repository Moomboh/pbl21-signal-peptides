import torch
import re
import pandas as pd
import numpy as np
from ..models.index import models
from torch.utils.data import DataLoader
from ..dataset import SignalPeptideDataset


def load_metric_df(model_file, exclude_folds=[]):
    checkpoint = torch.load(model_file)

    metrics = checkpoint['fold_metrics']

    # Reshape data
    metrics = {
        (stage, f"fold_{fold}", target, metric_class, metric_name):
        [metric[target][metric_class][metric_name] for metric in metrics[stage][fold]]
        for stage in metrics.keys()
        for fold in metrics[stage].keys()
        for target in metrics[stage][fold][0].keys()
        for metric_class in metrics[stage][fold][0][target].keys()
        for metric_name in metrics[stage][fold][0][target][metric_class].keys()
        if fold not in exclude_folds
    }

    df = pd.DataFrame.from_dict(metrics).round(3)
    df.index += 1
    df.index.name = 'epoch'

    return df


# Helper function for annotating the maximum
def annotate_max(x, y, ax, max_label, xtext=0.9, ytext=0.2):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = f"{max_label}={ymax:.3f}, epoch={xmax}"

    bbox_props = dict(
        boxstyle="square,pad=0.3",
        fc="w",
        ec="k",
        lw=0.72
    )

    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=60"
    )

    kw = dict(
        xycoords='data',
        textcoords="axes fraction",
        arrowprops=arrowprops,
        bbox=bbox_props,
        ha="right",
        va="top"
    )

    ax.annotate(text, xy=(xmax, ymax), xytext=(xtext, ytext), **kw)

def get_model_name_from_filename(filename, include_id=False):
    filename_parts = re.split('(\\\\|/)', filename)[-1].split('_')

    model_name = filename_parts[0]

    if filename_parts[1].isdigit():
        model_name += '_' + filename_parts[1]

        if include_id:
            model_name += '_' + filename_parts[2]

    elif include_id:
        model_name += '_' + filename_parts[1]

    return model_name

def create_model(filename, dataset_file, background_partitions, device):
    model_classname = get_model_name_from_filename(filename)
    background_freqs = models[model_classname].get_background(
        dataset_file,
        background_partitions
    )

    model = models[model_classname](
        background_freqs,
        1e-3,
        device
    )

    return model


def get_dataloader(model, dataset, partitions, batch_size, device):
    dataset = SignalPeptideDataset(
        dataset,
        partitions=partitions,
        model=model,
        device=device,
    )
    dataset.unpickle()

    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return test_dataloader