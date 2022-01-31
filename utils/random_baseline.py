#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates random baseline performance measures"""

import argparse
from collections import defaultdict
import dataclasses
from typing import Callable, NamedTuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import pycm
import sklearn.metrics as sklmetrics

from multimodal_dnn import transforms
from multimodal_dnn.constants import (
    KINGDOMS,
    ANNOTATION_4STATE_LABELS,
    ANNOTATION_4STATE_CHARS,
    ANNOTATION_6STATE_CHARS,
    ANNOTATION_9STATE_CHARS,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="The training dataset file. The file must be in the project-specific tsv format.",
)
parser.add_argument(
    "--output-file",
    type=str,
    default=None,
    help="The output file for the performance measure results in tsv format",
)
parser.add_argument("--all-label", type=str, default="_all_")
parser.add_argument("--epochs", type=int, default=16)
args = parser.parse_args()

# numerical constants
Z_SCORE_95_PERCENT = 1.95996

# constants from args
DATASET: str = args.dataset
OUTPUT_FILE: str = args.output_file
ALL_LABEL: str = args.all_label
EPOCHS: int = args.epochs

# constants for TSV-files
TYPE_COL = "sp_type"
ANNOTATION_COL = "annotation"
KINGDOM_COL = "kingdom"

PROTEIN_4STATE = "protein_4state"
RESIDUE_4STATE = "residue_4state"
RESIDUE_6STATE = "residue_6state"
RESIDUE_9STATE = "residue_9state"

TARGETS = [
    PROTEIN_4STATE,
    RESIDUE_4STATE,
    RESIDUE_6STATE,
    RESIDUE_9STATE,
]

TARGET_LABELS = {
    PROTEIN_4STATE: ANNOTATION_4STATE_LABELS,
    RESIDUE_4STATE: ANNOTATION_4STATE_CHARS,
    RESIDUE_6STATE: ANNOTATION_6STATE_CHARS,
    RESIDUE_9STATE: ANNOTATION_9STATE_CHARS,
}

OUTPUT_HEADER = [
    "accuracy",
    "precision",
    "recall",
    "accuracy_sd",
    "precision_sd",
    "recall_sd",
]

OUTPUT_INDEX_HEADER = ["target", "group", "label"]


class Key(NamedTuple):
    target: str
    group: str


@dataclass
class Metrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy_sd: float = 0.0
    precision_sd: float = 0.0
    recall_sd: float = 0.0

    @staticmethod
    def from_dict(data: dict[str, float]):
        metrics = Metrics()

        attrs = metrics.__dict__.keys()

        for key, val in data.items():
            if key in attrs:
                metrics.__setattr__(key, val)

        return metrics


df = pd.read_csv(DATASET, sep="\t")

# transform fanctions for extracting the target values series from the df
target_transforms: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    PROTEIN_4STATE: lambda df: df[TYPE_COL],
    RESIDUE_4STATE: lambda df: pd.Series(
        list(df[ANNOTATION_COL].apply(transforms.reduce_no_sp_annotations).str.cat())
    ),
    RESIDUE_6STATE: lambda df: pd.Series(list(df[ANNOTATION_COL].str.cat())),
    RESIDUE_9STATE: lambda df: pd.Series(
        list(df[ANNOTATION_COL].apply(transforms.expand_annotation_to_9state).str.cat())
    ),
}


# helper function for filtering by kingdom or all-label
def filter_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == ALL_LABEL:
        return df
    return df[df[KINGDOM_COL] == group]


# apply target transforms and filter_group over all targets and groups
target_vals: dict[Key, pd.Series] = {
    Key(target, group): target_transforms[target](filter_group(df, group))
    for group in KINGDOMS + [ALL_LABEL]
    for target in TARGETS
}


# helper function for calculating label freqs of target values
def label_freqs(vals: pd.Series, labels: list[str]) -> pd.Series:
    freqs = vals.value_counts(normalize=True)

    zero_labels = list(set(labels) - set(freqs.index))

    freqs = freqs.append(
        pd.Series(np.zeros(len(zero_labels)), index=zero_labels, dtype=np.float64)
    )

    return freqs


# apply label_freqs to target values
target_freqs = {
    key: label_freqs(vals, TARGET_LABELS[key.target]).reindex(TARGET_LABELS[key.target])
    for key, vals in target_vals.items()
}


# calculate confusion matrix analytically from freqs
target_cms = {
    key: vals.to_frame()
    .dot(vals.to_frame().transpose())
    .mul(len(target_vals[key]))
    .round()
    .astype(int)
    for key, vals in target_freqs.items()
}


# calculate metrics from cm
def metrics_from_cm(cm_df: pd.DataFrame) -> dict[str, Metrics]:
    metrics: dict[str, Metrics] = {}
    labels = cm_df.index
    cm = pycm.ConfusionMatrix(matrix=cm_df.to_dict())

    for label in labels:
        metrics[label] = Metrics(
            cm.class_stat["ACC"][label],
            cm.class_stat["PPV"][label],
            cm.class_stat["TPR"][label],
        )

    metrics[ALL_LABEL] = Metrics(
        cm.overall_stat["ACC Macro"],
        cm.overall_stat["PPV Macro"],
        cm.overall_stat["TPR Macro"],
    )

    return metrics


# calculate metric means from analytically derived cm
target_metrics = {key: metrics_from_cm(cm) for key, cm in target_cms.items()}


# generate cm df from pred target pairs with labels
def cm_from_preds(
    preds: pd.Series, target: pd.Series, labels: list[str]
) -> pd.DataFrame:
    cm = sklmetrics.confusion_matrix(target, preds, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    return df


# produce n=epochs random samplings with replacement from vals and calculate confusions matrix
def random_pred_cms_from_target(
    target: pd.Series, labels: list[str], epochs: int
) -> pd.DataFrame:
    preds: list[pd.DataFrame] = []

    for _ in range(epochs):
        preds.append(
            cm_from_preds(target.sample(n=len(target), replace=True), target, labels)
        )

    return preds


# map targets to random pred cm dfs
target_rnd_cms = {
    key: random_pred_cms_from_target(vals, TARGET_LABELS[key.target], EPOCHS)
    for key, vals in target_vals.items()
}

# map cms to metrics
target_rnd_metrics = {
    key: list(map(lambda cm: metrics_from_cm(cm), cms))
    for key, cms in target_rnd_cms.items()
}


# map list of per label metrics to per label sd
def metric_list_to_sd(metrics: list[dict[str, Metrics]]) -> dict[str, Metrics]:
    metric_lists: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # transpose metrics data from `type(metrics)` to fit `type(metric_lists)`
    for label_metrics in metrics:
        for label, metric in label_metrics.items():
            for metric_name, metric_value in dataclasses.asdict(metric).items():
                if "_sd" in metric_name:
                    continue
                if metric_value == "None":
                    metric_value = 0.0
                metric_lists[label][f"{metric_name}_sd"].append(metric_value)

    # calculate sd from metric lists and convert to `Metrics`
    metrics_sd = {
        key: Metrics.from_dict(
            {
                label: pd.Series(metrics).std()
                for label, metrics in label_metrics.items()
            }
        )
        for key, label_metrics in metric_lists.items()
    }

    return metrics_sd


# calculate sd estimate from rnd metrics list
target_sd = {
    key: metric_list_to_sd(metrics) for key, metrics in target_rnd_metrics.items()
}


def merge_sd_mean_metrics(mean_metrics: Metrics, sd_metrics: Metrics) -> Metrics:
    metrics = mean_metrics

    for name, val in sd_metrics.__dict__.items():
        if "_sd" in name:
            metrics.__setattr__(name, val)

    return metrics


# merge target_sd into target_metrics
target_metrics = {
    key: {
        label: merge_sd_mean_metrics(metrics, target_sd[key][label])
        for label, metrics in label_metrics.items()
    }
    for key, label_metrics in target_metrics.items()
}

# helper for injecting metrics into output_header typehinted mapper
def get_metric_output_mapper(metrics: Metrics) -> Callable[[str], float]:
    return lambda metric_name: metrics.__dict__[metric_name]


output_dict = {
    (key.target, key.group, label): list(
        map(get_metric_output_mapper(metrics), OUTPUT_HEADER)
    )
    for key, label_metrics in target_metrics.items()
    for label, metrics in label_metrics.items()
}

output_df = pd.DataFrame.from_dict(output_dict).transpose()
output_df.columns = OUTPUT_HEADER
output_df.index.set_names(OUTPUT_INDEX_HEADER)
output_df = output_df.sort_index(level=0)

# calculate 95% confidence interval from standard deviation
def ci95_from_std(std: float, n: int) -> float:
    return float(Z_SCORE_95_PERCENT * std / np.sqrt(n))


for col in output_df.columns:
    if "_sd" in col:
        for index in output_df.index:
            output_df.loc[index, col.replace("_sd", "_ci")] = ci95_from_std(
                output_df.loc[index, col], n=1
            )

if OUTPUT_FILE:
    output_df.to_csv(OUTPUT_FILE, sep="\t", index_label=OUTPUT_INDEX_HEADER)
else:
    print(output_df)

