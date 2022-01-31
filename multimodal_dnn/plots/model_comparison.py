#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from turtle import color
import numpy as np
import pandas as pd
import seaborn
from tap import Tap
import torch
import torchmetrics.functional as tmf
import torchmetrics.utilities.data as tmdutils
from tqdm import tqdm
import matplotlib.pyplot as plt

from multimodal_dnn import transforms
from multimodal_dnn.constants import (
    ANNOT_TYPE_LABEL,
    ANNOT_SP_CHARS,
    ANNOTATION_4STATE_LABELS,
    COL_ANNOT_PRED,
    COL_ANNOTATION,
    COL_SEQUENCE,
    COL_TYPE,
    COL_TYPE_PRED,
    TYPE_NO_SP,
    TYPES,
    Z_SCORE_95_PERCENT,
)

# minimum number of sp chars in annotation for being classified as sp
SP_COUNT_THRESHOLD = 5


class Arguments(Tap):
    pred_files: list[str]  # list of tsv files with preds (space separated)
    labels: list[str]  # list of labels corresponding to preds (space separated)
    colors: list[str]  # list of labels corresponding to preds (space separated)
    num_params: list[int]  # list of parameter numbers of models (space separated)
    test_set: str  # test set csv which was used for generating preds
    output: str  # plot output file
    per_protein: bool = False  # create per protein pred plot
    per_residue: bool = False  # create per residue pred plot
    save_processed_files: bool = False  # whether to save missing types calculated from annot
    bootstraps: int = 1024  # number of bootstraps for estimating errors
    metrics_file: str = ""  # metrics file to use values from instead of calculating


def main():
    seaborn.set_theme()
    args = Arguments(underscores_to_dashes=True).parse_args()

    if args.per_protein:
        per_protein_plot(args)

    if args.per_protein:
        per_residue_plot(args)


def per_protein_plot(args: Arguments):
    if args.metrics_file:
        df = pd.read_csv(args.metrics_file, sep="\t", index_col=0)
    else:
        dfs = {
            label: load_pred_df(
                file=file,
                test_set=args.test_set,
                missing_type_from_annot=True,
                save_processed_path=args.output if args.save_processed_files else None,
            )
            for label, file in zip(args.labels, args.pred_files)
        }

        mccs = {
            label: calc_mcc_with_std(
                predictions=transforms.sp_type_to_one_hot(df[COL_TYPE].tolist()),
                targets=transforms.sp_type_to_one_hot(df[COL_TYPE_PRED].tolist()),
                bootstraps=args.bootstraps,
                num_classes=len(ANNOTATION_4STATE_LABELS),
            )
            for label, df in dfs.items()
        }

        df = pd.DataFrame.from_dict(mccs).transpose()
        df.columns = ["mcc", "mcc_sd"]
        df["mcc_ci"] = df["mcc_sd"].apply(ci95_from_std, args=(1,))

        if args.save_processed_files:
            df.to_csv(Path(args.output, f"comparison_metrics.tsv"), sep="\t")

    df["num_params"] = args.num_params
    df["color"] = args.colors

    plt.gcf().set_size_inches(8, 6)

    plt.errorbar(
        x=df["num_params"],
        y=df["mcc"],
        yerr=df["mcc_ci"],
        fmt="None",
        color="lightgray",
        capsize=6
    )

    plt.scatter(
        x=df["num_params"],
        y=df["mcc"],
        s=64,
        c=df["color"]
    )


    for label in df.index:
        plt.annotate(
            label,
            xy=(df["num_params"].loc[label], df["mcc"].loc[label] - 0.075),
            ha="right",
        )

    plt.ylim(bottom=0, top=1)
    plt.xlabel("Number of trainable parameters")
    plt.ylabel("MCC")

    plt.savefig(Path(args.output, "per_protein_comparison.png"))

    pass


def per_residue_plot(args: Arguments):
    pass


def load_pred_df(
    file, test_set, missing_type_from_annot=False, save_processed_path=None
):
    df = pd.read_csv(file, sep="\t")
    test_seqs = pd.read_csv(test_set, sep="\t", usecols=[COL_SEQUENCE])[COL_SEQUENCE]
    df["test_set_idx"] = df[COL_SEQUENCE].apply(
        lambda s: test_seqs[test_seqs == s].index[0]
    )
    df["original_idx"] = df.index
    df.sort_values(by="test_set_idx").reindex()

    if missing_type_from_annot and COL_TYPE_PRED not in df.columns:
        df[COL_TYPE_PRED] = df[COL_ANNOT_PRED].apply(sp_type_from_annot)

        if save_processed_path:
            df.to_csv(
                Path(save_processed_path, f"PROCESSED_{Path(file).name}"), sep="\t"
            )

    return df


def sp_type_from_annot(annot: str) -> str:
    sp_counts = {c: annot.count(c) for c in ANNOT_SP_CHARS}

    if sum(sp_counts.values()) > SP_COUNT_THRESHOLD:
        return ANNOT_TYPE_LABEL[max(sp_counts, key=sp_counts.get)]  # type: ignore
    else:
        return TYPE_NO_SP


def calc_mcc_with_std(
    predictions: np.ndarray, targets: np.ndarray, bootstraps: int, num_classes: int
) -> tuple[float, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred = tmdutils.to_categorical(torch.tensor(predictions, device=device))
    target = tmdutils.to_categorical(torch.tensor(targets, device=device))

    mccs = torch.empty(bootstraps, dtype=torch.float, device=device)
    ones_like_pred = torch.ones_like(pred, dtype=torch.float)

    for i in tqdm(range(bootstraps)):
        bs_idxs = torch.multinomial(
            ones_like_pred, len(ones_like_pred), replacement=True
        )
        bs_pred = pred[bs_idxs]
        bs_target = target[bs_idxs]

        mccs[i] = tmf.matthews_corrcoef(
            preds=bs_pred, target=bs_target, num_classes=num_classes
        )

    return mccs.mean().item(), mccs.std().item()  # TODO 95% CI!


def ci95_from_std(std, n):
    return float(Z_SCORE_95_PERCENT * std / np.sqrt(n))


if __name__ == "__main__":
    main()
