import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

from . import plot_helpers
from ..constants import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()


df = plot_helpers.load_metric_df(args.model)
print(df.iloc[-1].loc['valid', 'fold_0', 'annotation', 'overall', 'confusion_matrix'])
print(df.iloc[-1].loc['valid', 'fold_0', 'type', 'overall', 'confusion_matrix'])

labels = {
    'annotation': ANNOTATION_6STATE_CHARS,
    'type': ANNOTATION_4STATE_CHARS,
}

for target in ['annotation', 'type']:
    fig, ax = plt.subplots(figsize=(15,12))
    cm = pd.DataFrame(
        df.iloc[-1].loc['valid', 'fold_0', target, 'overall', 'confusion_matrix'],
        index = labels[target],
        columns = labels[target]
    )

    print(cm)

    # Add sum columns
    cm.loc["Total", :] = cm.sum(axis=0)
    cm.loc[:, "Total"] = cm.sum(axis=1)
    cm = cm.astype(int)

    # Create dataframe with relative values
    cm_ratio = cm.applymap(lambda x: x / cm.loc["Total", "Total"])

    # Absolute values
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        annot_kws={"va": "bottom"},
        linewidths=2,
        norm=matplotlib.colors.LogNorm(),
        cmap="BuPu",
        ax=ax
    )

    # Percentage values on top
    sns.heatmap(
        cm_ratio,
        linewidths=2,
        annot=True,
        annot_kws={"va": "top"},
        fmt=".2%",
        norm=matplotlib.colors.LogNorm(),
        cmap="BuPu",
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    fig.set_size_inches(9, 6)
    fig.tight_layout()

    model_name = plot_helpers.get_model_name_from_filename(args.model, include_id=True)
    fig.savefig(f"{model_name}_{target}_confusion_matrix.png", dpi=300)
    plt.close()