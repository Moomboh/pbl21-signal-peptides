import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from .. import transforms
from ..constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--labelsize', type=int, default=24)
args = parser.parse_args()


blosum62_df = pd.DataFrame(transforms.NORMALIZED_BLOSUM62)
blosum62_df.columns = AMINO_ACIDS
blosum62_df.index = AMINO_ACIDS

print(blosum62_df.round(3))
print('Mean: ', round(blosum62_df.values.flatten().mean(), 3))
print('Std: ', round(blosum62_df.values.flatten().std(), 3))
plt.hist(blosum62_df.values.flatten())
plt.savefig('blosum62_hist.png')

grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
fig, (ax, cbar_ax) = plt.subplots(ncols=2, figsize=(15,12), gridspec_kw=grid_kws)
sns.heatmap(
    blosum62_df,
    annot=True,
    linewidths=2,
    cmap="BuPu",
    ax=ax,
    cbar_ax=cbar_ax,
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=args.labelsize)
cbar_ax.tick_params(labelsize=args.labelsize)

plt.savefig('blosum62_scaled.png')
plt.clf()
plt.close()

aaindex_ids = sorted([
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

aaindex = (transforms.FeatureEncoder(aaindex_ids=aaindex_ids)).aaindex

aaindex_df = pd.DataFrame(aaindex)
aaindex_df.columns = aaindex_ids
aaindex_df.index = AMINO_ACIDS

aaindex_df = aaindex_df.transpose()

grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
fig, (ax, cbar_ax) = plt.subplots(ncols=2, figsize=(20,12), gridspec_kw=grid_kws)
sns.heatmap(
    aaindex_df,
    annot=True,
    linewidths=2,
    cmap="BuPu",
    ax=ax,
    cbar_ax=cbar_ax,
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=args.labelsize)
cbar_ax.tick_params(labelsize=args.labelsize)

plt.tight_layout()
plt.savefig('aaindex_scaled.png')