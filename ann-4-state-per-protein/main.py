#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""4-state per protein prediction"""

import argparse
import pandas as pd
from collections import defaultdict
from utils.OneHotEncoder import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


# pickling model
cache_dir = './.cache'

def get_model_cache_path(partitions, fold):
    return Path(cache_dir, f"_p{'_'.join([str(p) for p in partitions])}_f{fold}.joblib")

def unpickle_model(partitions, fold):
    path = get_model_cache_path(partitions, fold)
    if path.exists():
        print('Loading model from: ', path)
        return joblib.load(path)

def pickle_model(model, partitions, fold):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = get_model_cache_path(partitions, fold)
    print('Saving model to: ', path)
    joblib.dump(model, path)


def flatten_second_dim(x):
    shape = x.shape
    x = x.reshape(shape[0], shape[1] * shape[2])

    return x

# read data and create dataframe
parser = argparse.ArgumentParser()
parser.add_argument(
    'input_file',
    type=str,
    help='The training and testing dataset file. The file must be in the project-specific tsv format.'
)
parser.add_argument(
    '--output',
    type=str,
    help='The output path for metric .tsv files.'
)
parser.add_argument(
    '--partitions',
    type=int,
    nargs='+',
    default=[1, 2, 3, 4]
)
parser.add_argument(
    '--folds',
    type=int,
    nargs='+',
    default=[1, 2, 3, 4]
)
parser.add_argument(
    '--bootstraps',
    type=int,
    default=1024
)
parser.add_argument(
    '--save-preds',
    dest='save_preds',
    action='store_true'
)
parser.set_defaults(save_preds=False)
args = parser.parse_args()

Path(args.output).mkdir(exist_ok=True, parents=True)

df = pd.read_csv(
    args.input_file,
    sep='\t',
    usecols=['sequence', 'sp_type', 'partition_no']
)

# numerical constants
Z_SCORE_95_PERCENT = 1.95996

# fixed random state for reproducibility
RANDOM_STATE = 42

# initialize encoder and variables
AMINO_ACIDS = list('ARNDCQEGHILKMFPSTWYV')
aa_encoder = OneHotEncoder(categories=AMINO_ACIDS)

SP_TYPES = ['NO_SP', 'SP', 'LIPO', 'TAT']
sp_types_encoder = OneHotEncoder(categories=SP_TYPES)

def save_prediction_tsv(filename, valid_df : pd.DataFrame, pred : np.ndarray):
    df = valid_df.copy()

    df['prediction'] = np.vectorize(lambda x: SP_TYPES[x])(pred)

    df.to_csv(args.output + filename, sep='\t')


partitions = args.partitions
folds = args.folds

acc_per_fold = defaultdict(list)
mcc_per_fold = defaultdict(list)
precision_per_fold = defaultdict(list)
recall_per_fold = defaultdict(list)

acc_per_type = defaultdict(list)
mcc_per_type = defaultdict(list)
precision_per_type = defaultdict(list)
recall_per_type = defaultdict(list)

# split train and validation data + k-fold Cross-validation
for fold in folds:
    partitions_without_fold = [p for p in partitions if p != fold]

    print('\n', 'Fold ', fold)
    print('-----------------------------------------------')

    train_df = df[df['partition_no'].isin(partitions_without_fold)]
    valid_df = df[df['partition_no'] == fold]

    # onehot encoded data
    x_train = np.stack(train_df['sequence'].apply(aa_encoder.transform).to_numpy())
    x_train = flatten_second_dim(x_train)
    y_train = sp_types_encoder.transform(train_df['sp_type'].to_numpy())

    x_valid = np.stack(valid_df['sequence'].apply(aa_encoder.transform).to_numpy())
    x_valid = flatten_second_dim(x_valid)
    y_valid = sp_types_encoder.transform(valid_df['sp_type'].to_numpy())

    # ANN
    print('Training model...')

    # hyperparameters
    hidden_layer_sizes = (10,4)
    batch_size = 64

    print(f'Hidden layer sizes: {hidden_layer_sizes}')
    print(f'Batch size: {batch_size}')

    model = unpickle_model(partitions, fold)

    if model is None:
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, batch_size=batch_size, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        pickle_model(model, partitions, fold)

    # calculate training ACC and MCC
    train_pred = np.argmax(model.predict(x_train), axis=1)
    train_target = np.argmax(y_train, axis=1)
    print('Accuracy: ', accuracy_score(train_pred, train_target))
    print('MCC: ', matthews_corrcoef(train_pred, train_target))


    print('\nTesting model...')
    # calculate validation ACC and MCC

    valid_pred = np.argmax(model.predict(x_valid), axis=1)
    valid_target = np.argmax(y_valid, axis=1)

    if args.save_preds:
        save_prediction_tsv(f"fold_{fold}_predictions.tsv", valid_df, valid_pred)

    n_bootstraps_per_fold = int(np.ceil(args.bootstraps / len(folds)))

    for _ in range(n_bootstraps_per_fold):
        bootstrap_idxs = pd.Series(np.arange(0, len(valid_pred))).sample(n=len(valid_pred), replace=True)
        bootstrap_pred = valid_pred[bootstrap_idxs]
        bootstrap_target = valid_target[bootstrap_idxs]

        acc = accuracy_score(bootstrap_pred, bootstrap_target)
        mcc = matthews_corrcoef(bootstrap_pred, bootstrap_target)
        precision = precision_score(bootstrap_pred, bootstrap_target, average='macro')
        recall = recall_score(bootstrap_pred, bootstrap_target, average='macro')

        acc_per_fold[fold].append(acc)
        mcc_per_fold[fold].append(mcc)
        precision_per_fold[fold].append(precision)
        recall_per_fold[fold].append(recall)

    for i_sp_type in range(len(SP_TYPES)):
        per_type_pred = np.copy(valid_pred)
        per_type_pred[per_type_pred != i_sp_type] = len(SP_TYPES)

        per_type_target = np.copy(valid_target)
        per_type_target[per_type_target != i_sp_type] = len(SP_TYPES)

        for _ in range(n_bootstraps_per_fold):
            bootstrap_idxs = pd.Series(np.arange(0, len(per_type_pred))).sample(n=len(per_type_pred), replace=True)
            bootstrap_pred = per_type_pred[bootstrap_idxs]
            bootstrap_target = per_type_target[bootstrap_idxs]

            acc = accuracy_score(bootstrap_pred, bootstrap_target)
            mcc = matthews_corrcoef(bootstrap_pred, bootstrap_target)
            precision = precision_score(bootstrap_pred, bootstrap_target, average='macro')
            recall = recall_score(bootstrap_pred, bootstrap_target, average='macro')

            mcc_per_type[SP_TYPES[i_sp_type]].append(mcc)
            acc_per_type[SP_TYPES[i_sp_type]].append(acc)
            precision_per_type[SP_TYPES[i_sp_type]].append(precision)
            recall_per_type[SP_TYPES[i_sp_type]].append(recall)



type_metric_df = pd.DataFrame.from_dict({
    'ACC': acc_per_type,
    'MCC': mcc_per_type,
    'precision': precision_per_type,
    'recall': recall_per_type,
})

fold_metric_df = pd.DataFrame.from_dict({
    'ACC': acc_per_fold,
    'MCC': mcc_per_fold,
    'precision': precision_per_fold,
    'recall': recall_per_fold,
})

all_metric_df = pd.DataFrame.from_dict({
    'ACC': [np.concatenate([v for v in acc_per_fold.values()])],
    'MCC': [np.concatenate([v for v in mcc_per_fold.values()])],
    'precision': [np.concatenate([v for v in precision_per_fold.values()])],
    'recall': [np.concatenate([v for v in recall_per_fold.values()])],
})


for df, n in [
    (type_metric_df, len(folds)),
    (fold_metric_df, 1),
    (all_metric_df, len(folds))
]:
    for metric in df.columns:
        df[metric + '_sd'] =  df[metric].apply(lambda x: np.std(x))
        df[metric + '_ci'] =  df[metric + '_sd'].apply(lambda sd: Z_SCORE_95_PERCENT * sd / np.sqrt(n))
        df[metric] =  df[metric].apply(lambda x: np.mean(x))



type_metric_df.to_csv(args.output + "/type_metrics.tsv", sep="\t")
fold_metric_df.to_csv(args.output + "/fold_metrics.tsv", sep="\t")
all_metric_df.to_csv(args.output + "/overall_metrics.tsv", sep="\t")

# confusion matrix plot
cm = confusion_matrix(np.copy(valid_target), np.copy(valid_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=SP_TYPES)
disp.plot()
plt.savefig('utils/cm.png', dpi=240)
plt.show()