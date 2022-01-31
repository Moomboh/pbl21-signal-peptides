import os
import shutil
import argparse
import torch
import time
import datetime
from collections import defaultdict
from torch.distributed.elastic.multiprocessing import start_processes
import torch.multiprocessing as mp
from tqdm import tqdm
import shutil


# Local modules
from .constants import *
from .models.index import models
from .utils import helpers
from . import checkpoint
from .train_fold import train_fold


def pbar_subprocess(pbar_queue, total):
    pbar = tqdm(total=total)

    for _ in iter(pbar_queue.get, None):
        pbar.update()


if __name__ == '__main__':

    #####
    # Parse cmdline arguments
    #####
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str,
                        help='The training dataset file. The file must be in the project-specific tsv format.')
    parser.add_argument('--checkpoint-file', type=str)
    parser.add_argument('--output-path', type=str, default='./')
    parser.add_argument('--model', type=str, default='Base')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--log-interval', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--single-fold', dest='single_fold',
                        action='store_true')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true')
    parser.add_argument('--print-log',
                        dest='print_log', action='store_true')
    parser.add_argument('--flush-cache',
                        dest='flush_cache', action='store_true')
    parser.add_argument('--partitions', nargs="+", default=PARTITIONS, type=int)
    parser.add_argument('--folds', nargs="+", default=PARTITIONS, type=int)
    parser.set_defaults(single_fold=False)
    parser.set_defaults(dry_run=False)
    parser.set_defaults(print_log=False)
    parser.set_defaults(flush_cache=False)
    args = parser.parse_args()

    #####
    # Hyperparameters
    #####
    n_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    #####
    # Options
    #####
    model_classname = args.model
    log_interval = args.log_interval
    partitions = args.partitions
    folds = args.folds
    seq_length = SEQ_LENGTH
    start_epoch = 0
    n_folds = len(folds)
    num_workers = min(args.num_workers, n_folds)
    log_dir = './.log'
    workers_log_dir = os.path.join(log_dir, 'worker')
    pbar_log_dir = os.path.join(log_dir, 'pbar')

    if args.dry_run:
        n_epochs = 1

    if args.single_fold or args.dry_run:
        n_folds = 1
        num_workers = 1

    #####
    # Check if GPU available
    #####
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    #####
    # Calculate background frequencies
    #####
    background_freqs = models[model_classname].get_background(
        args.train_file,
        partitions,
        flush_cache=args.flush_cache,
        log=args.print_log
    )

    #####
    # Train the model
    #####
    save_id = helpers.get_random_string(8)

    fold_states = defaultdict(object)
    fold_metrics = {
        'train': defaultdict(list),
        'valid': defaultdict(list),
    }

    # Maybe load checkpoint
    if args.checkpoint_file:
        fold_states, start_epoch, fold_metrics, save_id = checkpoint.load(
            args.checkpoint_file)

    start_time = time.perf_counter()

    #####
    # Start subprocesses for each fold
    #####

    # prepare log dirs
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    os.makedirs(pbar_log_dir)

    # start process for progress bar
    pbar_queue = mp.Queue()
    pbar_p_context = start_processes(
        name='pbar',
        entrypoint=pbar_subprocess,
        args={0: (pbar_queue, n_folds * n_epochs,)},
        envs={0: {}},
        log_dir=pbar_log_dir
    )

    # distribute folds across workers
    worker_folds = list(helpers.chunks(folds, num_workers))

    if args.single_fold or args.dry_run:
        worker_folds = [[1]]

    # start workers
    for w in range(0, len(worker_folds)):
        worker_log_dir = os.path.join(workers_log_dir, str(w))
        os.makedirs(worker_log_dir)

        subprocess_args = {}
        subprocess_envs = {}

        for i, fold in enumerate(worker_folds[w]):
            fold_args = [
                fold,
                partitions,
                model_classname,
                device,
                args.train_file,
                batch_size,
                background_freqs,
                learning_rate,
                start_epoch,
                n_epochs,
                seq_length,
                log_interval,
                pbar_queue,
                args.print_log,
                args.dry_run,
                args.flush_cache,
            ]

            if args.checkpoint_file:
                fold_args.append(fold_states[fold])  # checkpoint_states

            subprocess_args[i] = tuple(fold_args)
            subprocess_envs[i] = {}

        p_context = start_processes(
            name='train_fold',
            entrypoint=train_fold,
            args=subprocess_args,
            envs=subprocess_envs,
            log_dir=worker_log_dir,
        )

        results = p_context.wait()

        for i, fold in enumerate(worker_folds[w]):
            fold_metrics['train'][fold].extend(results.return_values[i][0])
            fold_metrics['valid'][fold].extend(results.return_values[i][1])
            fold_states[fold] = results.return_values[i][2]

    # close progress bar
    pbar_queue.put(None)
    pbar_p_context.wait()

    stop_time = time.perf_counter()
    print(
        f"\nTotal training time: {datetime.timedelta(seconds=stop_time - start_time)}\n")

    #####
    # Save the model
    #####
    checkpoint.save(
        output_path=args.output_path,
        start_epoch=start_epoch,
        end_epoch=start_epoch + n_epochs,
        n_epochs=n_epochs,
        model_class=model_classname,
        save_id=save_id,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fold_states=fold_states,
        fold_metrics=fold_metrics,
    )
