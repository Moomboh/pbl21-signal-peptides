from torch.utils.data import DataLoader

from .dataset import SignalPeptideDataset
from .models.index import models
from .train import train
from .validate import validate
from . import checkpoint


def train_fold(
    fold,
    partitions,
    model_classname,
    device,
    train_file,
    batch_size,
    background_freqs,
    learning_rate,
    start_epoch,
    n_epochs,
    seq_length,
    log_interval,
    pbar_queue,
    print_log,
    dry_run=False,
    flush_cache=False,
    checkpoint_state=None,
):
    if print_log:
        print(f"\n\nFold {fold}\n===============================")

    # Instantiate model
    model = models[model_classname](
        background_freqs,
        learning_rate,
        device
    )

    # Load data
    train_data = SignalPeptideDataset(
        train_file,
        partitions=[p for p in partitions if p != fold],
        model=model,
        device=device,
    )
    train_data.unpickle(flush_cache, log=print_log)

    valid_data = SignalPeptideDataset(
        train_file,
        partitions=[fold],
        model=model,
        device=device,
    )
    valid_data.unpickle(flush_cache, log=print_log)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
    )

    # Maybe load state from checkpoint
    if checkpoint_state:
        checkpoint.initialize(
            fold_state=checkpoint_state,
            model=model
        )

    if print_log:
        print(f'Initialized model for fold {fold}')

    train_metrics = []
    valid_metrics = []

    # Epoch loop
    for epoch in range(start_epoch, start_epoch + n_epochs):
        if print_log:
            print(f"\nEpoch {epoch+1}\n-------------------------------")

        train_metric = train(
            dataloader=train_dataloader,
            model=model,
            device=device,
            batch_size=batch_size,
            seq_length=seq_length,
            log_interval=log_interval,
            print_metrics=print_log,
            dry_run=dry_run
        )

        valid_metric = validate(
            dataloader=valid_dataloader,
            model=model,
            device=device,
            batch_size=batch_size,
            seq_length=seq_length,
            log_interval=log_interval,
            print_metrics=print_log,
            dry_run=dry_run
        )

        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)

        pbar_queue.put(1)

    fold_state = model.get_state()

    train_data.pickle(log=print_log)
    valid_data.pickle(log=print_log)

    return train_metrics, valid_metrics, fold_state
