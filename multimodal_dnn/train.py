from .constants import *
from .utils.MetricCalculator import MetricCalculator


def train(dataloader, model, device, batch_size, seq_length, log_interval, print_metrics=False, dry_run=False):
    annot_metric_calculator = MetricCalculator(
        num_samples=len(dataloader.dataset),
        num_batches=len(dataloader),
        batch_size=batch_size,
        seq_length=seq_length,
        class_labels=model.class_labels(),
        context_labels=model.context_labels(),
        device=device,
        log_interval=log_interval
    )

    type_metric_calculator = MetricCalculator(
        num_samples=len(dataloader.dataset),
        num_batches=len(dataloader),
        batch_size=batch_size,
        seq_length=1,
        class_labels=ANNOTATION_4STATE_LABELS,
        context_labels=model.context_labels(),
        device=device,
        log_interval=log_interval
    )

    for batch, (X, y, context) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred_annot, y_annot, pred_type, y_type, kingdom, loss = model.train_batch(
            X, y, context)

        pred_annot, y_annot, kingdom_annot, pred_type, y_type, kingdom_type = model.before_metrics(
            pred_annot, y_annot, pred_type, y_type, kingdom)

        annot_metric_calculator.update(
            pred_annot, y_annot, kingdom_annot, batch, loss, log=print_metrics)

        type_metric_calculator.update(
            pred_type, y_type, kingdom_type, batch, loss)

        if (dry_run):
            break

    annot_metrics = annot_metric_calculator.calc_metrics()
    type_metrics = type_metric_calculator.calc_metrics()

    if print_metrics:
        print('\nTraining metrics:')
        print('Annotation:')
        annot_metric_calculator.print_metrics()
        print('Type:')
        type_metric_calculator.print_metrics()

    return {
        'annotation': annot_metrics,
        'type': type_metrics,
    }
