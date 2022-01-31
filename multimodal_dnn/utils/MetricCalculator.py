import torch
import torchmetrics.functional as tmf
from torchmetrics.utilities.data import to_categorical, to_onehot
from math import nan

class MetricCalculator:
    def __init__(self, num_samples, num_batches, batch_size, seq_length, class_labels, context_labels, device, log_interval):
        self.num_samples = num_samples
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.num_classes = len(class_labels)
        self.num_contexts = len(context_labels)
        self.class_labels = class_labels
        self.context_labels = context_labels

        self.log_interval = log_interval

        self.average_loss = 0

        self.total_pred = torch.zeros(
            (num_samples, seq_length, self.num_classes),
            device=device
        )

        self.total_target = torch.zeros(
            (num_samples, seq_length, self.num_classes),
            device=device
        )

        self.total_context = torch.zeros(
            (num_samples, seq_length, self.num_contexts),
            device=device
        )

    def update(self, pred, target, context, batch, loss, log=False):
        self.average_loss += loss

        batch_length = pred.shape[0]

        self.total_pred[
            batch * self.batch_size:
            (batch * self.batch_size) + batch_length
        ] = pred

        self.total_target[
            batch * self.batch_size:
            (batch * self.batch_size) + batch_length
        ] = target

        self.total_context[
            batch * self.batch_size:
            (batch * self.batch_size) + batch_length
        ] = context

        if batch % self.log_interval == 0 and log:
            loss = loss.item()
            current = batch * self.batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{self.num_samples:>5d}]")

    def calc_metrics(self):
        pred_categorical = to_categorical(
            self.total_pred, argmax_dim=2).flatten()
        pred_one_hot = to_onehot(pred_categorical, self.num_classes)

        target_categorical = to_categorical(
            self.total_target, argmax_dim=2).flatten()
        target_one_hot = to_onehot(target_categorical, self.num_classes)

        context_categorical = to_categorical(
            self.total_context, argmax_dim=2).flatten()
        context_one_hot = to_onehot(context_categorical, self.num_contexts)

        self.average_loss /= self.num_batches

        self.mcc = tmf.matthews_corrcoef(
            pred_categorical,
            target_categorical,
            self.num_classes
        ).item()

        self.confusion_matrix = tmf.confusion_matrix(
            pred_categorical,
            target_categorical,
            num_classes=self.num_classes,
        ).cpu().numpy().transpose().tolist()


        # Calculate metrics per class
        self.class_metrics = self.__calculate_class_metrics(
            pred_one_hot,
            target_one_hot
        )

        # Calculate metrics per context
        self.context_metrics = self.__calculate_context_metrics(
            pred_categorical,
            target_categorical,
            context_one_hot
        )

        return {
            'overall': {
                'average_loss': self.average_loss.item(),
                'mcc': self.mcc,
                'confusion_matrix': self.confusion_matrix
            },
            **self.class_metrics,
            **self.context_metrics,
        }

    def __calculate_class_metrics(self, pred, target):
        class_metrics = {}

        for c in range(0, self.num_classes):

            class_pred = pred.select(1, c)
            class_target = target.select(1, c)

            class_mcc = tmf.matthews_corrcoef(
                class_pred,
                class_target,
                num_classes=2
            ).item()

            class_confmat = tmf.confusion_matrix(
                class_pred,
                class_target,
                num_classes=2
            ).cpu().numpy().transpose().tolist()

            class_metrics[self.class_labels[c]] = {
                'mcc': class_mcc,
                'confusion_matrix': class_confmat
            }

        return class_metrics

    def __calculate_context_metrics(self, pred_categorical, target_categorical, context_one_hot):
        context_metrics = {}

        for c in range(0, self.num_contexts):
            context_indices = context_one_hot.select(1, c).nonzero().flatten()

            context_pred = pred_categorical.index_select(0, context_indices)
            context_target = target_categorical.index_select(
                0, context_indices)

            if len(context_pred) == 0:
                context_metrics[self.context_labels[c]] = {
                    'mcc': nan,
                    'confusion_matrix': [[nan] * self.num_classes] * self.num_classes,
                    'per_class': dict(zip(self.class_labels, [{
                        'mcc': nan,
                        'confusion_matrix': [[nan, nan], [nan, nan]]
                    }])),
                }

                return context_metrics

            context_mcc = tmf.matthews_corrcoef(
                context_pred.cpu(),
                context_target.cpu(),
                num_classes=self.num_classes
            ).item()

            context_confmat = tmf.confusion_matrix(
                context_pred,
                context_target,
                num_classes=self.num_classes
            ).cpu().numpy().transpose().tolist()

            context_per_class = self.__calculate_class_metrics(
                to_onehot(context_pred, num_classes=self.num_classes),
                to_onehot(context_target, num_classes=self.num_classes),
            )

            context_metrics[self.context_labels[c]] = {
                'mcc': context_mcc,
                'confusion_matrix': context_confmat,
                'per_class': context_per_class,
            }

        return context_metrics

    def print_metrics(self):
        print(
            f"  Overall:\tMCC: {self.mcc:>0.3f}\tAvg loss: {self.average_loss:>8f}")

        class_metrics_output = '  '
        for label in self.class_labels:
            print_label = '{:10.10}'.format(label)
            class_metrics_output += f"{print_label}\tMCC: {self.class_metrics[label]['mcc']:>0.3f}\t"
        print(class_metrics_output)

        context_metrics_output = '  '
        for label in self.context_labels:
            print_label = '{:10.10}'.format(label)
            context_metrics_output += f"{print_label}\tMCC: {self.context_metrics[label]['mcc']:>0.3f}\t"
        print(context_metrics_output)
