import os
import torch
import datetime


def load(filename):
    start_epoch = 0
    save_id = ''

    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")

        checkpoint = torch.load(filename)

        save_id = checkpoint['save_id']
        start_epoch = checkpoint['end_epoch']
        fold_states = checkpoint['fold_states']
        fold_metrics = checkpoint['fold_metrics']

        print(
            f"Loaded checkpoint '{filename}' (epoch {checkpoint['end_epoch']})")

        return fold_states, start_epoch, fold_metrics, save_id

    else:
        print(f"No checkpoint found at '{filename}'")



def save(
    output_path,
    start_epoch,
    end_epoch,
    n_epochs,
    model_class,
    save_id,
    batch_size,
    learning_rate,
    fold_states,
    fold_metrics,
):
    end_epoch = start_epoch + n_epochs

    save_name = f"{model_class}_{save_id}_" + \
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + \
        f"_e{end_epoch}" + '.pth'

    save_path = os.path.join(output_path, save_name)

    state = {
        'start_epoch': start_epoch,
        'end_epoch': end_epoch,
        'save_id': save_id,
        'hyperparams': {
            'epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        },
        'fold_states': dict(fold_states),
        'fold_metrics': dict(fold_metrics),
    }

    print('Saving model to: ' + save_path)
    torch.save(state, save_path)


def initialize(fold_state, model):
    model.load_state(fold_state)
