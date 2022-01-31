from tap import Tap
import torch
import pandas as pd
import torchmetrics.utilities.data as tmutil

from multimodal_dnn import checkpoint, validate
from multimodal_dnn.constants import (
    ANNOTATION_4STATE_LABELS,
    ANNOTATION_6STATE_CHARS,
    COL_ANNOTATION,
    COL_TYPE,
    SEQ_LENGTH,
)

from . import plot_helpers


class Arguments(Tap):
    model: str  # model filename
    dataset: str  # dataset tsv file
    output: str # output tsv file
    partitions: list[int] = [0]  # partitions to make predictions for. Defaults to 0.
    batch_size: int = 64  # batch size used while iterating over dataset


def main():
    args = Arguments(underscores_to_dashes=True).parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_state = checkpoint.load(args.model)[0][0]
    model = plot_helpers.create_model(
        filename=args.model,
        dataset_file=args.dataset,
        background_partitions=args.partitions,
        device=device,
    )
    checkpoint.initialize(model_state, model)

    num_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", num_params)

    dataloader = plot_helpers.get_dataloader(
        model=model,
        dataset=args.dataset,
        partitions=args.partitions,
        batch_size=args.batch_size,
        device=device,
    )

    with torch.no_grad():
        metrics = validate.validate(
            dataloader=dataloader,
            model=model,
            device=device,
            batch_size=args.batch_size,
            seq_length=SEQ_LENGTH,
            log_interval=32,
            return_pred=True,
            print_metrics=False,
        )

    dataset_df = pd.read_csv(args.dataset, sep="\t")

    def categorical_to_annot_chars(categorical: list[int]) -> str:
        return "".join(ANNOTATION_6STATE_CHARS[i] for i in categorical)

    annot_pred = tmutil.to_categorical(metrics["annotation_pred"]["pred"], argmax_dim=2)
    annot_pred = pd.Series([categorical_to_annot_chars(seq) for seq in annot_pred])

    type_pred = tmutil.to_categorical(metrics["type_pred"]["pred"], argmax_dim=2)
    type_pred = pd.Series([ANNOTATION_4STATE_LABELS[i] for i in type_pred])

    dataset_df[f"{COL_ANNOTATION}_pred"] = annot_pred
    dataset_df[f"{COL_TYPE}_pred"] = type_pred

    dataset_df.to_csv(args.output, sep="\t")

if __name__ == "__main__":
    main()
