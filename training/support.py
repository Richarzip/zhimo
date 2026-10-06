"""Training argument, data-loading and metric helpers with lazy dependencies."""

import argparse


def positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def batch_size_int(value):
    number = positive_int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("must be at least 2 because the model uses BatchNorm")
    return number


def build_argument_parser(config):
    parser = argparse.ArgumentParser(description="Train Calligrapher Classifier")
    for key, val in config.items():
        if key in ("author_map", "author_alias"):
            continue
        if key == "max_samples":
            parser.add_argument("--max_samples", type=positive_int, default=val)
        elif key == "batch_size":
            parser.add_argument("--batch_size", type=batch_size_int, default=val)
        elif isinstance(val, bool):
            parser.add_argument(f"--{key}", action="store_true", default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val) if val is not None else str, default=val)
    parser.add_argument("--resume", type=str, default=None)
    return parser


def validate_max_samples(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError("max_samples must be a positive integer or None")


def create_data_loaders(train_dataset, val_dataset, batch_size, *, num_workers=4, pin_memory=True):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("batch_size must be at least 2 because the model uses BatchNorm")
    if len(train_dataset) < 2:
        raise ValueError("Training dataset must contain at least 2 samples for BatchNorm")
    if not len(val_dataset):
        raise ValueError("Validation dataset must contain at least 1 sample")
    from torch.utils.data import DataLoader
    # Keep useful partial batches; discard only a singleton that BatchNorm cannot train on.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=len(train_dataset) % batch_size == 1,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def classification_metrics(all_labels, all_preds, id_to_label):
    from sklearn.metrics import classification_report
    labels = sorted(id_to_label)
    return classification_report(
        all_labels, all_preds, labels=labels,
        target_names=[id_to_label[label] for label in labels],
        digits=4, output_dict=True, zero_division=0,
    )


def normalized_confusion_matrix(all_labels, all_preds, labels):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    labels = list(labels)
    if len(all_labels) == 0 and len(all_preds) == 0:
        return np.zeros((len(labels), len(labels)), dtype=float)
    matrix = confusion_matrix(all_labels, all_preds, labels=labels).astype(float)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)
