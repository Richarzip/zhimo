"""Checkpoint selection and metadata, independent of model dependencies."""

from pathlib import Path


def restore_best_accuracy(checkpoint, best_path, *, load):
    """Recover the historical best, including checkpoints from older training runs."""
    best_acc = checkpoint.get("best_acc", 0.0)
    if Path(best_path).is_file():
        historical_best = load(best_path)
        # A shared output directory may contain a different model or label mapping.
        keys = ("backbone", "num_classes", "id_to_label")
        if all(historical_best.get(key) == checkpoint.get(key) for key in keys):
            best_acc = max(best_acc, historical_best.get("best_acc", 0.0))
    return best_acc


def save_epoch_checkpoint(checkpoint, best_acc, output_dir, name, *, save):
    """Save the epoch and replace the best model only when accuracy improves."""
    val_acc = checkpoint["val_acc"]
    improved = val_acc > best_acc
    best_acc = max(best_acc, val_acc)
    checkpoint = {**checkpoint, "best_acc": best_acc}
    output_dir = Path(output_dir)
    save(checkpoint, output_dir / f"{name}_e{checkpoint['epoch'] + 1}.pth")
    if improved:
        save(checkpoint, output_dir / f"{name}.pth")
    return best_acc, improved


def capture_rng_state():
    """Use only tensors and basic types accepted by torch.load(weights_only=True)."""
    import random
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "generator": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["generator"], np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def restore_training_state(checkpoint, model, optimizer, scheduler):
    """Restore continuation state after constructing the optimizer and scheduler."""
    import warnings
    missing = [key for key in ("optimizer", "scheduler") if key not in checkpoint]
    if missing:
        raise ValueError("Cannot resume training: checkpoint is missing " + ", ".join(missing))
    model.load_state_dict(checkpoint["model_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    else:
        warnings.warn("Legacy checkpoint has no RNG state; stochastic training cannot resume exactly", RuntimeWarning)
    if "training_state" not in checkpoint:
        warnings.warn("Legacy checkpoint has no training history or early-stopping state; these restart", RuntimeWarning)
    return checkpoint.get("training_state", {})
