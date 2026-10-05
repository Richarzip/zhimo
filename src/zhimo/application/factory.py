"""Application-level dependency factories."""

from __future__ import annotations

from pathlib import Path

from zhimo.config import get_settings


def resolve_model_paths() -> list[Path]:
    return list(get_settings().model_paths)


def get_recognizer():
    # Keep torch/timm out of health checks and lightweight CLI commands.
    from zhimo.vision import CalligrapherRecognizer, EnsembleRecognizer

    paths = resolve_model_paths()
    if len(paths) == 1:
        return CalligrapherRecognizer(model_path=str(paths[0]))
    return EnsembleRecognizer(model_paths=[str(path) for path in paths])
