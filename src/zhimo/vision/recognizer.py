"""Stable public interface for single-model recognition."""

from .recognizer_impl import (
    CalligrapherRecognizer,
    DEFAULT_TRANSFORM,
    TTA_FLIP_TRANSFORM,
    TTA_NATIVE_TRANSFORM,
    TTA_SCALE_TRANSFORM,
)

__all__ = [
    "CalligrapherRecognizer",
    "DEFAULT_TRANSFORM",
    "TTA_FLIP_TRANSFORM",
    "TTA_NATIVE_TRANSFORM",
    "TTA_SCALE_TRANSFORM",
]
