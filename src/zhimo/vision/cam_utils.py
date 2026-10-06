"""Shared normalization and RGB reconstruction for model-aligned CAM overlays."""

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cam_background(input_tensor) -> np.ndarray:
    """Recover the exact RGB view used by CAM from a normalized 1CHW tensor."""
    image = input_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    image = image.astype(np.float32)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.clip(image * std + mean, 0.0, 1.0)
