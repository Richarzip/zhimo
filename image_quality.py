"""
image_quality.py
图像质量评估 —— 无监督诊断，不修复
"""

import cv2
import numpy as np


def assess_image_quality(img_rgb: np.ndarray) -> dict:
    """无监督评估图像劣化程度，返回各畸变评分（0~1，越高越严重）。"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_uint8 = gray.astype(np.uint8)
    h, w = gray.shape

    # 1. 泛黄：LAB B 通道偏离中性灰的程度
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    b_mean = float(np.mean(lab[..., 2]))
    yellow_score = float(np.clip(abs(b_mean - 128) / 40.0, 0, 1))

    # 2. 褪色：灰度标准差越低，对比度越低
    gray_std = float(np.std(gray))
    fade_score = float(np.clip((85 - gray_std) / 60.0, 0, 1))

    # 3. 噪声：中值滤波残差
    denoised = cv2.medianBlur(gray_uint8, 3)
    residual = np.abs(gray - denoised.astype(np.float32))
    noise_pixels = float(np.sum(residual > 30))
    noise_score = float(np.clip(noise_pixels / (h * w) * 25, 0, 1))

    # 4. 模糊：拉普拉斯方差
    laplacian_var = float(cv2.Laplacian(gray_uint8, cv2.CV_64F).var())
    blur_score = float(np.clip((100 - laplacian_var) / 100, 0, 1))

    # 综合评分（加权）
    overall = float(np.clip(
        0.35 * yellow_score + 0.25 * fade_score +
        0.25 * noise_score + 0.15 * blur_score,
        0, 1
    ))

    return {
        "yellow": round(yellow_score, 3),
        "fade": round(fade_score, 3),
        "noise": round(noise_score, 3),
        "blur": round(blur_score, 3),
        "overall": round(overall, 3),
    }