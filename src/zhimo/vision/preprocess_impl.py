"""图像预处理工具：反色检测与校正。

只依赖 numpy 和 PIL，不引入 torch/timm，方便前端服务与测试
在不加载深度学习模型的前提下独立使用。
"""
import numpy as np
from PIL import Image


def detect_and_fix_inversion(
    image: Image.Image,
    threshold: int = 128,
    min_bright_ratio: float = 0.02,
) -> tuple[Image.Image, dict]:
    """
    检测黑底白字的反色书法（如拓印），若是则反转为正常白底黑字。

    判断依据（拓印黑底白字 vs 正常白底黑字 / 深色宣纸）：
      1. 灰度中位数 < threshold：背景整体偏暗
         （正常白底黑字中位数应接近 255，黑底白字则接近 0）
      2. 亮像素（>200）占比 >= min_bright_ratio：确实存在"白色字迹"，
         用于防止把深色宣纸/暗背景的正常书法误判为反色并反转。

    Args:
        image: PIL.Image 输入图
        threshold: 背景明暗分界阈值（默认 128）
        min_bright_ratio: 亮像素最小占比，用于确认存在白色前景

    Returns:
        (fixed_image, info)
        fixed_image: 已修正为白底黑字的 RGB 图（非反色时原样返回）
        info: {"was_inverted": bool, "median": float, "bright_ratio": float}
    """
    rgb = image.convert("RGB")
    gray = np.array(rgb.convert("L"))
    median_val = float(np.median(gray))
    bright_ratio = float(np.mean(gray > 200))

    if median_val < threshold and bright_ratio >= min_bright_ratio:
        # 黑底白字：整图反色 → 白底黑字
        fixed = Image.fromarray((255 - np.array(rgb)).astype(np.uint8))
        return fixed, {
            "was_inverted": True,
            "median": round(median_val, 1),
            "bright_ratio": round(bright_ratio, 4),
        }

    return rgb, {
        "was_inverted": False,
        "median": round(median_val, 1),
        "bright_ratio": round(bright_ratio, 4),
    }
