"""
书法图像恶劣条件模拟数据生成器（改进版 v3）
纯 NumPy / PIL 实现，无需 OpenCV
在 v2 基础上新增三种畸变：
  - 褪色（对比度降低）
  - 椒盐噪声（霉斑/虫蛀点）
  - 轻微倾斜（拍摄不正）
保持原有四大畸变，支持任意组合，保证每张图至少触发两种畸变（若启用的类型≥2）
"""

import random
import os
import numpy as np
from PIL import Image, ImageFilter


# 1. 泛黄畸变 —— 手写 YUV 变换，效果加强

def rgb_to_yuv_np(img_rgb):
    """RGB -> YUV (BT.601) 手写，输入 uint8 0-255"""
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0

    return np.stack([y, u, v], axis=-1)


def yuv_to_rgb_np(img_yuv):
    """YUV -> RGB (BT.601)"""
    y = img_yuv[..., 0]
    u = img_yuv[..., 1] - 128.0
    v = img_yuv[..., 2] - 128.0

    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def aug_yellowing(img_rgb: np.ndarray, severity: float) -> np.ndarray:
    """
    宣纸泛黄（加强版）：增加 U 分量、减少 V 分量，并降低亮度
    参数调整：
        - add_u: 35 -> 50
        - sub_v: 22 -> 30
        - 亮度衰减: (1 - 0.1) -> (1 - 0.15)
    """
    yuv = rgb_to_yuv_np(img_rgb)
    add_u = int(50 * severity)
    sub_v = int(30 * severity)
    # 亮度微降，模拟老化
    yuv[..., 0] = np.clip(yuv[..., 0] * (1 - 0.15 * severity), 0, 255)
    yuv[..., 1] = np.clip(yuv[..., 1] + add_u, 0, 255)
    yuv[..., 2] = np.clip(yuv[..., 2] - sub_v, 0, 255)
    return yuv_to_rgb_np(yuv)


# 2. 墨渍脏斑畸变 —— 限制面积，半透明混合

def aug_stain(img_rgb: np.ndarray, severity: float) -> np.ndarray:
    """随机墨渍、脏斑，最多覆盖图像面积 12%"""
    h, w = img_rgb.shape[:2]
    total_pixels = h * w
    max_cover_ratio = 0.12
    covered_pixels = 0

    max_num = int(12 * severity)
    stain_num = random.randint(2, max(3, max_num))
    out = img_rgb.copy()

    for _ in range(stain_num):
        if covered_pixels / total_pixels > max_cover_ratio:
            break

        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        # 半径限制
        r = random.randint(int(4 * severity), int(18 * severity))
        r = min(r, 20)

        # 半透明深色
        alpha = random.uniform(0.5, 0.8)
        color_dark = np.array([random.randint(5, 35)] * 3, dtype=np.float32)

        # 生成圆形掩膜
        Y, X = np.ogrid[:h, :w]
        mask = (X - x) ** 2 + (Y - y) ** 2 <= r ** 2
        mask_area = np.sum(mask)
        covered_pixels += mask_area

        # 混合
        out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + color_dark * alpha).clip(0, 255).astype(np.uint8)

    return out

# 3. 残缺裁切畸变 —— 填充边缘均值，更自然

def aug_defect_cut(img_rgb: np.ndarray, severity: float) -> np.ndarray:
    """四边随机裁切，并用邻域均值填充"""
    h, w = img_rgb.shape[:2]
    cut_rate = 0.18 * severity
    cut_px_h = int(h * cut_rate)
    cut_px_w = int(w * cut_rate)
    out = img_rgb.copy()

    opt = random.choice(["top", "bottom", "left", "right"])
    if opt == "top":
        fill_color = out[cut_px_h:min(cut_px_h + 10, h), :].mean(axis=(0, 1)).astype(np.uint8)
        out[:cut_px_h, :] = fill_color
    elif opt == "bottom":
        fill_color = out[max(0, h - cut_px_h - 10):h - cut_px_h, :].mean(axis=(0, 1)).astype(np.uint8)
        out[-cut_px_h:, :] = fill_color
    elif opt == "left":
        fill_color = out[:, cut_px_w:min(cut_px_w + 10, w)].mean(axis=(0, 1)).astype(np.uint8)
        out[:, :cut_px_w] = fill_color
    else:
        fill_color = out[:, max(0, w - cut_px_w - 10):w - cut_px_w].mean(axis=(0, 1)).astype(np.uint8)
        out[:, -cut_px_w:] = fill_color
    return out


# 4. 拓印反色畸变 —— 使用 PIL 高斯模糊

def aug_rub_invert(img_rgb: np.ndarray, severity: float) -> np.ndarray:
    """黑底白字拓片效果，带轻微模糊"""
    if random.random() > severity:
        return img_rgb
    inv = 255 - img_rgb
    pil_inv = Image.fromarray(inv).filter(ImageFilter.GaussianBlur(radius=0.4))
    return np.array(pil_inv)


# 5. 新增畸变：褪色（对比度降低）

def aug_fade(img_rgb: np.ndarray, severity: float) -> np.ndarray:
    """
    褪色畸变：模拟墨迹年久淡化，通过线性对比度压缩实现。
    severity 控制淡化程度，0 无变化，1 几乎全灰。
    """
    # 对比度因子：1 - 0.7*severity，即 severity=1 时对比度降为原来的 0.3
    contrast = 1.0 - 0.7 * severity
    # 亮度保持不变（0），公式：output = contrast * input + (1 - contrast) * 128
    faded = contrast * img_rgb.astype(np.float32) + (1.0 - contrast) * 128.0
    faded = np.clip(faded, 0, 255).astype(np.uint8)
    return faded


# 6. 新增畸变：椒盐噪声（霉斑/虫蛀点）

def aug_salt_pepper(img_rgb: np.ndarray, severity: float) -> np.ndarray:
    """
    椒盐噪声：随机将少量像素替换为纯黑或纯白，模拟霉点、虫蛀。
    severity 控制噪声密度，最大密度约 5%（严重时）。
    """
    h, w = img_rgb.shape[:2]
    # 根据 severity 计算噪声比例（0.005 到 0.05）
    density = 0.005 + 0.045 * severity
    num_pixels = int(h * w * density)
    out = img_rgb.copy()
    # 随机生成位置
    for _ in range(num_pixels):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        # 随机选择黑色(0)或白色(255)
        if random.random() < 0.5:
            out[y, x] = [0, 0, 0]
        else:
            out[y, x] = [255, 255, 255]
    return out


# 7. 新增畸变：轻微倾斜（拍摄不正）

def aug_tilt(img_rgb: np.ndarray, severity: float) -> np.ndarray:
    """
    轻微倾斜畸变：模拟拍摄时相机未正对字画，产生小角度旋转。
    severity 控制最大旋转角度，范围 0~8 度。
    """
    max_angle = 8.0 * severity
    angle = random.uniform(-max_angle, max_angle)
    if abs(angle) < 0.5:
        return img_rgb  # 角度过小不处理
    pil_img = Image.fromarray(img_rgb)
    # 旋转，expand=True 可保留全部内容，但会产生黑边，这里用原图颜色填充
    rotated = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=None)
    # 如果出现黑边，用图像边缘颜色填充（简化处理：直接用白色？保持与原图背景一致）
    # 这里选择用白色填充，因为书法图像背景通常偏白。
    # 为保险，返回纯白背景的旋转图
    # PIL rotate 默认填充黑色，需要转换为填充白色
    rotated = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
    return np.array(rotated)


# 8. 集成增强类 —— 支持 7 种畸变，保证至少两种触发

class AdverseAugment:
    def __init__(self,
                 enable_yellow: bool = True,
                 enable_stain: bool = True,
                 enable_defect: bool = True,
                 enable_rub: bool = True,
                 enable_fade: bool = True,
                 enable_sp: bool = True,
                 enable_tilt: bool = True,
                 severity: float = 0.5):
        self.enable = {
            'yellow': enable_yellow,
            'stain': enable_stain,
            'defect': enable_defect,
            'rub': enable_rub,
            'fade': enable_fade,
            'sp': enable_sp,
            'tilt': enable_tilt
        }
        self.severity = severity

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        img = np.array(pil_img)
        applied = []

        # 各畸变按概率随机尝试
        if self.enable['yellow'] and random.random() < 0.6:
            img = aug_yellowing(img, self.severity)
            applied.append('yellow')
        if self.enable['stain'] and random.random() < 0.55:
            img = aug_stain(img, self.severity)
            applied.append('stain')
        if self.enable['defect'] and random.random() < 0.4:
            img = aug_defect_cut(img, self.severity)
            applied.append('defect')
        if self.enable['rub'] and random.random() < 0.3:
            img = aug_rub_invert(img, self.severity)
            applied.append('rub')
        if self.enable['fade'] and random.random() < 0.5:
            img = aug_fade(img, self.severity)
            applied.append('fade')
        if self.enable['sp'] and random.random() < 0.5:
            img = aug_salt_pepper(img, self.severity)
            applied.append('sp')
        if self.enable['tilt'] and random.random() < 0.4:
            img = aug_tilt(img, self.severity)
            applied.append('tilt')

        # 保证至少触发两种畸变（如果启用的类型≥2）
        available_effects = [k for k, v in self.enable.items() if v]
        min_required = min(2, len(available_effects))
        while len(applied) < min_required:
            remaining = [k for k in available_effects if k not in applied]
            if not remaining:
                break
            choice = random.choice(remaining)
            if choice == 'yellow':
                img = aug_yellowing(img, self.severity)
            elif choice == 'stain':
                img = aug_stain(img, self.severity)
            elif choice == 'defect':
                img = aug_defect_cut(img, self.severity)
            elif choice == 'rub':
                img = aug_rub_invert(img, self.severity)
            elif choice == 'fade':
                img = aug_fade(img, self.severity)
            elif choice == 'sp':
                img = aug_salt_pepper(img, self.severity)
            elif choice == 'tilt':
                img = aug_tilt(img, self.severity)
            applied.append(choice)

        return Image.fromarray(img)

# 9. 离线批量生成恶劣数据集

def generate_offline_adverse_dataset(
    src_root: str,
    dst_root: str,
    enable_yellow=True,
    enable_stain=True,
    enable_defect=True,
    enable_rub=True,
    enable_fade=True,
    enable_sp=True,
    enable_tilt=True,
    severity=0.5
):
    """
    src_root：原始数据集目录（按作者分文件夹）
    dst_root：输出恶劣图片目录，保持原目录结构
    """
    aug = AdverseAugment(
        enable_yellow=enable_yellow,
        enable_stain=enable_stain,
        enable_defect=enable_defect,
        enable_rub=enable_rub,
        enable_fade=enable_fade,
        enable_sp=enable_sp,
        enable_tilt=enable_tilt,
        severity=severity
    )

    author_list = os.listdir(src_root)
    for auth_id in author_list:
        src_dir = os.path.join(src_root, auth_id)
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(dst_root, auth_id)
        os.makedirs(dst_dir, exist_ok=True)

        img_names = [n for n in os.listdir(src_dir)
                     if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        for name in img_names:
            img_path = os.path.join(src_dir, name)
            img = Image.open(img_path).convert("RGB")
            bad_img = aug(img)
            bad_img.save(os.path.join(dst_dir, name))

    print(f"恶劣数据集生成完成，保存路径：{dst_root}")


# 10. 直接运行示例
if __name__ == "__main__":
    # 示例：生成全七种畸变数据集（每张图至少触发2种）
    generate_offline_adverse_dataset(
        src_root="calligraphy_dataset/train",
        dst_root="calligraphy_all_adverse_v3/train",
        enable_yellow=True,
        enable_stain=True,
        enable_defect=True,
        enable_rub=True,
        enable_fade=True,
        enable_sp=True,
        enable_tilt=True,
        severity=0.5
    )
    # 如需关闭某种畸变，将对应 enable_xxx 设为 False 即可