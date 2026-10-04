import os
import shutil
import random
from PIL import Image

SRC_IMG_DIR = "your_images"          # 原始图片目录
SRC_LABEL_DIR = "your_labels"        # 原始标签目录
DST_DIR = "dataset_yolo"             # 输出目录

TRAIN_RATIO = 0.85                   # 训练集比例
CLASS_ID = 0                         # 统一类别ID（只检测"字"）


def convert_bbox(size, box):
    """将 (x, y, w, h) 转换为 YOLO 格式的 (x_center, y_center, w, h)，并归一化"""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x, y, w, h = box
    x_center = (x + w / 2.0) * dw
    y_center = (y + h / 2.0) * dh
    w_norm = w * dw
    h_norm = h * dh
    return (x_center, y_center, w_norm, h_norm)

def main():
    # 创建输出目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DST_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DST_DIR, "labels", split), exist_ok=True)

    # 获取所有图片文件
    img_files = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(img_files)

    split_idx = int(len(img_files) * TRAIN_RATIO)
    splits = {
        "train": img_files[:split_idx],
        "val": img_files[split_idx:]
    }

    for split, files in splits.items():
        for img_name in files:
            img_path = os.path.join(SRC_IMG_DIR, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(SRC_LABEL_DIR, label_name)

            if not os.path.exists(label_path):
                print(f"警告：找不到标签文件 {label_path}，跳过 {img_name}")
                continue

            # 获取图片尺寸
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # 读取并转换标签
            yolo_lines = []
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:  # 至少需要: 字符 x y w h
                        continue
                    try:
                        x, y, w, h = map(float, parts[1:5])
                        x_c, y_c, w_n, h_n = convert_bbox((img_w, img_h), (x, y, w, h))
                        yolo_lines.append(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
                    except ValueError:
                        continue

            if not yolo_lines:
                continue

            # 复制图片并写入新标签
            dst_img_path = os.path.join(DST_DIR, "images", split, img_name)
            dst_label_path = os.path.join(DST_DIR, "labels", split, label_name)

            shutil.copy2(img_path, dst_img_path)
            with open(dst_label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))

        print(f"[{split}] 处理完成：{len(files)} 张图片")

if __name__ == "__main__":
    main()