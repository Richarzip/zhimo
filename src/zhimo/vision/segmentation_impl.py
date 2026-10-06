"""
segment.py
传统方法切分书法作品 —— 用于快速验证效果
"""

import cv2
import numpy as np
from PIL import Image


# 方法一：投影法（适合规整排列的书法）
def segment_by_projection(img_rgb: np.ndarray) -> list:
    """
    投影法切分：分别用垂直投影找列，水平投影找行
    
    参数：
        img_rgb: RGB 图像 (H, W, 3)
    返回：
        boxes: [(x, y, w, h), ...]
    """
    # 转灰度 + 二值化（白底黑字 → 字为白色前景）
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 轻微去噪（去掉孤立像素）
    binary = cv2.medianBlur(binary, 3)

    # 垂直投影：找出每一列的像素和
    vertical_proj = np.sum(binary, axis=0)
    # 水平投影：找出每一行的像素和
    horizontal_proj = np.sum(binary, axis=1)

    # 找列边界（按列切）
    col_ranges = _find_ranges(vertical_proj, threshold=vertical_proj.max() * 0.03)
    # 找行边界（按行切）
    row_ranges = _find_ranges(horizontal_proj, threshold=horizontal_proj.max() * 0.03)

    boxes = []
    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            region = binary[r_start:r_end, c_start:c_end]
            # 过滤掉内容太少的区域
            if np.sum(region) < 200:
                continue
            w = c_end - c_start
            h = r_end - r_start
            # 过滤掉宽高比异常的（比如太扁或太窄）
            if w < 10 or h < 10:
                continue
            aspect = w / h
            if aspect < 0.2 or aspect > 5.0:
                continue
            boxes.append((c_start, r_start, w, h))

    return boxes


def _find_ranges(proj: np.ndarray, threshold: float) -> list:
    """从投影数组里找出连续非零区间"""
    ranges = []
    in_range = False
    start = 0
    for i, v in enumerate(proj):
        if v > threshold and not in_range:
            start = i
            in_range = True
        elif v <= threshold and in_range:
            ranges.append((start, i))
            in_range = False
    if in_range:
        ranges.append((start, len(proj)))
    return ranges



# 方法二：连通域法（适合连笔、不规则排列）
def segment_by_connected_components(img_rgb: np.ndarray) -> list:
    """
    连通域法切分：把相邻的笔画连成一个整体
    
    参数：
        img_rgb: RGB 图像 (H, W, 3)
    返回：
        boxes: [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学膨胀：让相邻笔画连在一起（关键步骤！）
    # 膨胀核越大，越容易把相邻字连起来；核越小，越容易把一个字切碎
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)

    h_img, w_img = img_rgb.shape[:2]
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # 过滤太小的（噪点）
        if area < 500:
            continue
        # 过滤太大的（可能是整张图或大块背景）
        if area > h_img * w_img * 0.7:
            continue
        # 过滤宽高比异常的
        aspect = w / h
        if aspect < 0.15 or aspect > 6.0:
            continue
        boxes.append((x, y, w, h))

    return boxes


# 方法三：自适应选择（推荐用于生产）
def segment_auto(img_rgb: np.ndarray) -> dict:
    """
    自适应选择切分方法：
    1. 先试连通域（更适合书法）
    2. 如果结果不合理，退回投影法
    """
    # 先试连通域
    boxes_cc = segment_by_connected_components(img_rgb)
    
    # 试投影法
    boxes_proj = segment_by_projection(img_rgb)

    # 选择策略
    # 如果连通域结果在合理范围（2-30个），用连通域
    if 2 <= len(boxes_cc) <= 30:
        boxes = boxes_cc
        method = "connected_components"
    # 否则如果投影法结果合理，用投影法
    elif 2 <= len(boxes_proj) <= 30:
        boxes = boxes_proj
        method = "projection"
    # 2-30 只是优选范围，不能丢弃更长作品的有效切分。
    else:
        candidates = [
            (boxes_cc, "connected_components"),
            (boxes_proj, "projection"),
        ]
        multi = [item for item in candidates if len(item[0]) >= 2]
        nonempty = [item for item in candidates if item[0]]
        if multi:
            if len(multi) > 1:
                # 整页投影可能只保留顶部少数行，不能仅因框少就优先采用。
                # 先保留覆盖墨迹接近最多的候选，再减少笔画碎片。
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                coverage = []
                for candidate_boxes, _ in multi:
                    mask = np.zeros(gray.shape, dtype=bool)
                    for x, y, w, h in candidate_boxes:
                        mask[y:y + h, x:x + w] = True
                    coverage.append(np.count_nonzero(binary[mask]))
                # 允许少量边缘笔画差异，避免为极小的覆盖收益选取更多碎片。
                minimum_coverage = max(coverage) * 0.9
                multi = [item for item, score in zip(multi, coverage)
                         if score >= minimum_coverage]
            boxes, method = min(multi, key=lambda item: len(item[0]))
        else:
            boxes, method = (nonempty or candidates)[0]
        method += "(fallback)"

    return {
        "boxes": boxes,
        "method": method,
        "num_boxes": len(boxes),
        "cc_count": len(boxes_cc),
        "proj_count": len(boxes_proj),
    }


# 可视化
def visualize(img_rgb: np.ndarray, boxes: list, save_path: str = "segment_result.png"):
    """把边界框画到原图上保存"""
    vis = img_rgb.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(vis, str(i + 1), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    Image.fromarray(vis).save(save_path)
    print(f"可视化结果已保存到 {save_path}")


# 测试入口
if __name__ == "__main__":
    # 改成你的图片路径
    img_path = "./image_test/image.png"
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    print(f"图片尺寸: {img_np.shape}")

    # 三种方法都试
    print("\n=== 连通域法 ===")
    boxes_cc = segment_by_connected_components(img_np)
    print(f"检测到 {len(boxes_cc)} 个区域")
    visualize(img_np, boxes_cc, "result_cc.png")

    print("\n=== 投影法 ===")
    boxes_proj = segment_by_projection(img_np)
    print(f"检测到 {len(boxes_proj)} 个区域")
    visualize(img_np, boxes_proj, "result_proj.png")

    print("\n=== 自适应 ===")
    result = segment_auto(img_np)
    print(f"使用方法: {result['method']}")
    print(f"检测到 {result['num_boxes']} 个区域")
    visualize(img_np, result["boxes"], "result_auto.png")