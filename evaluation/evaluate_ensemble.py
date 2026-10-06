"""
评估软投票集成效果：convnext vs swin vs 集成（软投票）

口径：从 --data-root 指定的数据目录按类别随机抽样 N 张。
默认目录为项目中的 dataset/test。若所选目录包含训练数据，
准确率会偏高；比较集成增益时应明确数据来源，不能视为独立泛化测试。

实现说明：每张图只对 convnext / swin 各 forward 一次，
        集成概率 = 两模型 softmax 概率平均，避免重复推理。

用法：
    python evaluation/evaluate_ensemble.py [--data-root DATASET_DIR] [--n 120] [--seed 42] [--out evaluation_result.txt]
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhimo.vision.preprocess import detect_and_fix_inversion


DATASET_ROOT = ROOT / "dataset" / "test"
DEFAULT_OUT = os.path.join(ROOT, "evaluation_result.txt")


def collect_samples(n_per_class: int, seed: int, data_root=DATASET_ROOT):
    """每类抽样 n_per_class 张，返回 [(image_path, label), ...]。"""
    data_root = Path(data_root).expanduser().resolve()
    if n_per_class <= 0:
        raise ValueError("每类抽样数量 --n 必须为正整数")
    if not data_root.is_dir():
        raise ValueError(f"数据目录不存在或不是目录: {data_root}；请用 --data-root 指定数据集")
    rng = random.Random(seed)
    samples = []
    for cls_path in sorted(data_root.iterdir()):
        if not cls_path.is_dir():
            continue
        # 类目录名 → 书法家标签（去掉 "-楷"/"-行" 等后缀）
        label = cls_path.name.split("-")[0]
        files = sorted(path for path in cls_path.iterdir()
                       if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not files:
            print(f"[跳过] 空目录: {cls_path.name}")
            continue
        picked = rng.sample(files, min(n_per_class, len(files)))
        samples.extend((str(path), label) for path in picked)
        print(f"[抽样] {cls_path.name}: {len(picked)} 张")
    if not samples:
        raise ValueError(f"数据目录没有可评估图片: {data_root}；请按书法家建立子目录并放入 JPG/PNG/BMP 图片")
    return samples


def _load_models():
    """Load inference dependencies and weights only after validating the dataset."""
    import torch
    from zhimo.vision.recognizer import CalligrapherRecognizer, DEFAULT_TRANSFORM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convnext = CalligrapherRecognizer(model_path=str(ROOT / "checkpoints/convnext.pth"), device=device)
    swin = CalligrapherRecognizer(model_path=str(ROOT / "checkpoints/swin.pth"), device=device)
    convnext._load_model()
    swin._load_model()
    return device, convnext, swin, DEFAULT_TRANSFORM


def normalize_sample_labels(samples, recognizer):
    """Resolve Chinese directory names and checkpoint author IDs to output labels."""
    canonical_names = set(recognizer.id_to_label.values())
    normalized = []
    unknown = set()
    for path, label in samples:
        class_id = recognizer.label_to_id.get(label)
        if class_id is not None:
            name = recognizer.id_to_label[class_id]
        elif label in canonical_names:
            name = label
        else:
            unknown.add(label)
            continue
        normalized.append((path, name))
    if unknown:
        raise ValueError(
            "数据集包含权重不支持的类别: " + ", ".join(sorted(unknown))
            + "；请使用中文书法家名或权重 label_to_id 中的作者 ID 命名子目录"
        )
    return normalized


def infer_probs(rec, img_tensor, device):
    """单模型 forward → 1D softmax 概率"""
    import torch

    with torch.no_grad():
        logits = rec.model(img_tensor.to(device))
    return torch.softmax(logits, dim=-1).squeeze(0).cpu()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT, help="数据集根目录，子目录为中文书法家名或权重中的作者 ID")
    parser.add_argument("--n", type=int, default=120, help="每类抽样数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DEFAULT_OUT, help="结果输出文件")
    args = parser.parse_args(argv)

    try:
        data_root = args.data_root.expanduser().resolve()
        samples = collect_samples(args.n, args.seed, data_root=data_root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"\n共 {len(samples)} 张样本，开始评估（tta=False）...\n")

    device, convnext, swin, transform = _load_models()
    try:
        if convnext.id_to_label != swin.id_to_label:
            raise ValueError("两个模型的类别顺序不一致，无法按索引进行软投票评估")
        samples = normalize_sample_labels(samples, convnext)
    except ValueError as exc:
        parser.error(str(exc))
    id_to_label = convnext.id_to_label
    print(f"[评估] 设备: {device}\n")

    stats = {
        "convnext": {"correct": 0, "total": 0, "per_class": {}},
        "swin": {"correct": 0, "total": 0, "per_class": {}},
        "ensemble": {"correct": 0, "total": 0, "per_class": {}},
    }
    ens_saves = 0      # 集成对、且至少一个单模型错
    ens_breaks = 0     # 单模型都对、集成错
    both_wrong = 0     # 两个单模型都错
    agree_count = 0    # 两模型 top1 一致
    disagree_correct = {"convnext": 0, "swin": 0, "ensemble": 0}  # 分歧样本中谁对

    for i, (path, label) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        img, _ = detect_and_fix_inversion(img)
        t = transform(img).unsqueeze(0)

        p_cn = infer_probs(convnext, t, device)
        p_sw = infer_probs(swin, t, device)
        p_en = (p_cn + p_sw) / 2.0

        preds = {
            "convnext": id_to_label[p_cn.argmax().item()],
            "swin": id_to_label[p_sw.argmax().item()],
            "ensemble": id_to_label[p_en.argmax().item()],
        }
        for name, pred in preds.items():
            s = stats[name]
            s["total"] += 1
            s["per_class"].setdefault(label, {"correct": 0, "total": 0})
            s["per_class"][label]["total"] += 1
            if pred == label:
                s["correct"] += 1
                s["per_class"][label]["correct"] += 1

        if preds["convnext"] == preds["swin"]:
            agree_count += 1
        else:
            for name in ("convnext", "swin", "ensemble"):
                if preds[name] == label:
                    disagree_correct[name] += 1

        if preds["ensemble"] == label and (preds["convnext"] != label or preds["swin"] != label):
            ens_saves += 1
        if preds["convnext"] == label and preds["swin"] == label and preds["ensemble"] != label:
            ens_breaks += 1
        if preds["convnext"] != label and preds["swin"] != label:
            both_wrong += 1

        if (i + 1) % 100 == 0:
            print(f"  ... 已评估 {i + 1}/{len(samples)}")

    # ---------- 汇总 ----------
    lines = []
    lines.append("=" * 56)
    lines.append("软投票集成评估结果（按类别抽样，tta=False）")
    lines.append(f"数据目录: {data_root}")
    lines.append("时间: %s | 样本: %d 张 | seed: %d" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(samples), args.seed))
    lines.append("=" * 56)
    lines.append("")
    lines.append(f"{'配置':<10} {'正确':<7} {'总数':<7} {'准确率':<9}")
    for name in stats:
        s = stats[name]
        acc = s["correct"] / s["total"]
        lines.append(f"{name:<10} {s['correct']:<7} {s['total']:<7} {acc:.4f} ({acc*100:.2f}%)")
    lines.append("")
    lines.append("-" * 56)
    lines.append("分类别准确率")
    lines.append(f"{'类别':<6} {'convnext':<13} {'swin':<13} {'集成':<13}")
    for label in sorted(stats["convnext"]["per_class"].keys()):
        row = []
        for name in ("convnext", "swin", "ensemble"):
            pc = stats[name]["per_class"][label]
            row.append(f"{pc['correct']}/{pc['total']} ({pc['correct']/pc['total']:.3f})")
        lines.append(f"{label:<6} {'  '.join(row)}")
    lines.append("")
    lines.append("-" * 56)
    lines.append("分歧与纠错统计")
    lines.append(f"两个模型 top1 一致的样本: {agree_count}/{len(samples)} ({agree_count/len(samples)*100:.1f}%)")
    lines.append(f"分歧样本中各自答对的: convnext={disagree_correct['convnext']}, "
                 f"swin={disagree_correct['swin']}, 集成={disagree_correct['ensemble']}")
    lines.append(f"集成救回（至少一个单模型错、集成对）: {ens_saves}")
    lines.append(f"集成失误（单模型都对、集成错）: {ens_breaks}")
    lines.append(f"两个单模型都错: {both_wrong}")
    lines.append("")
    lines.append("注: 若所选目录包含训练数据，准确率绝对值会偏高；")
    lines.append("    以上用于对比集成相对单模型的增益，非严格泛化测试。")

    text = "\n".join(lines)
    print("\n" + text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[结果已写入] {args.out}")


if __name__ == "__main__":
    main()
