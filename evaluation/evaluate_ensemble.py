"""
评估软投票集成效果：convnext vs swin vs 集成（软投票）

口径：从 D:\\experiment3\\dataset_total\\dataset1 每类随机抽样 N 张
（该数据集为训练数据的一部分，模型可能见过，故准确率会偏高；
 本脚本目的只是对比"集成 vs 单模型"的相对增益，非严格泛化测试。）

实现说明：每张图只对 convnext / swin 各 forward 一次，
        集成概率 = 两模型 softmax 概率平均，避免重复推理。

用法：
    python evaluation/evaluate_ensemble.py [--n 120] [--seed 42] [--out evaluation_result.txt]
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhimo.vision.recognizer import CalligrapherRecognizer, DEFAULT_TRANSFORM
from zhimo.vision.preprocess import detect_and_fix_inversion


DATASET_ROOT = r"D:\experiment3\dataset_total\dataset1"
DEFAULT_OUT = os.path.join(ROOT, "evaluation_result.txt")


def collect_samples(n_per_class: int, seed: int):
    """每类抽样 n_per_class 张，返回 [(image_path, label), ...]"""
    random.seed(seed)
    samples = []
    for cls_dir in sorted(os.listdir(DATASET_ROOT)):
        cls_path = os.path.join(DATASET_ROOT, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        # 类目录名 → 54 类标签（去掉 "-楷"/"-行" 等后缀）
        label = cls_dir.split("-")[0]
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if not files:
            print(f"[跳过] 空目录: {cls_dir}")
            continue
        picked = random.sample(files, min(n_per_class, len(files)))
        for f in picked:
            samples.append((os.path.join(cls_path, f), label))
        print(f"[抽样] {cls_dir}: {len(picked)} 张")
    return samples


def infer_probs(rec, img_tensor, device):
    """单模型 forward → 1D softmax 概率"""
    with torch.no_grad():
        logits = rec.model(img_tensor.to(device))
    return torch.softmax(logits, dim=-1).squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120, help="每类抽样数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DEFAULT_OUT, help="结果输出文件")
    args = parser.parse_args()

    samples = collect_samples(args.n, args.seed)
    print(f"\n共 {len(samples)} 张样本，开始评估（tta=False）...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convnext = CalligrapherRecognizer(model_path="./checkpoints/convnext.pth", device=device)
    swin = CalligrapherRecognizer(model_path="./checkpoints/swin.pth", device=device)
    convnext._load_model()
    swin._load_model()
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
        t = DEFAULT_TRANSFORM(img).unsqueeze(0)

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
    lines.append("软投票集成评估结果（dataset1 抽样，tta=False）")
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
    lines.append("注: dataset1 是训练数据的一部分，准确率绝对值偏高；")
    lines.append("    以上用于对比集成相对单模型的增益，非严格泛化测试。")

    text = "\n".join(lines)
    print("\n" + text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[结果已写入] {args.out}")


if __name__ == "__main__":
    main()
