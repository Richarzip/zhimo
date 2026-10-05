"""
运行方式:
  - 双击本文件
  - 或在 IDE / 终端中运行: python training/train.py
  - 也可通过命令行覆盖参数: python training/train.py --epochs 10 --backbone resnet50
"""

import os
# hugging face 镜像，用于预训练权重下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  
import sys
import io
import random
import argparse
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import timm

# ====================== 可视化导入与配置 ======================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 解决中文显示问题（必须加，否则书法家名字乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# ==================================================================

# Windows 控制台 UTF-8 输出修复
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════
# ⚙️  训练配置 — 修改这里即可调整训练参数
# ════════════════════════════════════════════════════════════

CONFIG = {
    # ── 数据集路径 ──────────────────────────────────────
    # 请根据实际生成的文件夹名称修改，例如 "./dataset"
    "data_root": "./dataset", 

    # 书法家ID映射（为了兼容后续的训练，统一用拼音缩写）
    "author_map": {
        "wxz":  "王羲之", "yzq":  "颜真卿", "lgq":  "柳公权", "sgt":  "孙过庭",
        "smh":  "沙孟海", "mf":   "米芾",   "htj":  "黄庭坚", "oyx":  "欧阳询",
        "zmf":  "赵孟頫", "csl":  "褚遂良", "wzm":  "文徵明", "yyr":  "于右任",
        "hy":   "弘一",   "bdsr": "八大山人","shz":  "宋徽宗", "mzd":  "毛泽东",
        "lx":   "鲁迅",   "cx":   "蔡襄",   "dqc":  "董其昌", "ly":   "刘墉",
        "wd":   "王铎",   "wcs":  "吴昌硕", "ybs":  "伊秉绶", "zx":   "朱熹",
        "cdf":  "陈道复", "fs":   "傅山",   "jn":   "金农",   "qg":   "启功",
        "ss":   "苏轼",   "xys":  "鲜于枢", "zrt":  "张瑞图", "zzq":  "赵之谦",
        "zbq":  "郑板桥", "zym":  "祝允明", "hsj":  "何绍基", "klnn": "康里巎巎",
        "sz":   "沈周",   "ty":   "唐寅",   "wc":   "王宠",   "xw":   "徐渭",
        "nyl":  "倪元璐", "wfa":  "王福庵", "cqw":  "成亲王", "lyb":  "李阳冰",
        "txz":  "唐玄宗", "dsr":  "邓石如", "ysn":  "虞世南", "sgz":  "宋高宗",
        "hs":   "怀素",   "lsz":  "林散之",
        # 冲突名字处理
        "wxz2": "王献之", "zxu":  "张旭",  "zyao": "钟繇", "zy": "智永",
    },

    # 别名映射（用于解决异体字、不同写法的问题，统一映射到标准名）
    "author_alias": {
        "文征明": "文徵明",
        "赵孟俯": "赵孟頫",
    },

    # ── 输出路径 ────────────────────────────────────────
    "output_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints_test_gelu"),

    # ── 模型 ────────────────────────────────────────────
    # 支持: resnet50 / convnext_tiny / efficientnet_b3 / convnext_small / vit_small_patch16_224 / swin_tiny_patch4_window7_224
    "backbone": "convnext_tiny",

    # ── 训练超参数 ──────────────────────────────────────
    "epochs": 50,          
    "batch_size": 64,
    "lr": 8e-5,
    "weight_decay": 5e-2,
    "label_smoothing": 0.1,
    "max_samples": None,   
    "early_stop_patience": 10,  # 新增：连续10个epoch验证集Loss不下降就早停

    # ── 随机种子 ────────────────────────────────────────
    "seed": 42,
}

# 数据增强
TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandAugment(num_ops=2, magnitude=5), # 更强的自动数据增强
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.2) # 随机遮挡，防止模型只看局部特征
])

VAL_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ════════════════════════════════════════════════════════════
# 数据集
# ════════════════════════════════════════════════════════════

class ArchiveDataset(Dataset):
    def __init__(self, data_root, author_map, author_alias, split="train", transform=None, max_samples=None):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 建立标签映射
        self.id_to_label = {}
        self.label_to_id = {}
        for idx, (author_id, author_name) in enumerate(sorted(author_map.items())):
            self.id_to_label[idx] = author_name
            self.label_to_id[author_name] = idx
            self.label_to_id[author_id] = idx  # 同时支持拼音缩写和中文名查找

        # 扫描指定 split (train 或 test)
        sub_dir = os.path.join(data_root, split)
        if not os.path.isdir(sub_dir):
            raise ValueError(f"目录不存在: {sub_dir}，请检查数据集路径和 split 参数")

        for author_id in os.listdir(sub_dir):
            author_dir = os.path.join(sub_dir, author_id)
            if not os.path.isdir(author_dir):
                continue
                
            # 处理别名（异体字归一化）
            resolved_id = author_alias.get(author_id, author_id)
            
            if resolved_id not in self.label_to_id:
                print(f"[警告] 未识别的书法家文件夹: {author_id}，跳过")
                continue
                
            label = self.label_to_id[resolved_id]
            for fname in os.listdir(author_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                self.image_paths.append(os.path.join(author_dir, fname))
                self.labels.append(label)

        # 限制样本数（用于调试）
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.labels = self.labels[:max_samples]

        # 统计
        counter = Counter(self.labels)
        print(f"[{split.upper()}] total={len(self.image_paths)}")
        for label_id in sorted(counter.keys()):
            name = self.id_to_label[label_id]
            print(f"  {name}: {counter[label_id]} 张")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


# ════════════════════════════════════════════════════════════
# 模型
# ════════════════════════════════════════════════════════════

class ModelWrapper(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        return self.head(self.encoder(x))


def build_model(num_classes, backbone, pretrained=True):
    encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
    feature_dim = encoder.num_features
    head = nn.Sequential(
        nn.BatchNorm1d(feature_dim),
        nn.Dropout(0.3),
        nn.Linear(feature_dim, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(512, num_classes),
    )
    return ModelWrapper(encoder, head)


# ════════════════════════════════════════════════════════════
# 训练与验证
# ════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.3f}"})
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, id_to_label):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import classification_report
    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    report = classification_report(
        all_labels, all_preds, target_names=target_names,
        digits=4, output_dict=True, zero_division=0
    )
    return total_loss / total, correct / total, report, all_preds, all_labels


# ════════════════════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════════════════════

def plot_loss_acc_curve(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    ax1.set_title('Train & Validation Loss Curve', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Val Acc')
    ax2.set_title('Train & Validation Accuracy Curve', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1)  
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Loss-Acc curve: {save_path}")


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names,
        annot_kws={"size": 8},
        vmin=0, vmax=1
    )

    plt.title('Normalized Confusion Matrix (Best Model)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Normalized Confusion Matrix: {save_path}")


# ════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════

def main():
    # ── 解析命令行参数 ────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train Calligrapher Classifier")
    for key, val in CONFIG.items():
        if key in ["author_map", "author_alias"]:
            continue
        if val is None:
            parser.add_argument(f"--{key}", type=str, default=val)
        elif isinstance(val, bool):
            parser.add_argument(f"--{key}", action="store_true", default=val)
        elif isinstance(val, int):
            parser.add_argument(f"--{key}", type=int, default=val)
        elif isinstance(val, float):
            parser.add_argument(f"--{key}", type=float, default=val)
        else:
            parser.add_argument(f"--{key}", type=str, default=val)

    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    cfg = {**CONFIG, **{k: v for k, v in vars(args).items() if v is not None}}

    # ── 随机种子 ─────────────────────────────────────────
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # ── 设备选择 ─────────────────────────────────────────
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            torch.zeros(1).cuda()
            print("[GPU]  检测到 CUDA，可正常使用")
        except RuntimeError as e:
            print(f"[GPU]  CUDA 已安装但不可用: {e}")
            print("[CPU]  自动回退到 CPU")
            use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else None

    print(f"CalliSystem 书体分类器训练\n")
    device_str = f"GPU: {gpu_name}" if gpu_name else "CPU"
    print(f"  计算设备: {device_str}")
    print(f"  数据集:   {cfg['data_root']}")
    print(f"  书法家:   {len(cfg['author_map'])} 类")
    print(f"  模型:     {cfg['backbone']}")
    print(f"  Epochs:   {cfg['epochs']}")
    print(f"  Batch:    {cfg['batch_size']}")
    print(f"  LR:       {cfg['lr']}")
    print(f"  早停耐心: {cfg['early_stop_patience']} 个 Epoch")
    print(f"  输出目录:  {cfg['output_dir']}")
    print(f"{'='*50}\n")

    # ── 检查数据集 ────────────────────────────────────────
    if not os.path.isdir(cfg["data_root"]):
        print(f"[error] 目录不存在: {cfg['data_root']}")
        sys.exit(1)

    # ── 数据集 & DataLoader ─────────────────────────────
    print("加载训练集...")
    train_dataset = ArchiveDataset(
        data_root=cfg["data_root"],
        author_map=cfg["author_map"],
        author_alias=cfg["author_alias"],
        split="train",
        transform=TRAIN_TRANSFORM,
        max_samples=cfg["max_samples"],
    )

    print("\n加载测试/验证集...")
    val_dataset = ArchiveDataset(
        data_root=cfg["data_root"],
        author_map=cfg["author_map"],
        author_alias=cfg["author_alias"],
        split="test",
        transform=VAL_TRANSFORM,
        max_samples=cfg["max_samples"],
    )

    num_classes = len(train_dataset.id_to_label)
    class_names = [train_dataset.id_to_label[i] for i in sorted(train_dataset.id_to_label.keys())]

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张\n")

    # ── 模型 ─────────────────────────────────────────────
    model = build_model(num_classes=num_classes, backbone=cfg["backbone"]).to(device)

    criterion  = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer   = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

     # 新增：Warmup + Cosine 退火
    warmup_epochs = 5  
    min_lr = 1e-6      # 学习率下限
    
    # 预热阶段：学习率从 0.1*lr 线性升到 lr
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    # 余弦退火阶段：从 lr 降到 min_lr
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - warmup_epochs, eta_min=min_lr
    )
    # 串联调度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )

    best_acc    = 0.0
    start_epoch = 0

    # 初始化可视化数据列表
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_all_preds, best_all_labels = [], []

    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"从 epoch {start_epoch} 恢复，最佳准确率: {best_acc:.4f}\n")

    # ── 训练循环 ─────────────────────────────────────────
    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_name = f"calligrapher_classifier"

    # 初始化早停变量
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n{'='*70}")
        print(f"  Epoch {epoch+1}/{cfg['epochs']}  |  最佳 {best_acc:.4f}\n")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        scheduler.step()

        val_loss, val_acc, report, all_preds, all_labels = validate(
            model, val_loader, criterion, device, train_dataset.id_to_label
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\n[Epoch {epoch+1}/{cfg['epochs']}]")
        print(f"  训练   Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  验证   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # 打印每个书法家的指标
        for label_key in sorted(report.keys()):
            if label_key not in ("accuracy", "macro avg", "weighted avg"):
                r = report[label_key]
                print(f"  {label_key}: P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": val_acc,
            "num_classes": num_classes,
            "backbone": cfg["backbone"],
            "id_to_label": train_dataset.id_to_label,
            "label_to_id": train_dataset.label_to_id,
        }

        torch.save(ckpt, os.path.join(cfg["output_dir"], f"{ckpt_name}_e{epoch+1}.pth"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(cfg["output_dir"], f"{ckpt_name}.pth"))
            print(f"  [best] 准确率: {val_acc:.4f}  -> {ckpt_name}.pth")
            best_all_preds = all_preds
            best_all_labels = all_labels

        # 早停逻辑（监控验证集 Loss）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # 如果 Loss 下降，重置计数器
        else:
            patience_counter += 1
            print(f"  [早停监测] 验证集 Loss 未下降，已忍耐 {patience_counter}/{cfg['early_stop_patience']} 个 Epoch")
            
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"\n{'!'*60}")
                print(f"  触发早停！验证集 Loss 连续 {cfg['early_stop_patience']} 个 Epoch 未下降。")
                print(f"  停止训练，自动保留最佳模型 (Val Acc: {best_acc:.4f})")
                print(f"{'!'*60}")
                break  # 跳出训练循环

    print(f"\n{'*'*60}")
    print(f"  训练完成！最佳验证准确率: {best_acc:.4f}")
    print(f"  权重: {os.path.join(cfg['output_dir'], ckpt_name)}.pth")

    # ====================== 训练完成后生成可视化 ======================
    print(f"\n{'*'*60}")
    print("  生成可视化图表")
    
    loss_acc_path = os.path.join(cfg["output_dir"], "loss_accuracy_curve.png")
    plot_loss_acc_curve(train_losses, val_losses, train_accs, val_accs, loss_acc_path)
    
    cm_path = os.path.join(cfg["output_dir"], "confusion_matrix.png")
    plot_confusion_matrix(best_all_labels, best_all_preds, class_names, cm_path)
    
    print("\n可视化完成")
    print(f"   损失准确率曲线: {loss_acc_path}")
    print(f"   混淆矩阵: {cm_path}")
    # ==================================================================


if __name__ == "__main__":
    main()
