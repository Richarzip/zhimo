"""
Calligrapher Recognizer - 书法家分类推理端
对应训练脚本: train.py（类别数按 CONFIG["author_map"]，当前 54 类）
功能: 输入单张/批量书法图片 → 输出书法家名称、置信度、所有类别概率
"""

import os
import base64
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as T

from .cam_utils import IMAGENET_MEAN, IMAGENET_STD, cam_background


# 推理预处理（与训练验证集保持一致）
DEFAULT_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 翻转 TTA
TTA_FLIP_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(p=1.0),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 尺度 TTA（放大）
TTA_SCALE_TRANSFORM = T.Compose([
    T.Resize((288, 288)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 尺度 TTA（原尺寸直接 resize 到 224）
TTA_NATIVE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


from .preprocess_impl import detect_and_fix_inversion


# ====================== 模型定义 ======================

class ModelWrapper(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        return self.head(self.encoder(x))


def build_model(num_classes, backbone, pretrained=False):
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


# ====================== 书法家识别器 ======================

class CalligrapherRecognizer:
    def __init__(
        self,
        model_path: str = "./checkpoints/convnext.pth",    ##############权重
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.transform = DEFAULT_TRANSFORM
        self.model_path = model_path

        # 模型和元数据
        self.model = None
        self.id_to_label = {}   # 索引 → 书法家中文名
        self.label_to_id = {}
        self.num_classes = 0
        self.backbone = ""
        self.best_acc = 0.0

    # ---------- 模型加载 ----------

    def _load_model(self):
        """懒加载模型，第一次调用 recognize 时才加载权重"""
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"权重文件不存在: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)

        # 读取训练时保存的元数据
        self.id_to_label = ckpt["id_to_label"]
        self.label_to_id = ckpt["label_to_id"]
        self.num_classes = ckpt["num_classes"]
        self.backbone = ckpt["backbone"]
        self.best_acc = ckpt.get("best_acc", 0.0)

        print(f"[书法家识别器] 加载成功")
        print(f"  权重路径: {self.model_path}")
        print(f"  模型: {self.backbone}")
        print(f"  类别数: {self.num_classes}")
        print(f"  最佳验证准确率: {self.best_acc:.4f}")
        print(f"  支持书法家: {', '.join(self.id_to_label[i] for i in sorted(self.id_to_label.keys()))}\n")

        self.model = build_model(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained=False,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    # ---------- 内部工具 ----------

    def _sorted_class_ids(self):
        return sorted(self.id_to_label.keys())

    def _build_result_from_probs(self, probs: torch.Tensor) -> dict:
        """把 1D 概率张量组装成返回 dict"""
        top_idx = probs.argmax().item()
        return {
            "calligrapher": self.id_to_label[top_idx],
            "confidence": round(probs[top_idx].item(), 4),
            "all_probabilities": {
                self.id_to_label[i]: round(probs[i].item(), 4)
                for i in self._sorted_class_ids()
            },
            "num_classes": self.num_classes,
            "model_backbone": self.backbone,
        }

    def _get_target_layers(self):
        """根据 backbone 名字返回合适的 Grad-CAM 目标层"""
        name = self.backbone.lower()
        enc = self.model.encoder

        if "swin" in name:
            # timm SwinTransformer: encoder.layers[-1].blocks[-1].norm1
            return [enc.layers[-1].blocks[-1].norm1]
        if "convnext" in name:
            # timm ConvNeXt: encoder.stages[-1].blocks[-1]
            return [enc.stages[-1].blocks[-1]]
        if "vit" in name or "deit" in name or "beit" in name:
            return [enc.blocks[-1].norm1]
        if "resnet" in name:
            return [enc.layer4[-1]]
        if "efficientnet" in name:
            return [enc.conv_head]
        raise ValueError(f"尚未配置 {self.backbone} 的 Grad-CAM 空间特征层")

    def _get_cam_reshape_transform(self):
        """Convert transformer features to the NCHW layout required by Grad-CAM."""
        name = self.backbone.lower()
        enc = self.model.encoder
        if "swin" in name:
            # timm Swin norm1 exposes (batch, height, width, channels).
            return lambda tensor: tensor.permute(0, 3, 1, 2)
        if "vit" in name or "deit" in name or "beit" in name:
            height, width = enc.patch_embed.grid_size
            prefix_tokens = enc.num_prefix_tokens

            def reshape_tokens(tensor):
                patches = tensor[:, prefix_tokens:, :]
                return patches.reshape(
                    tensor.shape[0], height, width, tensor.shape[-1]
                ).permute(0, 3, 1, 2)

            return reshape_tokens
        return None

    # ---------- 单张推理 ----------

    @torch.no_grad()
    def recognize(self, image: Image.Image, tta: bool = False) -> dict:
        """
        识别单张书法图片
        Args:
            image: PIL.Image 格式的图片
            tta:   是否启用测试时增强（原图 + 翻转 + 两尺度平均）
        Returns:
            dict: 识别结果
        """
        self._load_model()

        # 反色预处理：拓印黑底白字 → 白底黑字，再进入正式推理
        image, _ = detect_and_fix_inversion(image)

        img = image.convert("RGB") if image.mode != "RGB" else image

        if tta:
            transforms = [
                self.transform,
                TTA_FLIP_TRANSFORM,
                TTA_SCALE_TRANSFORM,
                TTA_NATIVE_TRANSFORM,
            ]
            probs_list = []
            for tf in transforms:
                t = tf(img).unsqueeze(0).to(self.device)
                logits = self.model(t)
                probs_list.append(torch.softmax(logits, dim=-1).squeeze(0))
            probs = torch.stack(probs_list, dim=0).mean(dim=0)
        else:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        return self._build_result_from_probs(probs)

    # ---------- 带 Grad-CAM ----------

    def predict_with_cam(self, image: Image.Image, tta: bool = False) -> dict:
        """带 Grad-CAM 热力图的识别，热力图解释最终（可含 TTA）预测类别。"""
        # 反色预处理：保证热力图叠加在修正后的白底黑字图上
        image, _ = detect_and_fix_inversion(image)
        img = image.convert("RGB") if image.mode != "RGB" else image
        result = self.recognize(img, tta=tta)

        # Grad-CAM（惰性导入，缺库不影响普通识别）
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            # TTA 可能改变获胜类别；热力图在原图上解释最终选中的书法家。
            top_idx = self.label_to_id[result["calligrapher"]]
            target_layers = self._get_target_layers()
            targets = [ClassifierOutputTarget(top_idx)]
            cam = GradCAM(
                model=self.model,
                target_layers=target_layers,
                reshape_transform=self._get_cam_reshape_transform(),
            )
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

            # 从实际输入恢复底图，保留与热力图相同的缩放/裁剪坐标。
            img_np = cam_background(img_tensor)
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            # 转 base64
            pil_cam = Image.fromarray(cam_image)
            buf = BytesIO()
            pil_cam.save(buf, format="PNG")
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # 关注区域说明（3x3 网格）
            h, w = grayscale_cam.shape
            grid_h, grid_w = h // 3, w // 3
            region_names = [
                ["左上", "中上", "右上"],
                ["左中", "中央", "右中"],
                ["左下", "中下", "右下"],
            ]
            grid_scores = []
            for i in range(3):
                for j in range(3):
                    region = grayscale_cam[i * grid_h:(i + 1) * grid_h,
                                          j * grid_w:(j + 1) * grid_w]
                    grid_scores.append((region_names[i][j], float(region.mean())))

            grid_scores.sort(key=lambda x: x[1], reverse=True)
            top_regions = [name for name, score in grid_scores[:3] if score > 0.3]

            if top_regions:
                attention_note = f"模型主要关注：{'、'.join(top_regions)} 区域"
            else:
                attention_note = "模型关注区域较为分散"

            result["evidence"] = {
                "heatmap": heatmap_b64,
                "attention_note": attention_note,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            result["evidence"] = {
                "error": f"Grad-CAM 失败: {str(e)}",
                "attention_note": "热力图不可用",
            }

        return result

    # ---------- 批量推理 ----------

    @torch.no_grad()
    def recognize_batch(self, images: list, tta: bool = False) -> list:
        """
        批量识别
        Args:
            images: list[PIL.Image]
            tta:    是否启用 TTA
        Returns:
            list[dict]
        """
        self._load_model()

        if not images:
            return []

        if tta:
            # 逐张做 TTA（四路平均）
            results = []
            for img in images:
                results.append(self.recognize(img, tta=True))
            return results

        # 普通批量
        tensors = []
        for img in images:
            # 反色预处理：拓印黑底白字 → 白底黑字
            img, _ = detect_and_fix_inversion(img)
            img = img.convert("RGB") if img.mode != "RGB" else img
            tensors.append(self.transform(img))

        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        logits = self.model(batch_tensor)
        probs = torch.softmax(logits, dim=-1)

        results = []
        for i in range(len(images)):
            results.append(self._build_result_from_probs(probs[i]))
        return results


# ====================== 应用测试 ======================

if __name__ == "__main__":
    # 初始化识别器（默认路径与 train.py 的 output_dir 一致）
    recognizer = CalligrapherRecognizer(
        model_path="./checkpoints_test_gelu/calligrapher_classifier.pth"
    )

    print("===== 单张图片识别示例 =====")
    test_img_path = "./dataset_total/dataset0_new/褚遂良-楷/买.png"  # 修改为你的测试图片路径
    if os.path.exists(test_img_path):
        test_img = Image.open(test_img_path)
        result = recognizer.recognize(test_img, tta=False)
        print(f"识别结果: {result['calligrapher']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"所有类别概率: {result['all_probabilities']}\n")

        # 带 TTA 的对比
        result_tta = recognizer.recognize(test_img, tta=True)
        print(f"[TTA] 识别结果: {result_tta['calligrapher']}")
        print(f"[TTA] 置信度: {result_tta['confidence']:.4f}\n")

        # 带 Grad-CAM
        result_cam = recognizer.predict_with_cam(test_img)
        print(f"[CAM] {result_cam['calligrapher']} | {result_cam['evidence'].get('attention_note')}")
    else:
        print(f"[警告] 测试图片不存在: {test_img_path}")

    # 批量识别示例
    print("\n===== 批量识别示例 =====")
    batch_dir = "./dataset_total/dataset0_new/褚遂良-楷"
    if os.path.isdir(batch_dir):
        files = [f for f in os.listdir(batch_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][:5]
        batch_imgs = [Image.open(os.path.join(batch_dir, f)) for f in files]
        batch_results = recognizer.recognize_batch(batch_imgs, tta=False)
        for fname, r in zip(files, batch_results):
            print(f"  {fname} → {r['calligrapher']} ({r['confidence']:.4f})")
    else:
        print(f"[警告] 批量目录不存在: {batch_dir}")
