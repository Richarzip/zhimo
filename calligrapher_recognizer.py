"""
Calligrapher Recognizer - 书法家分类推理端
对应训练脚本: train.py (17位书法家分类)
功能: 输入单张/批量书法图片 → 输出书法家名称、置信度、所有类别概率
"""

import os
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as T


# 推理预处理
DEFAULT_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 模型定义
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


# 书法家识别器封装
class CalligrapherRecognizer:
    def __init__(
        self,
        model_path: str = "./checkpoints/calligrapher_classifier.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.transform = DEFAULT_TRANSFORM
        self.model_path = model_path
        
        # 模型和元数据
        self.model = None
        self.id_to_label = {}  # 索引 → 书法家中文名
        self.label_to_id = {}
        self.num_classes = 0
        self.backbone = ""
        self.best_acc = 0.0

    def _load_model(self):
        """懒加载模型，第一次调用recognize时才加载权重"""
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"权重文件不存在: {self.model_path}")

        # 加载权重和元数据
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
        print(f"  支持书法家: {', '.join(self.id_to_label.values())}\n")

        # 重建模型并加载权重
        self.model = build_model(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def recognize(self, image: Image.Image) -> dict:
        """
        识别单张书法图片
        Args:
            image: PIL.Image 格式的图片
        Returns:
            dict: 识别结果
        """
        self._load_model()

        # 预处理
        img = image.convert("RGB") if image.mode != "RGB" else image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 推理
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # 取Top1结果
        top_idx = probs.argmax().item()
        top_prob = probs[top_idx].item()
        calligrapher_name = self.id_to_label[top_idx]

        # 构建所有类别概率
        all_probs = {
            self.id_to_label[i]: round(probs[i].item(), 4)
            for i in range(self.num_classes)
        }

        return {
            "calligrapher": calligrapher_name,
            "confidence": round(top_prob, 4),
            "all_probabilities": all_probs,
            "num_classes": self.num_classes,
            "model_backbone": self.backbone
        }

    @torch.no_grad()
    def recognize_batch(self, images: list[Image.Image]):
        self._load_model()

        # 批量预处理与推理
        tensors = []
        for img in images:
            img = img.convert("RGB") if img.mode != "RGB" else img
            tensors.append(self.transform(img))
        batch_tensor = torch.stack(tensors).to(self.device)
        logits = self.model(batch_tensor)
        probs = torch.softmax(logits, dim=-1)

        results = []
        for i in range(len(images)):
            top_idx = probs[i].argmax().item()
            results.append({
                "calligrapher": self.id_to_label[top_idx],
                "confidence": round(probs[i][top_idx].item(), 4),
                "all_probabilities": {
                    self.id_to_label[j]: round(probs[i][j].item(), 4)
                    for j in range(self.num_classes)
                }
            })
        return results





# 简单识别器部分应用测试
if __name__ == "__main__":
    # 初始化识别器
    recognizer = CalligrapherRecognizer(
        model_path="./checkpoints/calligrapher_classifier.pth"
    )

    # 示例识别单张图片
    print(" 单张图片识别示例 ")
    test_img = Image.open("./dataset_total/dataset0_new/褚遂良-楷/买.png")  # **此处输入测试的图片路径**
    result = recognizer.recognize(test_img)
    print(f"识别结果: {result['calligrapher']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"所有类别概率: {result['all_probabilities']}\n")
