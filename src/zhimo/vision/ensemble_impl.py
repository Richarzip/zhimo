"""
Ensemble Recognizer - 多模型软投票集成推理端

把 convnext.pth 与 swin.pth 两个权重做软投票（soft voting）集成：
  1. 各自 forward 得到 softmax 概率
  2. 对两个概率向量取平均
  3. 取平均后概率最大的类别作为集成结果

不训练、不微调，仅推理期集成。两个模型为同一训练任务的
convnext_tiny（best_acc 0.8515）与 swin_tiny（best_acc 0.8481），
类别顺序已验证完全一致，可直接按索引平均。

用法：
    from zhimo.vision import EnsembleRecognizer
    rec = EnsembleRecognizer()
    result = rec.recognize(img)          # 软投票结果（含两个子模型详情）
    rec.recognize_batch(imgs)            # 批量
"""

import torch
from PIL import Image

from .recognizer_impl import CalligrapherRecognizer, DEFAULT_TRANSFORM
from .preprocess_impl import detect_and_fix_inversion
from .cam_utils import cam_background


class EnsembleRecognizer:
    """软投票集成的书法家识别器。"""

    def __init__(
        self,
        model_paths=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if model_paths is None:
            model_paths = [
                "./checkpoints/convnext.pth",
                "./checkpoints/swin.pth",
            ]
        self.device = torch.device(device)
        self.model_paths = list(model_paths)
        self._members = None          # [CalligrapherRecognizer, ...]，懒加载
        self.num_classes = 0
        self.id_to_label = {}
        self.label_to_id = {}

    # ---------- 模型加载 ----------

    def _load(self):
        """懒加载两个子模型，并校验类别对齐"""
        if self._members is not None:
            return

        members = []
        for path in self.model_paths:
            rec = CalligrapherRecognizer(model_path=path, device=self.device)
            rec._load_model()
            members.append(rec)

        # 类别对齐校验：软投票要求所有模型共享同一类别顺序
        first_seq = [members[0].id_to_label[i]
                     for i in sorted(members[0].id_to_label.keys())]
        for rec in members[1:]:
            seq = [rec.id_to_label[i] for i in sorted(rec.id_to_label.keys())]
            if seq != first_seq:
                raise ValueError(
                    "集成模型类别顺序不一致，无法软投票: "
                    f"{rec.model_path} vs {self.model_paths[0]}"
                )

        self._members = members
        self.num_classes = members[0].num_classes
        self.id_to_label = members[0].id_to_label
        self.label_to_id = members[0].label_to_id

        print(f"[集成识别器] 软投票加载成功（{len(members)} 个模型）")
        for rec in members:
            print(f"  - {rec.backbone}: best_acc={rec.best_acc:.4f}")
        print(f"  类别数: {self.num_classes}（顺序已验证一致）\n")

    @property
    def backbones(self):
        return [rec.backbone for rec in self._members]

    # ---------- 内部工具 ----------

    @torch.no_grad()
    def _member_probs(self, rec, img_tensor):
        """单个子模型 forward → 1D softmax 概率"""
        logits = rec.model(img_tensor)
        return torch.softmax(logits, dim=-1).squeeze(0).cpu()

    def _build_result(self, probs, member_probs):
        """把集成概率组装成返回 dict"""
        top_idx = probs.argmax().item()
        top_label = self.id_to_label[top_idx]

        members_detail = []
        agreement = True
        for rec, p in zip(self._members, member_probs):
            idx = p.argmax().item()
            if self.id_to_label[idx] != top_label:
                agreement = False
            members_detail.append({
                "backbone": rec.backbone,
                "calligrapher": self.id_to_label[idx],
                "confidence": round(p[idx].item(), 4),
            })

        return {
            "calligrapher": top_label,
            "confidence": round(probs[top_idx].item(), 4),
            "all_probabilities": {
                self.id_to_label[i]: round(probs[i].item(), 4)
                for i in sorted(self.id_to_label.keys())
            },
            "num_classes": self.num_classes,
            "model_backbone": "+".join(rec.backbone for rec in self._members),
            "ensemble": {
                "strategy": "soft_voting",
                "members": members_detail,
                "agreement": agreement,
            },
        }

    # ---------- 单张推理 ----------

    @torch.no_grad()
    def recognize(self, image: Image.Image, tta: bool = False) -> dict:
        """
        软投票识别单张书法图片（含反色校正）
        Args:
            image: PIL.Image
            tta:   每个子模型是否先做 TTA（四路平均）再参与投票
        Returns:
            dict: 集成识别结果，含 ensemble.members / agreement
        """
        self._load()

        # 反色预处理：拓印黑底白字 → 白底黑字
        image, _ = detect_and_fix_inversion(image)
        img = image.convert("RGB") if image.mode != "RGB" else image

        if tta:
            from .recognizer_impl import (
                TTA_FLIP_TRANSFORM,
                TTA_SCALE_TRANSFORM,
                TTA_NATIVE_TRANSFORM,
            )
            transforms = [
                DEFAULT_TRANSFORM,
                TTA_FLIP_TRANSFORM,
                TTA_SCALE_TRANSFORM,
                TTA_NATIVE_TRANSFORM,
            ]
            member_probs = []
            for rec in self._members:
                plist = []
                for tf in transforms:
                    t = tf(img).unsqueeze(0).to(self.device)
                    plist.append(self._member_probs(rec, t))
                member_probs.append(torch.stack(plist, dim=0).mean(dim=0))
        else:
            img_tensor = DEFAULT_TRANSFORM(img).unsqueeze(0).to(self.device)
            member_probs = [
                self._member_probs(rec, img_tensor) for rec in self._members
            ]

        # 软投票：加权平均（ConvNeXt 0.55 : Swin 0.45）
        probs = 0.55 * member_probs[0] + 0.45 * member_probs[1]
        return self._build_result(probs, member_probs)

    # ---------- 带 Grad-CAM ----------

    def predict_with_cam(self, image: Image.Image, tta: bool = False) -> dict:
        """带 Grad-CAM 热力图的集成识别。

        注意：本方法不能加 @torch.no_grad()——Grad-CAM 需要反向传播计算梯度。
        热力图用"代表成员"模型解释集成的最终类别：
        - 优先选与集成 top1 一致的成员模型；
        - 若两模型都不支持集成结论（罕见），用第一个成员模型。
        evidence 结构与 CalligrapherRecognizer.predict_with_cam 保持一致。
        """
        self._load()
        image, _ = detect_and_fix_inversion(image)
        img = image.convert("RGB") if image.mode != "RGB" else image
        result = self.recognize(img, tta=tta)

        try:
            import base64
            from io import BytesIO

            import numpy as np
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            top_label = result["calligrapher"]
            top_idx = self.label_to_id[top_label]

            # 选代表成员：与集成 top1 一致者优先，否则用第一个成员
            chosen = None
            for rec, m in zip(self._members, result["ensemble"]["members"]):
                if m["calligrapher"] == top_label:
                    chosen = rec
                    break
            if chosen is None:
                chosen = self._members[0]

            img_tensor = DEFAULT_TRANSFORM(img).unsqueeze(0).to(self.device)
            target_layers = chosen._get_target_layers()
            targets = [ClassifierOutputTarget(top_idx)]
            cam = GradCAM(
                model=chosen.model,
                target_layers=target_layers,
                reshape_transform=chosen._get_cam_reshape_transform(),
            )
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

            # 使用同一输入张量恢复底图，避免中心裁剪后叠图发生错位。
            img_np = cam_background(img_tensor)
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

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
                "cam_model": chosen.backbone,
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
        """批量软投票识别"""
        self._load()
        if not images:
            return []
        return [self.recognize(img, tta=tta) for img in images]


# ====================== 应用测试 ======================

if __name__ == "__main__":
    import os
    import random

    rec = EnsembleRecognizer()

    print("===== 单张图片识别示例 =====")
    test_path = "./image_test/test.png"
    if os.path.exists(test_path):
        img = Image.open(test_path)
        r = rec.recognize(img)
        print(f"集成结果: {r['calligrapher']} ({r['confidence']:.4f})")
        print(f"成员: {[(m['backbone'], m['calligrapher'], m['confidence']) for m in r['ensemble']['members']]}")
        print(f"投票一致性: {r['ensemble']['agreement']}")
    else:
        print(f"[警告] {test_path} 不存在")
