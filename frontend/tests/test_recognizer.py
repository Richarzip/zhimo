"""Exercise the original inference methods without model weights or PyTorch.

Only the tensor backend and Grad-CAM integration are substituted. In particular,
the production four-view averaging and class selection run in these tests.
"""

import ast
import base64
from io import BytesIO
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

<<<<<<< HEAD
=======
from image_preprocess import detect_and_fix_inversion

>>>>>>> origin/训练

class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.values, axis=dim))

    def squeeze(self, dim):
        return Tensor(np.squeeze(self.values, axis=dim))

    def to(self, device):
        return self

    def mean(self, dim):
        return Tensor(self.values.mean(axis=dim))

    def argmax(self):
        return self.values.argmax()

    def __getitem__(self, index):
        return self.values[index]


def softmax(tensor, dim):
    shifted = tensor.values - tensor.values.max(axis=dim, keepdims=True)
    values = np.exp(shifted)
    return Tensor(values / values.sum(axis=dim, keepdims=True))


def load_inference_methods():
    path = Path(__file__).resolve().parents[2] / "calligrapher_recognizer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    recognizer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CalligrapherRecognizer"
    )
    required = {
        "_sorted_class_ids", "_build_result_from_probs", "recognize", "predict_with_cam",
    }
    recognizer.body = [node for node in recognizer.body if node.name in required]
    scope = {
        "np": np,
        "Image": Image,
        "BytesIO": BytesIO,
        "base64": base64,
        "torch": types.SimpleNamespace(
            Tensor=Tensor,
            no_grad=lambda: lambda function: function,
            softmax=softmax,
            stack=lambda tensors, dim: Tensor(np.stack([t.values for t in tensors], axis=dim)),
        ),
        "TTA_FLIP_TRANSFORM": lambda image: Tensor([1]),
        "TTA_SCALE_TRANSFORM": lambda image: Tensor([2]),
        "TTA_NATIVE_TRANSFORM": lambda image: Tensor([3]),
<<<<<<< HEAD
=======
        # 反色预处理（真实实现，只依赖 numpy/PIL）
        "detect_and_fix_inversion": detect_and_fix_inversion,
>>>>>>> origin/训练
    }
    exec(compile(ast.Module(body=[recognizer], type_ignores=[]), str(path), "exec"), scope)
    return scope["CalligrapherRecognizer"]


class CamTtaTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = load_inference_methods()()
        self.recognizer.device = "cpu"
        self.recognizer.id_to_label = {0: "书法家甲", 1: "书法家乙"}
        self.recognizer.label_to_id = {"书法家甲": 0, "书法家乙": 1}
        self.recognizer.num_classes = 2
        self.recognizer.backbone = "test"
        self.recognizer._load_model = Mock()
        self.recognizer._get_target_layers = Mock(return_value=["final_layer"])
        self.recognizer.transform = Mock(side_effect=lambda image: Tensor([0]))
        # The original image selects A; the three augmented views select B.
        self.recognizer.model = Mock(side_effect=[
            Tensor(np.log([[0.9, 0.1]])),
            Tensor(np.log([[0.1, 0.9]])),
            Tensor(np.log([[0.1, 0.9]])),
            Tensor(np.log([[0.1, 0.9]])),
        ])
        self.image = Image.new("RGB", (32, 32), "white")
        self.cam = Mock(return_value=np.ones((1, 9, 9), dtype=np.float32))
        self.grad_cam = Mock(return_value=self.cam)
        self.target = Mock(side_effect=lambda category: types.SimpleNamespace(category=category))
        self.cam_modules = {}
        for name in (
            "pytorch_grad_cam", "pytorch_grad_cam.utils",
            "pytorch_grad_cam.utils.image", "pytorch_grad_cam.utils.model_targets",
        ):
            self.cam_modules[name] = types.ModuleType(name)
        self.cam_modules["pytorch_grad_cam"].GradCAM = self.grad_cam
        self.cam_modules["pytorch_grad_cam.utils.model_targets"].ClassifierOutputTarget = self.target
        self.cam_modules["pytorch_grad_cam.utils.image"].show_cam_on_image = (
            lambda image, heatmap, use_rgb: np.zeros((224, 224, 3), dtype=np.uint8)
        )

    def test_default_uses_original_prediction_and_one_classification_forward(self):
        with patch.dict(sys.modules, self.cam_modules):
            result = self.recognizer.predict_with_cam(self.image)
        self.assertEqual(result["calligrapher"], "书法家甲")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(self.recognizer.model.call_count, 1)
        self.target.assert_called_once_with(0)
        self.assertTrue(base64.b64decode(result["evidence"]["heatmap"]).startswith(b"\x89PNG"))

    def test_tta_changes_prediction_and_heatmap_target_without_extra_classification(self):
        with patch.dict(sys.modules, self.cam_modules):
            result = self.recognizer.predict_with_cam(self.image, tta=True)
        self.assertEqual(result["calligrapher"], "书法家乙")
        self.assertEqual(result["confidence"], 0.7)
        self.assertEqual(result["all_probabilities"], {"书法家甲": 0.3, "书法家乙": 0.7})
        self.assertEqual(self.recognizer.model.call_count, 4)
        self.target.assert_called_once_with(1)
        self.assertEqual(self.cam.call_args.kwargs["targets"][0].category, 1)
        # The explanation is rendered against the original view.
        np.testing.assert_array_equal(self.cam.call_args.kwargs["input_tensor"].values, [[0]])
        self.assertIn("heatmap", result["evidence"])

    def test_missing_cam_dependency_retains_tta_prediction(self):
        with patch.dict(sys.modules, {"pytorch_grad_cam": None}), patch("traceback.print_exc"):
            result = self.recognizer.predict_with_cam(self.image, tta=True)
        self.assertEqual(result["calligrapher"], "书法家乙")
        self.assertEqual(result["confidence"], 0.7)
        self.assertEqual(self.recognizer.model.call_count, 4)
        self.assertIn("error", result["evidence"])
        self.assertNotIn("heatmap", result["evidence"])

    def test_cam_failure_retains_tta_prediction(self):
        self.cam.side_effect = RuntimeError("CAM backend failed")
        with patch.dict(sys.modules, self.cam_modules), patch("traceback.print_exc"):
            result = self.recognizer.predict_with_cam(self.image, tta=True)
        self.assertEqual(result["calligrapher"], "书法家乙")
        self.assertEqual(result["confidence"], 0.7)
        self.assertIn("CAM backend failed", result["evidence"]["error"])


if __name__ == "__main__":
    unittest.main()
