"""CAM integration tests using real timm models without pretrained weights.

The model tests skip when optional torch/timm/grad-cam dependencies are absent.
"""

import base64
from io import BytesIO
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import pytorch_grad_cam
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils import image as cam_image_utils
    from zhimo.vision.recognizer_impl import CalligrapherRecognizer, build_model
    from zhimo.vision.ensemble_impl import EnsembleRecognizer
except ImportError as exc:
    CAM_AVAILABLE = False
    CAM_SKIP_REASON = str(exc)
else:
    CAM_AVAILABLE = True
    CAM_SKIP_REASON = ""


@unittest.skipUnless(CAM_AVAILABLE, f"Optional CAM dependencies unavailable: {CAM_SKIP_REASON}")
class VisionCamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.thread_count)

    def setUp(self):
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)
        torch.manual_seed(0)
        pixels = np.full((256, 256, 3), 255, dtype=np.uint8)
        pixels[32:48, 32:48] = 0
        pixels[140:220, 150:180] = 0
        self.image = Image.fromarray(pixels)
        # This fixture needs only the center crop; the initial resize is identity.
        self.expected_background = pixels[16:240, 16:240].astype(np.float32) / 255.0

    def recognizer(self, backbone):
        recognizer = CalligrapherRecognizer(device="cpu")
        recognizer.model = build_model(2, backbone, pretrained=False).eval()
        recognizer.backbone = backbone
        recognizer.id_to_label = {0: "甲", 1: "乙"}
        recognizer.label_to_id = {"甲": 0, "乙": 1}
        recognizer.num_classes = 2
        return recognizer

    def run_cam(self, recognizer):
        shapes = []

        class InspectCAM(GradCAM):
            def get_cam_image(self, *args, **kwargs):
                result = super().get_cam_image(*args, **kwargs)
                shapes.append(result.shape)
                return result

        with patch.object(pytorch_grad_cam, "GradCAM", InspectCAM), patch.object(
            cam_image_utils, "show_cam_on_image", wraps=cam_image_utils.show_cam_on_image
        ) as overlay:
            result = recognizer.predict_with_cam(self.image)
        self.assertNotIn("error", result["evidence"])
        self.assertEqual(Image.open(BytesIO(base64.b64decode(result["evidence"]["heatmap"]))).size, (224, 224))
        np.testing.assert_allclose(overlay.call_args.args[0], self.expected_background, atol=1e-6)
        return result, shapes

    def test_supported_backbones_produce_spatial_maps_on_the_exact_input_view(self):
        for backbone, grid in (
            ("convnext_tiny", 7),
            ("swin_tiny_patch4_window7_224", 7),
            ("resnet18", 7),
            ("efficientnet_b0", 7),
            ("vit_tiny_patch16_224", 14),
        ):
            with self.subTest(backbone=backbone):
                recognizer = self.recognizer(backbone)
                _, shapes = self.run_cam(recognizer)
                self.assertEqual(shapes, [(1, grid, grid)])

    def test_swin_representative_uses_its_transform_in_the_ensemble(self):
        swin = self.recognizer("swin_tiny_patch4_window7_224")
        other = types.SimpleNamespace(backbone="other")
        ensemble = EnsembleRecognizer(model_paths=["first.pth", "second.pth"], device="cpu")
        ensemble._members = [other, swin]
        ensemble.id_to_label = swin.id_to_label
        ensemble.label_to_id = swin.label_to_id
        ensemble.num_classes = 2
        ensemble._member_probs = Mock(side_effect=[torch.tensor([0.6, 0.4]), torch.tensor([0.1, 0.9])])
        result, shapes = self.run_cam(ensemble)
        self.assertEqual(result["calligrapher"], "乙")
        self.assertEqual(result["evidence"]["cam_model"], swin.backbone)
        self.assertEqual(shapes, [(1, 7, 7)])

    def test_token_layout_removes_prefixes_and_preserves_rectangular_grids(self):
        recognizer = CalligrapherRecognizer(device="cpu")
        recognizer.backbone = "deit_test"
        recognizer.model = types.SimpleNamespace(encoder=types.SimpleNamespace(
            patch_embed=types.SimpleNamespace(grid_size=(2, 3)), num_prefix_tokens=2,
        ))
        tokens = torch.tensor([[[1000.0, 2000.0], [1000.0, 2000.0]] + [[float(i), float(i + 10)] for i in range(6)]])
        spatial = recognizer._get_cam_reshape_transform()(tokens)
        self.assertEqual(tuple(spatial.shape), (1, 2, 2, 3))
        torch.testing.assert_close(spatial[0, 0], torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
        torch.testing.assert_close(spatial[0, 1], torch.tensor([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]))


if __name__ == "__main__":
    unittest.main()
