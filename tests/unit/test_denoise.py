"""Known-image checks for noise reduction and preservation of real strokes."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from zhimo.vision.denoise import denoise_image


def stroke_image():
    image = np.full((256, 256, 3), 230, np.uint8)
    cv2.line(image, (35, 35), (220, 35), (25, 25, 25), 1)
    cv2.line(image, (70, 20), (70, 230), (25, 25, 25), 2)
    cv2.line(image, (40, 200), (205, 65), (25, 25, 25), 6)
    cv2.rectangle(image, (150, 120), (210, 200), (25, 25, 25), 3)
    return image


def mse(first, second):
    return float(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))


class DenoiseTests(unittest.TestCase):
    def test_clean_image_preserves_all_pixels_including_one_pixel_strokes(self):
        clean = stroke_image()
        # Include a single punctuation-like dot; do not mistake one dot for noise.
        clean[240, 240] = 25
        output, info = denoise_image(Image.fromarray(clean))
        np.testing.assert_array_equal(np.asarray(output), clean)
        self.assertFalse(info["applied"])
        self.assertEqual(info["quality_before"], info["quality_after"])

    def test_isolated_noise_is_removed_without_erasing_thin_lines(self):
        clean = stroke_image()
        noisy = clean.copy()
        # Known background positions, well away from all strokes.
        noisy[215:250:4, 90:240:4] = 0
        output, info = denoise_image(Image.fromarray(noisy))
        self.assertLess(mse(output, clean), mse(noisy, clean) * 0.05)
        np.testing.assert_array_equal(np.asarray(output)[35, 35:221], clean[35, 35:221])
        self.assertIn("selective_median", info["steps"])
        self.assertEqual(info["stroke_change"], 0.0)

    def test_gaussian_noise_reduced_against_known_clean_image(self):
        clean = stroke_image()
        noisy = np.clip(clean.astype(float) + np.random.default_rng(42).normal(0, 12, clean.shape), 0, 255).astype(np.uint8)
        original = noisy.copy()
        output, info = denoise_image(Image.fromarray(noisy))
        self.assertLess(mse(output, clean), mse(noisy, clean) * 0.7)
        self.assertIn("protected_bilateral", info["steps"])
        self.assertLess(info["quality_after"]["noise_sigma"], info["quality_before"]["noise_sigma"])
        self.assertGreater(info["stroke_retention"], 0.99)
        np.testing.assert_array_equal(noisy, original)
        # The same input must produce byte-for-byte reproducible output/metadata.
        again, metadata = denoise_image(Image.fromarray(noisy))
        np.testing.assert_array_equal(output, again)
        self.assertEqual(info, metadata)
        json.dumps(info, allow_nan=False)

    def test_shading_is_reduced_and_original_ink_locations_remain_dark(self):
        clean = stroke_image()
        shaded = np.rint(clean * np.linspace(0.7, 1.0, clean.shape[1])[None, :, None]).astype(np.uint8)
        output, info = denoise_image(Image.fromarray(shaded))
        self.assertIn("background_normalization", info["steps"])
        self.assertLess(mse(output, clean), mse(shaded, clean) * 0.5)
        self.assertLess(info["quality_after"]["background_variation"], info["quality_before"]["background_variation"] * 0.5)
        self.assertTrue(np.all(np.asarray(output)[clean[..., 0] == 25] < 40))
        # Paper right beside a stroke must not retain a dark correction halo.
        pixels = np.asarray(output).astype(int)
        self.assertLessEqual(abs(pixels[100, 68, 0] - pixels[100, 64, 0]), 3)

    def test_faint_ink_contrast_increases_with_bounded_pixel_change(self):
        faded = np.rint(195 + (stroke_image().astype(float) - 25) * 40 / 205).astype(np.uint8)
        output, info = denoise_image(Image.fromarray(faded))
        self.assertIn("mild_contrast", info["steps"])
        self.assertGreater(info["quality_after"]["ink_contrast"], info["quality_before"]["ink_contrast"])
        self.assertLess(np.max(np.abs(np.asarray(output).astype(int) - faded)), 16)

    def test_rubbing_polarity_is_preserved_and_processing_is_symmetric(self):
        noisy = stroke_image()
        noisy[215:250:4, 90:240:4] = 0
        normal, _ = denoise_image(Image.fromarray(noisy))
        rubbing, info = denoise_image(Image.fromarray(255 - noisy))
        np.testing.assert_array_equal(np.asarray(rubbing), 255 - np.asarray(normal))
        self.assertTrue(info["polarity_normalized"])

    def test_guard_reverts_a_candidate_that_damages_ink(self):
        noisy = np.clip(stroke_image().astype(float) + np.random.default_rng(9).normal(0, 12, (256, 256, 3)), 0, 255).astype(np.uint8)
        with patch("zhimo.vision.denoise_impl.cv2.bilateralFilter", side_effect=lambda image, *args: np.full_like(image, 255)):
            output, info = denoise_image(Image.fromarray(noisy))
        np.testing.assert_array_equal(output, noisy)
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "stroke_guard_reverted")
        self.assertEqual(info["quality_before"], info["quality_after"])

    def test_blank_tiny_and_grayscale_inputs_stay_valid(self):
        for size in ((1, 1), (1, 20), (20, 1), (8, 8), (64, 64)):
            for color in (0, 128, 255):
                with self.subTest(size=size, color=color):
                    image = Image.new("L", size, color)
                    output, info = denoise_image(image)
                    self.assertEqual(output.mode, "RGB")
                    self.assertEqual(output.size, size)
                    np.testing.assert_array_equal(output, image.convert("RGB"))
                    json.dumps(info, allow_nan=False)

    def test_uniform_yellow_paper_is_not_whitened(self):
        image = Image.new("RGB", (100, 100), (235, 219, 178))
        output, info = denoise_image(image)
        np.testing.assert_array_equal(output, image)
        self.assertFalse(info["applied"])

    def test_exif_orientation_applied_only_once(self):
        image = Image.fromarray(stroke_image()[:150])
        image.getexif()[274] = 6
        expected = ImageOps.exif_transpose(image)
        output, _ = denoise_image(image)
        self.assertEqual(output.size, (150, 256))
        np.testing.assert_array_equal(output, expected)
        again, _ = denoise_image(output)
        np.testing.assert_array_equal(again, output)

    def test_basic_remains_available_but_default_preserves_clean_thin_strokes(self):
        clean = stroke_image()
        basic, info = denoise_image(Image.fromarray(clean), mode="basic")
        adaptive, _ = denoise_image(Image.fromarray(clean))
        self.assertTrue(info["applied"])
        self.assertEqual(info["mode"], "basic")
        self.assertGreater(mse(basic, clean), mse(adaptive, clean))
        self.assertNotIn("preserves_edges", info)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            denoise_image(Image.new("RGB", (8, 8)), mode="unknown")


if __name__ == "__main__":
    unittest.main()
