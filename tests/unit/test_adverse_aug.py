"""Regression checks for the physical direction and boundaries of augmentations."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.adverse_aug import aug_defect_cut, aug_yellowing


class YellowingTests(unittest.TestCase):
    def test_neutral_paper_turns_yellow_and_keeps_dark_ink_distinct(self):
        image = np.array([[[0, 0, 0], [128, 128, 128], [220, 220, 220], [255, 255, 255]]], dtype=np.uint8)
        original = image.copy()
        for severity in (0.25, 0.5, 1.0):
            with self.subTest(severity=severity):
                result = aug_yellowing(image, severity)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.shape, image.shape)
                for red, green, blue in result[0, 1:]:
                    self.assertGreaterEqual(int(red), int(green))
                    self.assertGreater(int(green), int(blue))
                self.assertLess(result[0, 0].mean(), result[0, -1].mean() / 4)
        np.testing.assert_array_equal(image, original)

    def test_zero_severity_preserves_every_channel_exactly(self):
        image = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
        np.testing.assert_array_equal(aug_yellowing(image, 0), image)

    def test_increasing_severity_strengthens_the_yellow_cast(self):
        image = np.full((2, 3, 3), 255, dtype=np.uint8)
        colors = [aug_yellowing(image, severity)[0, 0].astype(int) for severity in (0, 0.25, 0.5, 1)]
        self.assertTrue(all(a[2] > b[2] for a, b in zip(colors, colors[1:])))
        self.assertTrue(all(a[0] - a[2] < b[0] - b[2] for a, b in zip(colors, colors[1:])))


class DefectCutTests(unittest.TestCase):
    @staticmethod
    def image(height, width):
        return (np.arange(height * width * 3).reshape(height, width, 3) % 251).astype(np.uint8)

    def test_zero_severity_and_small_images_never_erase_pixels(self):
        for shape, severity in (((40, 60), 0), ((10, 10), 0.5), ((1, 1), 1.0)):
            for edge in ("top", "bottom", "left", "right"):
                with self.subTest(shape=shape, severity=severity, edge=edge):
                    image = self.image(*shape)
                    with patch("training.adverse_aug.random.choice", return_value=edge):
                        result = aug_defect_cut(image, severity)
                    np.testing.assert_array_equal(result, image)

    def test_zero_cut_is_checked_on_the_selected_axis_of_rectangles(self):
        for shape, edges in (((10, 100), ("top", "bottom")), ((100, 10), ("left", "right"))):
            for edge in edges:
                with self.subTest(shape=shape, edge=edge):
                    image = self.image(*shape)
                    with patch("training.adverse_aug.random.choice", return_value=edge):
                        result = aug_defect_cut(image, 0.5)
                    np.testing.assert_array_equal(result, image)

    def test_positive_cut_only_changes_the_requested_edge(self):
        image = self.image(30, 50)
        original = image.copy()
        regions = {"top": (slice(0, 2), slice(None)), "bottom": (slice(-2, None), slice(None)),
                   "left": (slice(None), slice(0, 4)), "right": (slice(None), slice(-4, None))}
        for edge, region in regions.items():
            with self.subTest(edge=edge), patch("training.adverse_aug.random.choice", return_value=edge):
                result = aug_defect_cut(image, 0.5)
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask[region] = True
                np.testing.assert_array_equal(result[~mask], image[~mask])
                self.assertEqual(len(np.unique(result[mask], axis=0)), 1)
                self.assertTrue(np.any(result[mask] != image[mask]))
        np.testing.assert_array_equal(image, original)

    def test_zero_other_axis_does_not_disable_a_valid_cut(self):
        for shape, edge, region in (
            ((10, 100), "right", (slice(None), slice(-9, None))),
            ((100, 10), "bottom", (slice(-9, None), slice(None))),
        ):
            with self.subTest(shape=shape, edge=edge):
                image = self.image(*shape)
                with patch("training.adverse_aug.random.choice", return_value=edge):
                    result = aug_defect_cut(image, 0.5)
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask[region] = True
                np.testing.assert_array_equal(result[~mask], image[~mask])
                self.assertTrue(np.any(result[mask] != image[mask]))


if __name__ == "__main__":
    unittest.main()
