"""Exercise segmentation fallbacks with real image morphology."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from zhimo.vision import segmentation_impl as segmentation


class SegmentationTests(unittest.TestCase):
    @staticmethod
    def grid_image(count, gap=8):
        step = 20 + gap
        image = np.full((count * step, count * step, 3), 255, dtype=np.uint8)
        margin = gap // 2
        for row in range(count):
            for col in range(count):
                y, x = row * step + margin, col * step + margin
                image[y:y + 20, x:x + 20] = 0
        return image

    def test_dense_pages_keep_projection_results_above_thirty_characters(self):
        for count in (5, 6, 7):
            with self.subTest(count=count):
                result = segmentation.segment_auto(self.grid_image(count))
                self.assertEqual(result["cc_count"], 0)
                self.assertEqual(result["proj_count"], count * count)
                self.assertEqual(result["num_boxes"], count * count)
                self.assertTrue(result["method"].startswith("projection"))

    def test_long_page_keeps_valid_connected_components(self):
        result = segmentation.segment_auto(self.grid_image(6, gap=20))
        self.assertEqual(result["cc_count"], 36)
        self.assertEqual(result["num_boxes"], 36)
        self.assertTrue(result["method"].startswith("connected_components"))

    def test_real_page_does_not_select_projection_that_omits_lower_text(self):
        import cv2
        from PIL import Image

        with Image.open(ROOT / "image_test" / "image.png") as image:
            array = np.asarray(image.convert("RGB"))
        result = segmentation.segment_auto(array)
        self.assertGreater(result["cc_count"], 30)
        self.assertGreater(result["proj_count"], 30)
        height, width = array.shape[:2]
        covered = np.zeros((height, width), dtype=bool)
        for x, y, w, h in result["boxes"]:
            covered[y:y + h, x:x + w] = True
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # The old 33-box choice covered just 13.9% of ink and stopped at 27% height.
        self.assertGreater(np.count_nonzero(ink[covered]) / np.count_nonzero(ink), 0.75)
        self.assertGreater(max(y + h for x, y, w, h in result["boxes"]), height * 0.8)

    def test_single_detected_region_is_not_discarded(self):
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[4:36, 4:36] = 0
        result = segmentation.segment_auto(image)
        self.assertEqual(result["cc_count"], 0)
        self.assertEqual(result["proj_count"], 1)
        self.assertEqual(result["num_boxes"], 1)

    def test_empty_image_stays_empty(self):
        result = segmentation.segment_auto(np.full((80, 80, 3), 255, dtype=np.uint8))
        self.assertEqual(result["num_boxes"], 0)
        self.assertEqual(result["boxes"], [])

    def test_merged_component_does_not_hide_a_long_projection(self):
        image = self.grid_image(6)
        with patch.object(segmentation, "segment_by_connected_components", return_value=[(0, 0, 168, 168)]):
            result = segmentation.segment_auto(image)
        self.assertEqual(result["num_boxes"], 36)
        self.assertEqual(result["method"], "projection(fallback)")


if __name__ == "__main__":
    unittest.main()
