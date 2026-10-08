import base64
import re
import sys
import tempfile
import unittest
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from zhimo.application import report

# Initialize PDF/JPEG plugins before any other test temporarily replaces sys.modules.
Image.init()


class ReportPaginationTests(unittest.TestCase):
    def test_markdown_is_cleaned_for_report_text(self):
        source = """# 结论

**重点**：[来源](https://example.com)
- 第一项
1. 第二项
> 引用内容
```python
print('hidden fence')
```
| 字段 | 值 |
| --- | --- |
| 风格 | 行书 |
"""
        cleaned = report.markdown_to_report_text(source)
        self.assertEqual(
            cleaned,
            "结论\n\n重点：来源\n• 第一项\n• 第二项\n引用内容\nprint('hidden fence')\n字段    值\n风格    行书",
        )
        self.assertNotIn("**", cleaned)
        self.assertNotIn("[来源]", cleaned)
        self.assertNotIn("```", cleaned)

    def render(self, original, processed, result, *, examples=None):
        runs, pictures, pages = [], [], []
        original_text = ImageDraw.ImageDraw.text
        original_paste = Image.Image.paste
        original_save = Image.Image.save
        page_size = (1654, 2339)

        def draw_text(draw, xy, text, *args, **kwargs):
            if draw._image.size == page_size:
                bounds = draw.textbbox(xy, text, font=kwargs.get("font"))
                if str(text).strip():
                    self.assertGreaterEqual(bounds[0], 70, (text, bounds))
                    self.assertGreaterEqual(bounds[1], 70, (text, bounds))
                    self.assertLessEqual(bounds[2], page_size[0] - 70, (text, bounds))
                    self.assertLessEqual(bounds[3], page_size[1] - 70, (text, bounds))
                runs.append((draw._image, str(text), bounds))
            return original_text(draw, xy, text, *args, **kwargs)

        def paste(image, source, box=None, mask=None):
            if image.size == page_size and isinstance(source, Image.Image):
                x, y = box[:2]
                self.assertGreaterEqual(x, 70)
                self.assertGreaterEqual(y, 70)
                self.assertLessEqual(x + source.width, page_size[0] - 70)
                self.assertLessEqual(y + source.height, page_size[1] - 70)
                pictures.append((image, source.size, box))
            return original_paste(image, source, box, mask)

        def save(image, destination, format=None, **params):
            if format == "PDF":
                pages.extend([image, *params.get("append_images", [])])
            return original_save(image, destination, format=format, **params)

        with ExitStack() as stack:
            stack.enter_context(patch.object(ImageDraw.ImageDraw, "text", draw_text))
            stack.enter_context(patch.object(Image.Image, "paste", paste))
            stack.enter_context(patch.object(Image.Image, "save", save))
            pdf = report.build_pdf_report(original, processed, result, examples=examples)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertEqual(len(re.findall(rb"/Type /Page\b", pdf)), len(pages))
        visible_pages = {id(page) for page, text, _ in runs if text.strip()}
        visible_pages.update(id(page) for page, _, _ in pictures)
        self.assertEqual({id(page) for page in pages}, visible_pages, "PDF has an empty trailing page")
        return pages, runs, pictures

    @staticmethod
    def text(runs):
        return "".join(text for _, text, _ in runs)

    def test_long_text_only_reply_is_complete_and_paginated(self):
        reply = "正文" * 4000 + "正文最后一句。"
        pages, runs, pictures = self.render(None, None, {"reply": reply})
        self.assertGreater(len(pages), 1)
        self.assertIn(reply, self.text(runs))
        self.assertIn("本轮未附图片", self.text(runs))
        self.assertEqual(pictures, [])

    def test_long_summary_details_and_quality_are_not_clipped(self):
        summary = "摘要" * 2200 + "摘要结束。"
        author = "书法家说明" * 500 + "作者结束。"
        reliability = "可靠性依据" * 500 + "可靠性结束。"
        quality = "质量细节" * 500 + "质量结束。"
        image = Image.new("RGB", (100, 100), "black")
        pages, runs, pictures = self.render(image, image, {
            "summary": summary,
            "recognition": {"calligrapher": author},
            "reliability": reliability,
            "quality": {"detail": quality},
        })
        self.assertGreaterEqual(len(pages), 3)
        for value in [summary, author, reliability, quality]:
            self.assertIn(value, self.text(runs))
        self.assertEqual(len(pictures), 2)
        self.assertIs(pictures[0][0], pictures[1][0])

    def test_cam_moves_with_its_heading_and_long_knowledge_continues(self):
        image = Image.new("RGB", (100, 100), "black")
        heatmap = BytesIO()
        image.save(heatmap, format="PNG")
        knowledge = "知识依据" * 1400 + "https://example.com/" + "W" * 3000 + "SOURCE_END"
        pages, runs, pictures = self.render(image, image, {
            "summary": "\n".join(f"正文第 {index} 行" for index in range(38)),
            "recognition": {"evidence": {"heatmap": base64.b64encode(heatmap.getvalue()).decode()}},
            "knowledge": knowledge,
        })
        self.assertGreaterEqual(len(pages), 3)
        self.assertIn(knowledge, self.text(runs))
        self.assertEqual(len(pictures), 3)
        cam_picture = pictures[-1]
        cam_heading = next(run for run in runs if run[1] == "Grad-CAM 关注区域")
        self.assertIs(cam_heading[0], cam_picture[0])
        self.assertIsNot(cam_picture[0], pictures[0][0])
        self.assertLessEqual(cam_heading[2][3], cam_picture[2][1])

    def test_long_example_title_and_url_continue_without_overlapping_next_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "example.png"
            Image.new("RGB", (100, 100), "black").save(image_path)
            title = "作品标题" * 300 + "标题结束。"
            url = "https://example.com/" + "W" * 5000 + "URL_END"
            pages, runs, pictures = self.render(None, None, {"summary": "短摘要"}, examples=[
                {"path": image_path, "title": title, "source_url": url},
                {"path": image_path, "title": "下一幅作品", "source_url": "https://example.com/next"},
            ])
        self.assertGreaterEqual(len(pages), 3)
        self.assertIn(title, self.text(runs))
        self.assertIn(url, self.text(runs))
        self.assertIn("下一幅作品", self.text(runs))
        self.assertEqual(len(pictures), 2)
        next_picture_page, _, next_picture_box = pictures[-1]
        for page, text, bounds in runs:
            if page is next_picture_page and "下一幅作品" not in text and "https://example.com/next" not in text:
                self.assertLessEqual(bounds[3], next_picture_box[1])

    def test_four_short_examples_do_not_create_an_empty_last_page(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "example.png"
            Image.new("RGB", (100, 100), "black").save(image_path)
            pages, _, pictures = self.render(None, None, {"summary": "短摘要"}, examples=[
                {"path": image_path, "title": f"作品 {index}", "source_url": "https://example.com/"}
                for index in range(4)
            ])
        self.assertEqual(len(pages), 2)
        self.assertEqual(len(pictures), 4)

    def test_wrap_uses_pixel_width_and_preserves_chinese_words_and_urls(self):
        font = report._font(28)
        text = "中文内容 WWW iii https://example.com/" + "W" * 120
        lines = list(report._wrapped_lines(text, 150, font))
        self.assertEqual("".join(lines), text)
        for line in lines:
            left, _, right, _ = font.getbbox(line)
            self.assertLessEqual(right - left, 150)


if __name__ == "__main__":
    unittest.main()
