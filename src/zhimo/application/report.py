"""Dependency-light PDF report generation using Pillow."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def _wrapped_lines(text: str, width: int, font):
    """Wrap by rendered width, including Chinese and URLs without spaces."""
    for paragraph in str(text or "").splitlines() or [""]:
        if not paragraph:
            yield ""
            continue
        start = 0
        while start < len(paragraph):
            def fits(length):
                left, _, right, _ = font.getbbox(paragraph[start:start + length])
                return right - left <= width

            remaining = len(paragraph) - start
            low, high = 0, 1
            while high < remaining and fits(high):
                low = high
                high = min(remaining, high * 2)
            if fits(high):
                length = high
            else:
                while low + 1 < high:
                    middle = (low + high) // 2
                    if fits(middle):
                        low = middle
                    else:
                        high = middle
                length = max(1, low)
            if start + length < len(paragraph):
                # Prefer a word boundary without discarding whitespace or URL characters.
                space = paragraph.rfind(" ", start, start + length)
                if space > start:
                    length = space - start + 1
            yield paragraph[start:start + length]
            start += length


def _line_height(text: str, font) -> int:
    _, top, _, bottom = font.getbbox(text)
    return max(getattr(font, "size", 16), bottom - top) + 8


class _ReportLayout:
    width = 1654
    height = 2339
    margin = 70

    def __init__(self):
        self.pages: list[Image.Image] = []
        self.new_page()

    def new_page(self):
        self.page = Image.new("RGB", (self.width, self.height), "white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = self.margin
        self.pages.append(self.page)

    def ensure_space(self, height: int):
        if self.y + height > self.height - self.margin:
            self.new_page()

    def text(self, text: str, font, *, x=70, width=1500, fill=(25, 32, 48)):
        for line in _wrapped_lines(text, width, font):
            line_height = _line_height(line, font)
            self.ensure_space(line_height)
            left, top, _, _ = font.getbbox(line)
            # Align the visible glyph bounds to the requested position.
            self.draw.text((x - left, self.y - top), line, font=font, fill=fill)
            self.y += line_height

    def heading(self, title: str, font, *, keep_with=0, gap=9, fill=(20, 45, 90)):
        self.ensure_space(_line_height(title, font) + gap + keep_with)
        self.text(title, font, fill=fill)
        self.y += gap

    def image_pair(self, original, processed, heading_font, small_font):
        self.heading("输入图像与预处理结果", heading_font, keep_with=570)
        self.page.paste(_fit_image(original, 700, 500), (70, self.y))
        self.page.paste(_fit_image(processed, 700, 500), (884, self.y))
        self.draw.text((70, self.y + 510), "用户上传图像", font=small_font, fill=(80, 90, 105))
        self.draw.text((884, self.y + 510), "模型输入图像", font=small_font, fill=(80, 90, 105))
        self.y += 570

    def example(self, image, title, source_url, body_font, small_font):
        self.ensure_space(500)
        image_page, image_y = self.page, self.y
        self.page.paste(_fit_image(image, 700, 430), (70, image_y))
        self.y += 20
        self.text(title, body_font, x=884, width=650)
        self.y += 20
        self.text(source_url, small_font, x=884, width=650, fill=(80, 90, 105))
        if self.page is image_page:
            self.y = max(self.y, image_y + 430)
        self.y += 40


def build_pdf_report(
    original: Image.Image | None,
    processed: Image.Image | None,
    result: dict[str, Any],
    *,
    examples: list[dict[str, Any]] | None = None,
) -> bytes:
    """Build a human-readable report, paginating all text and visual evidence."""
    title_font = _font(42)
    heading_font = _font(28)
    body_font = _font(21)
    small_font = _font(16)
    layout = _ReportLayout()

    layout.text("智墨书法识别与可解释性报告", title_font, fill=(20, 45, 90))
    layout.y += 35
    has_image = original is not None and processed is not None
    introduction = ("本报告由用户上传图像、传统图像预处理、双模型集成识别和可解释性证据生成。模型判断不是艺术史鉴定结论。"
                    if has_image else "本轮未附图片，以下记录对话回复及工具返回的结果。")
    layout.text(introduction, body_font)
    layout.y += 25
    if has_image:
        layout.image_pair(original, processed, heading_font, small_font)

    recognition = result.get("recognition") or {}
    layout.heading("最终结果", heading_font, keep_with=_line_height("正文", body_font))
    summary = result.get("summary") or result.get("reply") or "未生成文本摘要。"
    layout.text(summary, body_font)
    layout.y += 18
    ensemble = recognition.get("ensemble") or {}
    details = [
        f"书法家：{recognition.get('calligrapher', '-')}",
        f"置信度：{recognition.get('confidence', '-')}",
        f"可靠性：{result.get('reliability', '-')}",
        f"集成策略：{ensemble.get('strategy', '-')}",
        f"双模型一致：{ensemble.get('agreement', '-')}",
    ]
    layout.text("\n".join(details), body_font)
    layout.y += 20
    quality = result.get("quality") or {}
    if quality:
        layout.text("图像质量：" + json.dumps(quality, ensure_ascii=False), small_font)

    # 识别关注字：从处理后图像按切分框裁出投票靠前的单字图，作为识别依据展示。
    # 替代原 Grad-CAM 热力图——热力图有时关注背景，展示出来反而削弱可解释性。
    per_char = recognition.get("per_char_results") or []
    if has_image and per_char and processed is not None:
        top_chars = sorted(
            per_char,
            key=lambda item: float(item.get("confidence") or 0),
            reverse=True,
        )[:3]
        layout.heading("识别关注字", heading_font, keep_with=840)
        for item in top_chars:
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, w, h = (int(v) for v in bbox)
            try:
                crop = processed.crop((x, y, x + w, y + h))
            except Exception:
                continue
            layout.ensure_space(280)
            layout.page.paste(_fit_image(crop, 300, 250), (70, layout.y))
            layout.draw.text(
                (70, layout.y + 258),
                f"{item.get('name', '-')}（{float(item.get('confidence') or 0):.3f}）",
                font=small_font, fill=(60, 66, 82),
            )
            layout.y += 280

    knowledge = result.get("knowledge")
    if knowledge:
        layout.heading("知识库依据", heading_font, keep_with=_line_height("正文", small_font))
        layout.text(str(knowledge), small_font)

    if examples:
        layout.new_page()
        layout.heading("书法家作品示例", heading_font, keep_with=500, gap=19)
        for example in examples[:4]:
            try:
                with Image.open(example["path"]) as source:
                    image = source.convert("RGB")
            except Exception:
                continue
            layout.example(
                image, str(example.get("title") or "作品示例"),
                str(example.get("source_url") or ""), body_font, small_font,
            )

    output = BytesIO()
    layout.pages[0].save(output, format="PDF", save_all=True, append_images=layout.pages[1:], resolution=150)
    return output.getvalue()


def build_chat_pdf(entries: list[dict[str, Any]]) -> bytes:
    """Build a conversation-record PDF from chat entries.

    由前端导出对话时调用：前端负责剥离 markdown 并把图片压缩为
    base64 data URL（<=600px），这里只负责确定性渲染，不做任何生成式处理。

    Each entry:
        {"role": "user" | "ai",
         "text": str,                       # 纯文本（前端已剥离 markdown 标记）
         "image": str | None,               # 可选 base64 data URL（用户消息的图）
         "recognition": {"calligrapher": str, "confidence": float} | None}
    """
    title_font = _font(42)
    heading_font = _font(24)
    body_font = _font(21)
    small_font = _font(16)
    layout = _ReportLayout()

    layout.text("智墨 · 书法鉴赏对话记录", title_font, fill=(20, 45, 90))
    layout.y += 12
    layout.text(
        f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        small_font, fill=(80, 90, 105),
    )
    layout.y += 24

    user_fill = (60, 66, 82)
    ai_fill = (176, 58, 46)

    for index, entry in enumerate(entries, start=1):
        role = entry.get("role")
        text = str(entry.get("text") or "").strip()
        if role == "user":
            layout.y += 10
            layout.heading(f"用户  ·  第 {index} 条", heading_font, gap=8, fill=user_fill)
            image = entry.get("image")
            if image:
                try:
                    raw = image.split(",", 1)[1] if image.startswith("data:") else image
                    with Image.open(BytesIO(base64.b64decode(raw))) as source:
                        pic = _fit_image(source.convert("RGB"), 600, 400)
                except Exception:
                    pic = None
                if pic is not None:
                    layout.ensure_space(410)
                    layout.page.paste(pic, (70, layout.y))
                    layout.y += 410
            if text:
                layout.text(text, body_font, fill=user_fill)
        else:
            layout.y += 10
            layout.heading(f"智墨 AI  ·  第 {index} 条", heading_font, gap=8, fill=ai_fill)
            if text:
                layout.text(text, body_font, fill=ai_fill)
            recognition = entry.get("recognition") or {}
            if recognition.get("calligrapher"):
                layout.text(
                    f"识别结果：{recognition['calligrapher']} · 置信度 {recognition.get('confidence', '-')}",
                    small_font, fill=(20, 45, 90),
                )
            layout.y += 14

    output = BytesIO()
    layout.pages[0].save(output, format="PDF", save_all=True, append_images=layout.pages[1:], resolution=150)
    return output.getvalue()
