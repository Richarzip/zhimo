"""Dependency-light PDF report generation using Pillow."""

from __future__ import annotations

import json
import textwrap
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


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int, font, fill=(25, 32, 48)) -> int:
    x, y = xy
    for paragraph in str(text or "").splitlines() or [""]:
        lines = textwrap.wrap(paragraph, width=max(1, width // max(1, int(font.size * 0.95)))) or [""]
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + 8
    return y


def build_pdf_report(
    original: Image.Image,
    processed: Image.Image,
    result: dict[str, Any],
    *,
    examples: list[dict[str, Any]] | None = None,
) -> bytes:
    """Build a human-readable report with visual evidence and raw results."""
    page_w, page_h = 1654, 2339
    title_font = _font(42)
    heading_font = _font(28)
    body_font = _font(21)
    small_font = _font(16)
    pages: list[Image.Image] = []

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        page = Image.new("RGB", (page_w, page_h), "white")
        return page, ImageDraw.Draw(page), 70

    page, draw, y = new_page()
    draw.text((70, y), "智墨书法识别与可解释性报告", font=title_font, fill=(20, 45, 90))
    y += 85
    y = _draw_wrapped(draw, "本报告由用户上传图像、传统图像预处理、双模型集成识别和可解释性证据生成。模型判断不是艺术史鉴定结论。", (70, y), 1500, body_font)
    y += 25
    draw.text((70, y), "输入图像与预处理结果", font=heading_font, fill=(20, 45, 90))
    y += 45
    left = _fit_image(original, 700, 500)
    right = _fit_image(processed, 700, 500)
    page.paste(left, (70, y))
    page.paste(right, (884, y))
    draw.text((70, y + 510), "用户上传图像", font=small_font, fill=(80, 90, 105))
    draw.text((884, y + 510), "模型输入图像", font=small_font, fill=(80, 90, 105))
    y += 570

    recognition = result.get("recognition") or {}
    draw.text((70, y), "最终结果", font=heading_font, fill=(20, 45, 90))
    y += 48
    summary = result.get("summary") or result.get("reply") or "未生成文本摘要。"
    y = _draw_wrapped(draw, summary, (70, y), 1500, body_font)
    y += 18
    ensemble = recognition.get("ensemble") or {}
    details = [
        f"书法家：{recognition.get('calligrapher', '-')}",
        f"置信度：{recognition.get('confidence', '-')}",
        f"可靠性：{result.get('reliability', '-')}",
        f"集成策略：{ensemble.get('strategy', '-')}",
        f"双模型一致：{ensemble.get('agreement', '-')}",
    ]
    y = _draw_wrapped(draw, "\n".join(details), (70, y), 1500, body_font)
    y += 20
    quality = result.get("quality") or {}
    if quality:
        y = _draw_wrapped(draw, "图像质量：" + json.dumps(quality, ensure_ascii=False), (70, y), 1500, small_font)

    heatmap = (recognition.get("evidence") or {}).get("heatmap")
    if heatmap:
        if isinstance(heatmap, str) and heatmap.startswith("data:"):
            import base64
            heatmap = base64.b64decode(heatmap.split(",", 1)[1])
        if isinstance(heatmap, str):
            import base64
            heatmap = base64.b64decode(heatmap)
        try:
            heat = Image.open(BytesIO(heatmap))
            y += 18
            draw.text((70, y), "Grad-CAM 关注区域", font=heading_font, fill=(20, 45, 90))
            page.paste(_fit_image(heat, 700, 430), (70, y + 45))
            y += 500
        except Exception:
            pass

    knowledge = result.get("knowledge")
    if knowledge:
        if y > page_h - 450:
            pages.append(page)
            page, draw, y = new_page()
        draw.text((70, y), "知识库依据", font=heading_font, fill=(20, 45, 90))
        y += 45
        _draw_wrapped(draw, str(knowledge), (70, y), 1500, small_font)

    if examples:
        pages.append(page)
        page, draw, y = new_page()
        draw.text((70, y), "书法家作品示例", font=heading_font, fill=(20, 45, 90))
        y += 55
        for example in examples[:4]:
            try:
                image = Image.open(example["path"])
                page.paste(_fit_image(image, 700, 430), (70, y))
                draw.text((884, y + 20), str(example.get("title") or "作品示例"), font=body_font, fill=(25, 32, 48))
                _draw_wrapped(draw, str(example.get("source_url") or ""), (884, y + 70), 650, small_font, fill=(80, 90, 105))
                y += 500
                if y > page_h - 500:
                    pages.append(page)
                    page, draw, y = new_page()
            except Exception:
                continue

    pages.append(page)
    output = BytesIO()
    pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    return output.getvalue()

