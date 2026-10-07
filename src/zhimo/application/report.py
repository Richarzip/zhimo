"""ReportLab PDF reports for identification results and conversations."""

from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image as PdfImage,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


_MARKDOWN_FENCE_RE = re.compile(r"^\s*(```+|~~~+).*$")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_MARKDOWN_LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
_MARKDOWN_QUOTE_RE = re.compile(r"^\s{0,3}>\s?")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 22 * mm
MARGIN_BOTTOM = 18 * mm

INK = colors.HexColor("#262522")
MUTED = colors.HexColor("#77736B")
ACCENT = colors.HexColor("#9E2A24")
NAVY = colors.HexColor("#263B4D")
PAPER = colors.HexColor("#F7F5F0")
LINE = colors.HexColor("#D9D3C8")
PANEL = colors.HexColor("#EFEBE3")
WHITE = colors.white

# Keep these helpers available for callers that used the old text-wrapping API.
# They are no longer used to draw PDF pages; ReportLab now owns layout and pagination.
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("C:/Windows/Fonts/Deng.ttf"),
    Path("C:/Windows/Fonts/msyh.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)


def _strip_inline_markdown(text: str) -> str:
    """Remove presentation syntax while keeping the user's visible wording."""
    text = html.unescape(str(text or ""))
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<((?:https?://|mailto:)[^>]+)>", r"\1", text)
    text = re.sub(r"(`+)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)([*_])(?!\s)(.*?)(?<!\s)\1", r"\2", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"\\([\\`*{}\[\]()#+.!_>\-])", r"\1", text)
    return text.strip()


def markdown_to_report_text(text: Any) -> str:
    """Convert common Markdown into readable report text without Markdown syntax."""
    source = str(text or "")
    if not re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s|>\s?|[-+*]\s+|\d+[.)]\s+|```|~~~|\|)"
        r"|(?:\*\*|__|~~|\[[^]]+\]\(|`)",
        source,
    ):
        return source
    lines: list[str] = []
    for raw_line in source.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _MARKDOWN_FENCE_RE.match(raw_line) or _MARKDOWN_TABLE_SEPARATOR_RE.match(raw_line):
            continue
        line = _MARKDOWN_HEADING_RE.sub("", raw_line)
        line = _MARKDOWN_QUOTE_RE.sub("", line)
        line = _MARKDOWN_LIST_RE.sub("• ", line)
        if "|" in line and line.strip().startswith("|"):
            line = line.strip().strip("|").replace("|", "  ")
        lines.append(_strip_inline_markdown(line))
    return "\n".join(lines).strip()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the legacy Pillow font helper used by older integrations/tests."""
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Return a letterboxed RGB copy for compatibility with old callers."""
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
                space = paragraph.rfind(" ", start, start + length)
                if space > start:
                    length = space - start + 1
            yield paragraph[start:start + length]
            start += length


def _line_height(text: str, font) -> int:
    _, top, _, bottom = font.getbbox(text)
    return max(getattr(font, "size", 16), bottom - top) + 8


def _register_cjk_font() -> str:
    """Register the first usable CJK font without making the module font-path bound."""
    name = "ZhimoCJK"
    if name in pdfmetrics.getRegisteredFontNames():
        return name
    for path in FONT_CANDIDATES:
        if not path.exists():
            continue
        try:
            kwargs = {"subfontIndex": 0} if path.suffix.lower() == ".ttc" else {}
            pdfmetrics.registerFont(TTFont(name, str(path), **kwargs))
            return name
        except Exception:
            continue
    # ReportLab ships a CID font mapping even when the host has no CJK font
    # file.  It keeps Chinese text searchable instead of silently dropping it.
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        return "Helvetica"


PDF_FONT = _register_cjk_font()


def _plain(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    text = html.escape(_plain(value, ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
    return Paragraph(text or "&nbsp;", style)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    common = {"fontName": PDF_FONT, "wordWrap": "CJK", "splitLongWords": 1}
    return {
        "title": ParagraphStyle("zhimo-title", parent=base, **common, fontSize=25, leading=31, textColor=NAVY, spaceAfter=5),
        "subtitle": ParagraphStyle("zhimo-subtitle", parent=base, **common, fontSize=9, leading=14, textColor=MUTED),
        "section": ParagraphStyle("zhimo-section", parent=base, **common, fontSize=15, leading=21, textColor=NAVY, spaceBefore=13, spaceAfter=7),
        "body": ParagraphStyle("zhimo-body", parent=base, **common, fontSize=10.5, leading=17, textColor=INK, spaceAfter=4),
        "small": ParagraphStyle("zhimo-small", parent=base, **common, fontSize=8.5, leading=13, textColor=MUTED, spaceAfter=2),
        "label": ParagraphStyle("zhimo-label", parent=base, **common, fontSize=8, leading=11, textColor=MUTED, alignment=TA_CENTER),
        "metric": ParagraphStyle("zhimo-metric", parent=base, **common, fontSize=16, leading=21, textColor=ACCENT, alignment=TA_CENTER),
        "chat_user": ParagraphStyle("zhimo-chat-user", parent=base, **common, fontSize=10.5, leading=17, textColor=NAVY),
        "chat_ai": ParagraphStyle("zhimo-chat-ai", parent=base, **common, fontSize=10.5, leading=17, textColor=ACCENT),
    }


class _ReportDoc(BaseDocTemplate):
    """A4 document with a paper theme and stable header/footer on every page."""

    def __init__(self, output: BytesIO, *, title: str):
        super().__init__(
            output,
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=title,
            author="智墨",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=self._draw_page)])

    def _draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(PAPER)
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.6)
        canvas.line(self.leftMargin, PAGE_HEIGHT - 13 * mm, PAGE_WIDTH - self.rightMargin, PAGE_HEIGHT - 13 * mm)
        canvas.setFont(PDF_FONT, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, PAGE_HEIGHT - 10 * mm, "智墨 · 书法识别报告")
        canvas.drawString(self.leftMargin, 9 * mm, "模型辅助分析结果，不构成艺术史鉴定结论")
        canvas.drawRightString(PAGE_WIDTH - self.rightMargin, 9 * mm, f"{doc.page}")
        canvas.restoreState()


def _section(title: str, number: str, styles: dict[str, ParagraphStyle]):
    return [
        _paragraph(f"{number}  {title}", styles["section"]),
        HRFlowable(width="100%", thickness=0.8, color=ACCENT, spaceBefore=0, spaceAfter=6),
    ]


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return _plain(value)


def _confidence_level(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未评级"
    if number >= 0.85:
        return "高"
    if number >= 0.65:
        return "中"
    return "低"


def _compact(value: Any, length: int = 16) -> str:
    text = _plain(value)
    return text if len(text) <= length else text[: length - 1] + "…"


def _metric_panel(recognition: dict[str, Any], result: dict[str, Any], styles: dict[str, ParagraphStyle]):
    ensemble = recognition.get("ensemble") or {}
    agreement = ensemble.get("agreement", "-")
    if agreement is True:
        agreement = "一致"
    elif agreement is False:
        agreement = "不一致"
    confidence = recognition.get("confidence")
    rows = [
        [_paragraph(item, styles["label"]) for item in ("推测书法家", "置信度", "判断等级", "模型一致性")],
        [_paragraph(_compact(recognition.get("calligrapher", "-")), styles["metric"]),
         _paragraph(_format_percent(confidence), styles["metric"]),
         _paragraph(_confidence_level(confidence), styles["metric"]),
         _paragraph(_compact(agreement), styles["metric"])],
    ]
    table = Table(rows, colWidths=[(PAGE_WIDTH - 2 * MARGIN_X) / 4] * 4, rowHeights=[9 * mm, 16 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    strategy = ensemble.get("strategy", "-")
    return [
        table,
        Spacer(1, 8),
        _paragraph(
            f"集成策略：{_plain(strategy)}  ·  可靠性：{_plain(result.get('reliability'))}",
            styles["small"],
        ),
    ]


def _detail_lines(recognition: dict[str, Any], result: dict[str, Any], styles):
    ensemble = recognition.get("ensemble") or {}
    agreement = ensemble.get("agreement", "-")
    if agreement is True:
        agreement = "一致"
    elif agreement is False:
        agreement = "不一致"
    return [
        _paragraph(f"书法家：{_plain(recognition.get('calligrapher'))}", styles["body"]),
        _paragraph(f"置信度：{_format_percent(recognition.get('confidence'))}", styles["body"]),
        _paragraph(f"可靠性依据：{_plain(result.get('reliability'))}", styles["body"]),
        _paragraph(f"集成策略：{_plain(ensemble.get('strategy'))}  ·  双模型一致：{_plain(agreement)}", styles["body"]),
    ]


def _image_bytes(image: Image.Image) -> BytesIO:
    stream = BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    stream.seek(0)
    return stream


def _pdf_image(image: Image.Image, width: float, height: float) -> PdfImage:
    copy = image.convert("RGB").copy()
    if copy.width <= 0 or copy.height <= 0:
        raise ValueError("image has no pixels")
    ratio = min(width / copy.width, height / copy.height)
    result = PdfImage(_image_bytes(copy), width=copy.width * ratio, height=copy.height * ratio)
    result.hAlign = "CENTER"
    return result


def _decode_image(value: Any) -> Image.Image | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        raw = value.split(",", 1)[1] if value.startswith("data:") else value
        with Image.open(BytesIO(base64.b64decode(raw, validate=True))) as source:
            image = source.convert("RGB")
            image.load()
            return image
    except Exception:
        return None


def _image_card(label: str, image: Image.Image, styles):
    content = [_paragraph(label, styles["small"]), Spacer(1, 3), _pdf_image(image, 76 * mm, 55 * mm)]
    card = Table([[content]], colWidths=[82 * mm])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return card


def _evidence_pair(original: Image.Image, processed: Image.Image, styles):
    table = Table(
        [[_image_card("证据 A · 用户上传图像", original, styles), _image_card("证据 B · 模型输入图像", processed, styles)]],
        colWidths=[84 * mm, 84 * mm],
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def _quality_table(quality: dict[str, Any], styles):
    labels = {
        "blur": "清晰度",
        "noise": "噪声水平",
        "contrast": "墨迹对比度",
        "illumination": "纸面均匀性",
        "detail": "细节质量",
    }
    rows = [[_paragraph("指标", styles["label"]), _paragraph("评估值", styles["label"])]]
    for key, value in quality.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        label = labels.get(key, str(key))
        value_text = _plain(value, "")
        # A single giant Paragraph inside a Table row cannot always be split
        # when the row starts near the page bottom.  Chunking only the table
        # cell preserves all content while giving Platypus safe split points.
        chunks = [value_text[index:index + 140] for index in range(0, len(value_text), 140)] or [""]
        for index, chunk in enumerate(chunks):
            rows.append([_paragraph(label if index == 0 else "", styles["body"]), _paragraph(chunk, styles["body"])])
    table = Table(rows, colWidths=[45 * mm, 120 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _character_cards(items: list[dict[str, Any]], processed: Image.Image, styles):
    cards = []
    for item in sorted(items, key=lambda value: float(value.get("confidence") or 0), reverse=True)[:3]:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x, y, width, height = (int(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(processed.width, x + max(0, width)), min(processed.height, y + max(0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = processed.crop((x1, y1, x2, y2))
        card_content = [
            _pdf_image(crop, 38 * mm, 34 * mm),
            Spacer(1, 2),
            _paragraph(f"{_plain(item.get('name'))}  ·  {_format_percent(item.get('confidence'))}", styles["label"]),
        ]
        card = Table([[card_content]], colWidths=[48 * mm])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("BOX", (0, 0), (-1, -1), 0.5, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        cards.append(card)
    if not cards:
        return None
    table = Table([cards], colWidths=[52 * mm] * len(cards))
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def _cam_block(evidence: dict[str, Any], styles):
    image = _decode_image(evidence.get("heatmap"))
    if image is None:
        return None
    content = _section("Grad-CAM 关注区域", "05", styles)
    note = evidence.get("attention_note")
    if note:
        content.append(_paragraph(note, styles["small"]))
    content.extend([Spacer(1, 4), _pdf_image(image, 125 * mm, 88 * mm)])
    return KeepTogether(content)


def _load_examples(examples: list[dict[str, Any]] | None):
    loaded = []
    for example in (examples or [])[:4]:
        try:
            with Image.open(example["path"]) as source:
                image = source.convert("RGB")
                image.load()
        except Exception:
            continue
        loaded.append((example, image))
    return loaded


def build_pdf_report(
    original: Image.Image | None,
    processed: Image.Image | None,
    result: dict[str, Any],
    *,
    examples: list[dict[str, Any]] | None = None,
) -> bytes:
    """Build a searchable, themed, paginated identification report."""
    styles = _styles()
    output = BytesIO()
    doc = _ReportDoc(output, title="智墨书法识别与可解释性报告")
    story = [
        _paragraph("智墨书法识别与可解释性报告", styles["title"]),
        _paragraph(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", styles["subtitle"]),
        Spacer(1, 7),
    ]
    has_image = original is not None and processed is not None
    introduction = (
        "本报告结合用户上传图像、传统图像预处理、双模型集成识别和可解释性证据生成。模型判断不是艺术史鉴定结论。"
        if has_image else "本轮未附图片，以下记录对话回复及工具返回的结果。"
    )
    story.extend([_paragraph(introduction, styles["body"]), Spacer(1, 6)])

    recognition = result.get("recognition") or {}
    story.extend(_section("识别结论", "01", styles))
    story.extend(_metric_panel(recognition, result, styles))
    story.extend(_detail_lines(recognition, result, styles))
    summary = markdown_to_report_text(result.get("summary") or result.get("reply")) or "未生成文本摘要。"
    story.extend([Spacer(1, 4), _paragraph(summary, styles["body"])])

    if has_image:
        story.append(KeepTogether(_section("图像证据", "02", styles) + [_evidence_pair(original, processed, styles)]))

    quality = result.get("quality") or {}
    if quality:
        story.append(KeepTogether(_section("图像质量", "03", styles) + [_quality_table(quality, styles)]))

    per_char = recognition.get("per_char_results") or []
    if has_image and per_char and processed is not None:
        cards = _character_cards(per_char, processed, styles)
        if cards is not None:
            story.append(KeepTogether(_section("识别关注字", "04", styles) + [cards]))

    cam = _cam_block(recognition.get("evidence") or {}, styles)
    if cam is not None:
        story.append(cam)

    knowledge = markdown_to_report_text(result.get("knowledge"))
    if knowledge:
        story.extend(_section("知识库依据", "06", styles))
        story.append(_paragraph(knowledge, styles["body"]))

    loaded_examples = _load_examples(examples)
    if loaded_examples:
        story.append(PageBreak())
        story.extend(_section("参考作品", "07", styles))
        for example, image in loaded_examples:
            story.extend([
                _pdf_image(image, 78 * mm, 52 * mm),
                _paragraph(example.get("title") or "作品示例", styles["body"]),
                _paragraph(example.get("source_url") or "", styles["small"]),
                _paragraph(example.get("license") or "", styles["small"]),
                Spacer(1, 9),
            ])

    doc.build(story)
    return output.getvalue()


def build_chat_pdf(entries: list[dict[str, Any]]) -> bytes:
    """Build a searchable, themed, paginated conversation-record PDF."""
    styles = _styles()
    output = BytesIO()
    doc = _ReportDoc(output, title="智墨 · 书法鉴赏对话记录")
    story = [
        _paragraph("智墨 · 书法鉴赏对话记录", styles["title"]),
        _paragraph(f"导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}", styles["subtitle"]),
        Spacer(1, 10),
    ]
    for index, raw_entry in enumerate(entries, start=1):
        entry = {**raw_entry, "text": markdown_to_report_text(raw_entry.get("text") or "")}
        is_user = entry.get("role") == "user"
        role_text = "用户" if is_user else "智墨 AI"
        heading_style = ParagraphStyle(
            f"chat-heading-{index}",
            parent=styles["section"],
            fontName=PDF_FONT,
            fontSize=13,
            leading=18,
            textColor=NAVY if is_user else ACCENT,
            spaceBefore=8,
            spaceAfter=5,
            wordWrap="CJK",
        )
        story.append(_paragraph(f"{role_text}  ·  第 {index} 条", heading_style))
        image = _decode_image(entry.get("image"))
        if image is not None:
            story.extend([_pdf_image(image, 105 * mm, 65 * mm), Spacer(1, 4)])
        text = str(entry.get("text") or "").strip()
        if text:
            story.append(_paragraph(text, styles["chat_user"] if is_user else styles["chat_ai"]))
        recognition = entry.get("recognition") or {}
        if recognition.get("calligrapher"):
            story.append(_paragraph(
                f"识别结果：{recognition['calligrapher']} · 置信度 {_plain(recognition.get('confidence'))}",
                styles["small"],
            ))
        story.extend([HRFlowable(width="100%", thickness=0.35, color=LINE, spaceBefore=5, spaceAfter=8)])
    doc.build(story)
    return output.getvalue()
