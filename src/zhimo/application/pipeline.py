"""Framework-neutral image analysis pipelines.

The aiohttp adapter supplies the optional dependencies as callables. Keeping
those dependencies injectable makes the pipeline usable from HTTP, CLI, and
tests without importing the web framework.
"""

from __future__ import annotations

import base64
import json
import traceback
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Step:
    name: str
    status: str = "pending"
    detail: str = ""
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RunContext:
    steps: list[Step] = field(default_factory=list)

    def add(self, name: str, status: str = "running", detail: str = "") -> Step:
        step = Step(name=name, status=status, detail=detail)
        self.steps.append(step)
        return step

    def finish(self, step: Step, data: Any = None, detail: str = "") -> None:
        step.status = "ok"
        step.data = data
        if detail:
            step.detail = detail

    def fail(self, step: Step, exc: BaseException, classify: Callable[[BaseException], dict]) -> None:
        step.status = "error"
        step.error = str(exc)
        step.detail = classify(exc)["message"]
        step.data = classify(exc)


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def top_k(probabilities: dict[str, float], k: int = 3) -> list[dict[str, Any]]:
    return [
        {"name": name, "confidence": score}
        for name, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]
    ]


def reliability_from(quality: dict[str, float], confidence: float) -> str:
    q, c = float(quality.get("overall", 1.0)), float(confidence)
    if q < 0.3 and c > 0.7:
        return "high"
    if q < 0.5 or c > 0.5:
        return "medium"
    return "low"


def build_summary(result: dict[str, Any]) -> str:
    if result.get("blocked"):
        return (result.get("diagnostic") or {}).get("message", "分析流程未完成")
    recognition = result.get("recognition") or {}
    name = recognition.get("calligrapher")
    if not name:
        return "已完成图像预处理，但没有得到书法家识别结论。"
    parts = [f"本次本地 Agent 判断最可能的书法家为：{name}。"]
    if result.get("inversion", {}).get("was_inverted"):
        parts.append("检测到反色拓印，已自动反转后再识别。")
    if recognition.get("confidence") is not None:
        parts.append(
            f"模型置信度为 {recognition['confidence']}，综合可靠性评级为 {result.get('reliability', 'unknown')}。"
        )
    if result.get("quality"):
        q = result["quality"]
        parts.append(
            "图像质量评估："
            f"泛黄 {q.get('yellow')}、褪色 {q.get('fade')}、噪声 {q.get('noise')}、"
            f"模糊 {q.get('blur')}、综合退化 {q.get('overall')}。"
        )
    note = recognition.get("evidence", {}).get("attention_note")
    if note:
        parts.append(f"可解释性提示：{note}。")
    if result.get("knowledge"):
        parts.append("知识库已返回相关风格材料，可结合 Top-K 候选和热力图进一步解释。")
    return "".join(parts)


def overlay_boxes(image: Image.Image, boxes: list[list[int]]) -> str:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, box in enumerate(boxes, start=1):
        x, y, w, h = map(int, box)
        draw.rectangle([x, y, x + w, y + h], outline=(24, 119, 242), width=3)
        draw.text((x + 4, max(0, y - 17)), str(index), fill=(24, 119, 242))
    return image_to_data_url(canvas)


def _finish(result, ctx, original, processed, *, use_examples, use_pdf, finalize):
    result["summary"] = build_summary(result)
    result["steps"] = [step.to_dict() for step in ctx.steps]
    return finalize(result, original, processed, use_examples=use_examples, use_pdf=use_pdf)



def parse_tool_result(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"raw": content}
        except (json.JSONDecodeError, TypeError):
            return {"raw": content}
    return content if isinstance(content, dict) else {"raw": str(content)}


def run_chat(
    image: Image.Image | None,
    fields: dict[str, str],
    *,
    get_agent: Callable[[], Any],
    denoise: Callable[[Image.Image], tuple[Image.Image, dict]],
    classify: Callable[[BaseException], dict],
    finalize: Callable[..., dict],
) -> dict[str, Any]:
    """Run one conversational turn while keeping the session contract stable."""
    from langchain.messages import HumanMessage

    text = fields.get("text", "").strip() or fields.get("prompt", "").strip()
    if image is None and not text:
        raise ValueError("请输入问题或上传图片。")

    original = image.convert("RGB") if image is not None else None
    processed = original
    if processed is not None and fields.get("denoise", "true") == "true":
        processed, denoise_info = denoise(processed)
    else:
        denoise_info = {"enabled": False, "method": "no_image" if image is None else "disabled"}

    session_id = fields.get("session_id") or uuid.uuid4().hex
    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = agent.get_state(config)
    previous_messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    previous_ids = {message.id for message in previous_messages if getattr(message, "id", None)}
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    if processed is not None:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(processed, "PNG")}})
    turn = HumanMessage(content=content, id=uuid.uuid4().hex)

    try:
        # Supply a complete set on every turn so checkpointed options cannot leak
        # from a previous request or a different checkbox selection.
        options = {
            "rag": fields.get("rag", "true") == "true",
            "cam": fields.get("cam", "true") == "true",
            "tta": fields.get("tta", "false") == "true",
            "analysis_mode": fields.get("analysis_mode", "auto"),
        }
        result = agent.invoke({"messages": [turn], "analysis_options": options}, config=config)
    except Exception as exc:
        return {
            "session_id": session_id,
            "error": "agent_error",
            "message": f"Agent 执行失败：{type(exc).__name__}: {str(exc)[:200]}",
            "diagnostic": classify(exc),
            "reply": "抱歉，本次分析遇到异常。请稍后重试，或换一张图片 / 换一种问法。",
        }

    messages = result.get("messages", [])
    # Summarization can remove earlier history (even this turn's HumanMessage).
    # Message IDs survive checkpoint serialization, unlike list indices.
    turn_index = next(
        (index for index, message in enumerate(messages) if getattr(message, "id", None) == turn.id),
        None,
    )
    if turn_index is not None:
        new_messages = messages[turn_index + 1:]
    else:
        new_messages = [message for message in messages if getattr(message, "id", None) not in previous_ids]
    reply = ""
    steps: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for message in new_messages:
        message_type = getattr(message, "type", "")
        if message_type == "ai" and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                name = call.get("name", "tool")
                steps.append({"name": name, "status": "ok", "detail": f"调用工具 {name}"})
        elif message_type == "tool":
            tool_results.append(parse_tool_result(message.content))
            steps.append({"name": getattr(message, "name", "tool"), "status": "ok", "detail": "工具返回结果"})
        elif message_type == "ai" and getattr(message, "content", None):
            reply = message.content if isinstance(message.content, str) else str(message.content)

    recognition: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    reliability = None
    mode = "chat"
    inversion = None
    for item in tool_results:
        if not isinstance(item, dict) or item.get("error") or "calligrapher" not in item:
            continue
        mode = "single"
        recognition = {
            "calligrapher": item.get("calligrapher"),
            "confidence": item.get("confidence"),
            "top_k": item.get("top_k", []),
            "model_backbone": item.get("model_backbone"),
            "evidence": item.get("evidence", {}),
            "ensemble": item.get("ensemble"),
        }
        if "num_characters" in item:
            mode = "multi"
            recognition.update({key: item.get(key) for key in ("num_characters", "per_char_results", "consistency", "vote_distribution")})
        if "quality" in item:
            quality = item.get("quality", {})
            reliability = item.get("reliability")
        elif "consistency" in item:
            reliability = item.get("consistency")
        if item.get("inversion"):
            inversion = item["inversion"]

    response = {
        "session_id": session_id,
        "mode": mode,
        "reply": reply,
        "steps": steps,
        "recognition": recognition,
        "quality": quality,
        "reliability": reliability,
        "inversion": inversion,
        "message_count": len(messages),
        "denoise": denoise_info,
    }
    return finalize(response, original, processed,
                    use_examples=fields.get("examples", "false") == "true",
                    use_pdf=fields.get("pdf", "false") == "true")
def run_single(
    image: Image.Image,
    *,
    use_tta: bool,
    use_cam: bool,
    use_rag: bool,
    use_denoise: bool = False,
    use_examples: bool = False,
    use_pdf: bool = False,
    prompt: str = "",
    get_recognizer: Callable[[], Any],
    assess_quality: Callable[[np.ndarray], dict],
    detect_inversion: Callable[[Image.Image], tuple[Image.Image, dict]],
    denoise: Callable[[Image.Image], tuple[Image.Image, dict]],
    search_knowledge: Callable[..., Any] | None,
    classify: Callable[[BaseException], dict],
    finalize: Callable[..., dict],
) -> dict[str, Any]:
    ctx = RunContext()
    original = image.convert("RGB")
    result: dict[str, Any] = {"mode": "single"}
    rgb = original

    step = ctx.add("extract_image", detail="读取上传图片并转换为 RGB")
    if use_denoise:
        rgb, info = denoise(rgb)
        result["denoise"] = info
        ctx.finish(step, info, "图像去噪完成")
    else:
        ctx.finish(step, {"size": rgb.size, "preview": image_to_data_url(rgb)}, "图像读取完成")
        ctx.add("denoise", status="skipped", detail="用户关闭了去噪")

    step = ctx.add("detect_inversion", detail="检测反色拓印")
    rgb, inversion = detect_inversion(rgb)
    result["inversion"] = inversion
    ctx.finish(step, inversion, "已完成反色检测")

    step = ctx.add("assess_image_quality", detail="评估图像退化程度")
    quality = assess_quality(np.array(rgb))
    result["quality"] = quality
    ctx.finish(step, quality, "图像质量评估完成")

    step = ctx.add("analyze_calligraphy", detail="执行书法家分类推理")
    try:
        recognizer = get_recognizer()
        raw = recognizer.predict_with_cam(rgb, tta=use_tta) if use_cam else recognizer.recognize(rgb, tta=use_tta)
        recognition = {
            "calligrapher": raw.get("calligrapher"),
            "confidence": raw.get("confidence"),
            "top_k": top_k(raw.get("all_probabilities", {})),
            "model_backbone": raw.get("model_backbone"),
            "num_classes": raw.get("num_classes"),
            "evidence": raw.get("evidence", {}),
            "ensemble": raw.get("ensemble"),
        }
        result["recognition"] = recognition
        result["reliability"] = reliability_from(quality, float(recognition.get("confidence") or 0))
        ctx.finish(step, recognition, "模型推理完成")
    except Exception as exc:
        ctx.fail(step, exc, classify)
        result.update(blocked=True, diagnostic=classify(exc))
        return _finish(result, ctx, original, rgb, use_examples=use_examples, use_pdf=use_pdf, finalize=finalize)

    if use_rag and search_knowledge:
        step = ctx.add("search_knowledge", detail="检索书法家风格知识")
        try:
            query = f"{result['recognition']['calligrapher']} 书法风格特点 {prompt.strip()}".strip()
            result["knowledge"] = search_knowledge(query=query)
            ctx.finish(step, result["knowledge"], "知识库检索完成")
        except Exception as exc:
            ctx.fail(step, exc, classify)
            result["knowledge_diagnostic"] = classify(exc)
    else:
        ctx.add("search_knowledge", status="skipped", detail="用户关闭了知识库检索")

    return _finish(result, ctx, original, rgb, use_examples=use_examples, use_pdf=use_pdf, finalize=finalize)


def run_multi(
    image: Image.Image,
    *,
    use_tta: bool,
    use_rag: bool,
    use_denoise: bool = False,
    use_examples: bool = False,
    use_pdf: bool = False,
    prompt: str = "",
    get_recognizer: Callable[[], Any],
    assess_quality: Callable[[np.ndarray], dict],
    detect_inversion: Callable[[Image.Image], tuple[Image.Image, dict]],
    denoise: Callable[[Image.Image], tuple[Image.Image, dict]],
    segment: Callable[[np.ndarray], dict],
    search_knowledge: Callable[..., Any] | None,
    classify: Callable[[BaseException], dict],
    finalize: Callable[..., dict],
) -> dict[str, Any]:
    ctx = RunContext()
    original = image.convert("RGB")
    rgb = original
    result: dict[str, Any] = {"mode": "multi"}
    if use_denoise:
        rgb, result["denoise"] = denoise(rgb)
    else:
        ctx.add("denoise", status="skipped", detail="用户关闭了去噪")
    step = ctx.add("extract_image")
    ctx.finish(step, {"size": rgb.size, "preview": image_to_data_url(rgb)}, "图像读取完成")
    step = ctx.add("detect_inversion")
    rgb, result["inversion"] = detect_inversion(rgb)
    ctx.finish(step, result["inversion"], "已完成反色检测")
    step = ctx.add("assess_image_quality")
    result["quality"] = assess_quality(np.array(rgb))
    ctx.finish(step, result["quality"], "图像质量评估完成")
    step = ctx.add("analyze_multi_char.segment")
    seg = segment(np.array(rgb))
    boxes = [list(map(int, box)) for box in seg.get("boxes", [])]
    result["segmentation"] = {"method": seg.get("method"), "num_boxes": len(boxes), "boxes": boxes, "overlay": overlay_boxes(rgb, boxes) if boxes else None}
    ctx.finish(step, result["segmentation"], f"切分完成，检测到 {len(boxes)} 个候选字符")
    if not boxes:
        result.update(blocked=True, diagnostic={"kind": "segmentation_empty", "message": "未检测到字符区域。", "suggestion": "换用更清晰或对比度更高的书法图片。"})
        return _finish(result, ctx, original, rgb, use_examples=use_examples, use_pdf=use_pdf, finalize=finalize)

    step = ctx.add("analyze_multi_char.recognize")
    try:
        recognizer = get_recognizer()
        array = np.array(rgb)
        per_char = []
        for x, y, w, h in boxes:
            pad = 4
            crop = Image.fromarray(array[max(0, y - pad):min(array.shape[0], y + h + pad), max(0, x - pad):min(array.shape[1], x + w + pad)])
            item = recognizer.recognize(crop, tta=use_tta)
            per_char.append({"name": item["calligrapher"], "confidence": item["confidence"], "bbox": [x, y, w, h], "ensemble": item.get("ensemble")})
        votes: dict[str, list[float]] = {}
        for item in per_char:
            votes.setdefault(item["name"], []).append(float(item["confidence"]))
        scores = {name: len(values) * (sum(values) / len(values)) for name, values in votes.items()}
        winner = max(scores, key=scores.get)
        ratio = len(votes[winner]) / max(1, len(per_char))
        result["recognition"] = {"calligrapher": winner, "confidence": round(sum(votes[winner]) / len(votes[winner]), 4), "per_char_results": per_char, "vote_distribution": {name: len(values) for name, values in votes.items()}, "consistency": "high" if ratio > 0.7 else "medium" if ratio > 0.4 else "low"}
        result["reliability"] = result["recognition"]["consistency"]
        ctx.finish(step, result["recognition"], "逐字识别和投票完成")
    except Exception as exc:
        ctx.fail(step, exc, classify)
        result.update(blocked=True, diagnostic=classify(exc))
        return _finish(result, ctx, original, rgb, use_examples=use_examples, use_pdf=use_pdf, finalize=finalize)
    if use_rag and search_knowledge:
        step = ctx.add("search_knowledge")
        try:
            query = f"{result['recognition']['calligrapher']} 书法风格特点 {prompt.strip()}".strip()
            result["knowledge"] = search_knowledge(query=query)
            ctx.finish(step, result["knowledge"], "知识库检索完成")
        except Exception as exc:
            ctx.fail(step, exc, classify)
            result["knowledge_diagnostic"] = classify(exc)
    return _finish(result, ctx, original, rgb, use_examples=use_examples, use_pdf=use_pdf, finalize=finalize)
