from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from aiohttp import BodyPartReader, web
from PIL import Image, ImageDraw, UnidentifiedImageError


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
STATIC_DIR = FRONTEND_DIR / "static"
DEFAULT_MODEL_PATHS = (
    PROJECT_ROOT / "checkpoints" / "convnext.pth",
    PROJECT_ROOT / "checkpoints" / "swin.pth",
)
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_FIELD_BYTES = 16 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "GIF"})
UPLOAD_FIELDS = frozenset({"mode", "tta", "cam", "rag", "prompt", "session_id", "text"})

sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Step:
    name: str
    status: str = "pending"
    detail: str = ""
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "data": self.data,
            "error": self.error,
        }


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

    def fail(self, step: Step, exc: BaseException, detail: str = "") -> None:
        step.status = "error"
        step.error = str(exc)
        step.detail = detail or classify_exception(exc)["message"]
        step.data = classify_exception(exc)


def json_response(data: dict[str, Any], status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(data, ensure_ascii=False),
        status=status,
        content_type="application/json",
    )


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def overlay_boxes(image: Image.Image, boxes: list[list[int]] | list[tuple[int, int, int, int]]) -> str:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for idx, box in enumerate(boxes, start=1):
        x, y, w, h = [int(v) for v in box]
        draw.rectangle([x, y, x + w, y + h], outline=(24, 119, 242), width=3)
        draw.rectangle([x, max(0, y - 18), x + 26, y], fill=(24, 119, 242))
        draw.text((x + 4, max(0, y - 17)), str(idx), fill=(255, 255, 255))
    return image_to_data_url(canvas)


def classify_exception(exc: BaseException) -> dict[str, Any]:
    text = str(exc)
    if isinstance(exc, FileNotFoundError) or "权重" in text or "convnext.pth" in text or ".pth" in text:
        return {
            "kind": "missing_model_weight",
            "message": "已成功运行到模型推理入口，但未找到模型权重。补齐权重后可继续推理。",
            "suggestion": "将权重放到 zhimo/checkpoints/convnext.pth，或启动服务时设置 ZHIMO_MODEL_PATH。",
            "raw": text,
        }
    if "Ollama" in text or "bge-m3" in text or "Connection refused" in text:
        return {
            "kind": "rag_unavailable",
            "message": "知识库检索不可用，通常是 Ollama 未启动或 bge-m3 未拉取。",
            "suggestion": "运行 ollama pull bge-m3，并确认 Ollama 服务已启动。",
            "raw": text,
        }
    if "No module named 'pytorch_grad_cam'" in text or "pytorch_grad_cam" in text:
        return {
            "kind": "cam_dependency_missing",
            "message": "Grad-CAM 依赖缺失，普通识别不受影响。",
            "suggestion": "在环境中安装 python -m pip install grad-cam。",
            "raw": text,
        }
    return {
        "kind": "runtime_error",
        "message": "运行阶段出现异常。",
        "suggestion": "查看 raw 字段和服务端日志定位。",
        "raw": text,
    }


def top_k(all_probabilities: dict[str, float], k: int = 3) -> list[dict[str, Any]]:
    return [
        {"name": name, "confidence": score}
        for name, score in sorted(all_probabilities.items(), key=lambda item: item[1], reverse=True)[:k]
    ]


def reliability_from(quality: dict[str, float], confidence: float) -> str:
    q = float(quality.get("overall", 1.0))
    c = float(confidence)
    if q < 0.3 and c > 0.7:
        return "high"
    if q < 0.5 or c > 0.5:
        return "medium"
    return "low"


def build_summary(result: dict[str, Any]) -> str:
    if result.get("blocked"):
        diag = result.get("diagnostic") or {}
        return diag.get("message", "流程已运行，但在某一步停止。")

    recognition = result.get("recognition") or {}
    calligrapher = recognition.get("calligrapher")
    confidence = recognition.get("confidence")
    quality = result.get("quality") or {}
    reliability = result.get("reliability") or "unknown"
    knowledge = result.get("knowledge")

    if not calligrapher:
        return "已完成图像预处理，但没有得到书法家识别结论。"

    parts = [f"本次本地 Agent 判断最可能的书法家为：{calligrapher}。"]
    inversion = result.get("inversion") or {}
    if inversion.get("was_inverted"):
        parts.append("该图片为黑底白字的反色拓印，已自动反转为白底黑字后再识别。")
    if confidence is not None:
        parts.append(f"模型置信度为 {confidence}，综合可靠性评级为 {reliability}。")
    if quality:
        parts.append(
            "图像质量评估："
            f"泛黄 {quality.get('yellow')}、褪色 {quality.get('fade')}、"
            f"噪声 {quality.get('noise')}、模糊 {quality.get('blur')}、综合退化 {quality.get('overall')}。"
        )
    if recognition.get("evidence", {}).get("attention_note"):
        parts.append(f"可解释性提示：{recognition['evidence']['attention_note']}。")
    if knowledge:
        parts.append("知识库检索已返回相关风格材料，可结合 Top-K 候选和热力图进一步解释。")
    return "".join(parts)


def resolve_model_paths() -> list[Path]:
    raw_paths = os.environ.get("ZHIMO_MODEL_PATHS")
    if raw_paths:
        paths = [Path(item.strip()) for item in raw_paths.split(os.pathsep) if item.strip()]
    elif os.environ.get("ZHIMO_MODEL_PATH"):
        # Backward-compatible single-model override for local diagnostics.
        paths = [Path(os.environ["ZHIMO_MODEL_PATH"])]
    else:
        paths = list(DEFAULT_MODEL_PATHS)
    return [path if path.is_absolute() else PROJECT_ROOT / path for path in paths]


def get_recognizer():
    paths = resolve_model_paths()
    if len(paths) == 1:
        from calligrapher_recognizer import CalligrapherRecognizer

        return CalligrapherRecognizer(model_path=str(paths[0]))

    from ensemble_recognizer import EnsembleRecognizer

    return EnsembleRecognizer(model_paths=[str(path) for path in paths])


# ====================== 多轮对话 Agent ======================

# 进程内单例：Agent 图 + 记忆检查点。checkpointer 按 thread_id 保存
# 每轮 messages，从而实现多轮记忆；服务重启后记忆清空。
_AGENT = None
_AGENT_CHECKPOINTER = None


def get_agent():
    """惰性构建带多轮记忆的书法鉴赏 Agent（单例）。"""
    global _AGENT, _AGENT_CHECKPOINTER
    if _AGENT is None:
        from langgraph.checkpoint.memory import MemorySaver
        from agent_builder import create_calligraphy_agent

        _AGENT_CHECKPOINTER = MemorySaver()
        _AGENT = create_calligraphy_agent(checkpointer=_AGENT_CHECKPOINTER)
    return _AGENT


def parse_tool_result(content: Any) -> dict[str, Any]:
    """把 langchain ToolMessage 的 content 解析成 dict。"""
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"raw": content}
        except (json.JSONDecodeError, TypeError):
            return {"raw": content}
    return content if isinstance(content, dict) else {"raw": str(content)}


def run_chat(image: Image.Image, fields: dict[str, str]) -> dict[str, Any]:
    """多轮对话主流程：把当前轮（文字 + 图片）交给带记忆的 Agent。

    - session_id 作为 langgraph thread_id，续接该会话的历史消息。
    - 返回 AI 最终回复、工具调用步骤、以及最新一次识别工具的结构化结果。
    """
    from langchain.messages import HumanMessage

    session_id = fields.get("session_id") or uuid.uuid4().hex
    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}

    # 记录调用前消息数，用于从返回里切出"本轮新增"的消息
    snapshot = agent.get_state(config)
    before = len((snapshot.values or {}).get("messages", [])) if snapshot else 0

    # 构造用户消息：文字 + 图片（图片转 data URL 交给 Agent 工具）
    content: list[dict[str, Any]] = []
    text = (fields.get("text") or fields.get("prompt") or "").strip()
    if text:
        content.append({"type": "text", "text": text})
    content.append({
        "type": "image_url",
        "image_url": {"url": image_to_data_url(image.convert("RGB"), "PNG")},
    })

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - 中间件兜底后的最后防线，返回可读错误
        traceback.print_exc()
        return {
            "session_id": session_id,
            "error": "agent_error",
            "message": f"Agent 执行失败：{type(exc).__name__}: {str(exc)[:200]}",
            "diagnostic": classify_exception(exc),
            "reply": "抱歉，本次分析遇到异常。请稍后重试，或换一张图片 / 换一种问法。",
        }
    msgs = result.get("messages", [])
    new_msgs = msgs[before:]

    # 1) AI 最终回复（最后一条有文本的 ai 消息）
    reply = ""
    for m in reversed(new_msgs):
        if getattr(m, "type", "") == "ai" and m.content:
            reply = m.content if isinstance(m.content, str) else str(m.content)
            break

    # 2) 工具调用步骤 + 各工具的结构化返回
    steps: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for m in new_msgs:
        mtype = getattr(m, "type", "")
        if mtype == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name", "tool")
                steps.append({"name": name, "status": "ok", "detail": f"调用工具 {name}"})
        elif mtype == "tool":
            parsed = parse_tool_result(m.content)
            steps.append({"name": getattr(m, "name", "tool"), "status": "ok", "detail": "工具返回结果"})
            tool_results.append(parsed)

    # 3) 提炼最新一次识别工具的结构化信息（供前端富卡片展示）
    recognition: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    reliability: Any = None
    mode = "single"
    inversion: dict[str, Any] | None = None
    for r in tool_results:
        if not isinstance(r, dict) or r.get("error"):
            continue
        if "calligrapher" in r:
            recognition = {
                "calligrapher": r.get("calligrapher"),
                "confidence": r.get("confidence"),
                "top_k": r.get("top_k", []),
                "model_backbone": r.get("model_backbone"),
                "evidence": r.get("evidence", {}),
                "ensemble": r.get("ensemble"),
            }
            if "num_characters" in r:  # 多字投票结果
                mode = "multi"
                recognition["num_characters"] = r.get("num_characters")
                recognition["per_char_results"] = r.get("per_char_results", [])
                recognition["consistency"] = r.get("consistency")
                recognition["vote_distribution"] = r.get("vote_distribution", {})
            else:
                mode = "single"
            if "quality" in r:  # analyze_calligraphy 附带质量报告
                quality = r.get("quality", {})
                reliability = r.get("reliability")
            elif "consistency" in r:
                reliability = r.get("consistency")
            if r.get("inversion"):
                inversion = r["inversion"]

    return {
        "session_id": session_id,
        "mode": mode,
        "reply": reply,
        "steps": steps,
        "recognition": recognition,
        "quality": quality,
        "reliability": reliability,
        "inversion": inversion,
        "message_count": len(msgs),
    }


def run_single(image: Image.Image, *, use_tta: bool, use_cam: bool, use_rag: bool, prompt: str) -> dict[str, Any]:
    from image_quality import assess_image_quality

    ctx = RunContext()
    result: dict[str, Any] = {"mode": "single"}

    step = ctx.add("extract_image", detail="读取上传图片并转换为 RGB")
    rgb = image.convert("RGB")
    ctx.finish(step, {"size": rgb.size, "preview": image_to_data_url(rgb)}, "图片读取成功")

    step = ctx.add("detect_inversion", detail="检测是否为黑底白字拓印，若是则反转为白底黑字")
    from image_preprocess import detect_and_fix_inversion

    rgb, inversion = detect_and_fix_inversion(rgb)
    result["inversion"] = inversion
    detail = "检测为反色拓印，已自动反转" if inversion["was_inverted"] else "非反色，无需处理"
    ctx.finish(step, inversion, detail)

    step = ctx.add("assess_image_quality", detail="评估泛黄、褪色、噪声和模糊")
    quality = assess_image_quality(__import__("numpy").array(rgb))
    result["quality"] = quality
    ctx.finish(step, quality, "图像质量评估完成")

    step = ctx.add("analyze_calligraphy", detail="加载分类模型并开始单字推理")
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
    except Exception as exc:  # noqa: BLE001 - returned as UI diagnostic intentionally
        ctx.fail(step, exc)
        result["blocked"] = True
        result["diagnostic"] = classify_exception(exc)
        result["summary"] = build_summary(result)
        result["steps"] = [s.to_dict() for s in ctx.steps]
        return result

    if use_rag:
        step = ctx.add("search_knowledge", detail="按识别结果检索书法家风格知识")
        try:
            from calligrapher_tool import search_knowledge

            query = f"{result['recognition']['calligrapher']} 书法风格特点 {prompt.strip()}".strip()
            knowledge = search_knowledge.invoke({"query": query})
            result["knowledge"] = knowledge
            ctx.finish(step, knowledge, "知识库检索完成")
        except Exception as exc:  # noqa: BLE001
            ctx.fail(step, exc)
            result["knowledge_diagnostic"] = classify_exception(exc)
    else:
        ctx.add("search_knowledge", status="skipped", detail="用户关闭了知识库检索")

    step = ctx.add("compose_answer", detail="综合工具返回结果生成本地回答")
    result["summary"] = build_summary(result)
    ctx.finish(step, {"summary": result["summary"]}, "回答生成完成")

    result["steps"] = [s.to_dict() for s in ctx.steps]
    return result


def run_multi(image: Image.Image, *, use_tta: bool, use_rag: bool, prompt: str) -> dict[str, Any]:
    from image_quality import assess_image_quality
    from segment import segment_auto

    import numpy as np

    ctx = RunContext()
    result: dict[str, Any] = {"mode": "multi"}
    rgb = image.convert("RGB")

    step = ctx.add("extract_image", detail="读取上传图片并转换为 RGB")
    ctx.finish(step, {"size": rgb.size, "preview": image_to_data_url(rgb)}, "图片读取成功")

    step = ctx.add("detect_inversion", detail="检测是否为黑底白字拓印，若是则反转为白底黑字")
    from image_preprocess import detect_and_fix_inversion

    rgb, inversion = detect_and_fix_inversion(rgb)
    result["inversion"] = inversion
    detail = "检测为反色拓印，已自动反转" if inversion["was_inverted"] else "非反色，无需处理"
    ctx.finish(step, inversion, detail)

    step = ctx.add("assess_image_quality", detail="评估图像退化程度")
    quality = assess_image_quality(np.array(rgb))
    result["quality"] = quality
    ctx.finish(step, quality, "图像质量评估完成")

    step = ctx.add("analyze_multi_char.segment", detail="调用传统图像处理切分多字作品")
    seg = segment_auto(np.array(rgb))
    boxes = [list(map(int, b)) for b in seg.get("boxes", [])]
    seg_payload = {
        "method": seg.get("method"),
        "num_boxes": len(boxes),
        "boxes": boxes,
        "overlay": overlay_boxes(rgb, boxes) if boxes else None,
    }
    result["segmentation"] = seg_payload
    ctx.finish(step, seg_payload, f"切分完成，检测到 {len(boxes)} 个候选字符")

    if not boxes:
        result["blocked"] = True
        result["diagnostic"] = {"kind": "segmentation_empty", "message": "未检测到字符区域。", "suggestion": "换用更清晰或对比度更高的书法图片。"}
        result["summary"] = build_summary(result)
        result["steps"] = [s.to_dict() for s in ctx.steps]
        return result

    step = ctx.add("analyze_multi_char.recognize", detail="逐字调用书法家分类模型")
    try:
        recognizer = get_recognizer()
        per_char = []
        img_np = np.array(rgb)
        for x, y, w, h in boxes:
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_np.shape[1], x + w + pad)
            y2 = min(img_np.shape[0], y + h + pad)
            char_img = Image.fromarray(img_np[y1:y2, x1:x2])
            r = recognizer.recognize(char_img, tta=use_tta)
            per_char.append({
                "name": r["calligrapher"],
                "confidence": r["confidence"],
                "bbox": [x, y, w, h],
                "ensemble": r.get("ensemble"),
            })

        votes: dict[str, list[float]] = {}
        for item in per_char:
            votes.setdefault(item["name"], []).append(float(item["confidence"]))
        scores = {name: len(vals) * (sum(vals) / len(vals)) for name, vals in votes.items()}
        best_name = max(scores, key=scores.get)
        avg_conf = sum(votes[best_name]) / len(votes[best_name])
        consistency_ratio = len(votes[best_name]) / max(1, len(per_char))
        consistency = "high" if consistency_ratio > 0.7 else "medium" if consistency_ratio > 0.4 else "low"
        recognition = {
            "calligrapher": best_name,
            "confidence": round(avg_conf, 4),
            "per_char_results": per_char,
            "vote_distribution": {name: len(vals) for name, vals in votes.items()},
            "consistency": consistency,
        }
        result["recognition"] = recognition
        result["reliability"] = consistency
        ctx.finish(step, recognition, "逐字识别和投票完成")
    except Exception as exc:  # noqa: BLE001
        ctx.fail(step, exc)
        result["blocked"] = True
        result["diagnostic"] = classify_exception(exc)
        result["summary"] = build_summary(result)
        result["steps"] = [s.to_dict() for s in ctx.steps]
        return result

    if use_rag:
        step = ctx.add("search_knowledge", detail="按投票结果检索书法家风格知识")
        try:
            from calligrapher_tool import search_knowledge

            query = f"{result['recognition']['calligrapher']} 书法风格特点 {prompt.strip()}".strip()
            knowledge = search_knowledge.invoke({"query": query})
            result["knowledge"] = knowledge
            ctx.finish(step, knowledge, "知识库检索完成")
        except Exception as exc:  # noqa: BLE001
            ctx.fail(step, exc)
            result["knowledge_diagnostic"] = classify_exception(exc)

    step = ctx.add("compose_answer", detail="综合工具返回结果生成本地回答")
    result["summary"] = build_summary(result)
    ctx.finish(step, {"summary": result["summary"]}, "回答生成完成")

    result["steps"] = [s.to_dict() for s in ctx.steps]
    return result


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def health(_: web.Request) -> web.Response:
    model_paths = resolve_model_paths()
    model_exists = all(path.exists() for path in model_paths)
    data = {
        "project_root": str(PROJECT_ROOT),
        "model_path": os.pathsep.join(str(path) for path in model_paths),
        "model_paths": [str(path) for path in model_paths],
        "model_exists": model_exists,
        "ensemble_enabled": len(model_paths) > 1,
        "chroma_db_exists": (PROJECT_ROOT / "chroma_db" / "chroma.sqlite3").exists(),
        "sample_images": sorted(p.name for p in (PROJECT_ROOT / "image_test").glob("*.png")) if (PROJECT_ROOT / "image_test").exists() else [],
        "python": sys.version.split()[0],
    }
    return json_response(data)


class UploadError(Exception):
    def __init__(self, message: str, *, status: int = 400, error: str = "invalid_image") -> None:
        super().__init__(message)
        self.status = status
        self.error = error


async def read_upload(request: web.Request) -> tuple[bytes, dict[str, str]]:
    if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
        raise UploadError("上传请求过大，图片不能超过 32 MiB。", status=413, error="upload_too_large")
    if request.content_type != "multipart/form-data":
        raise UploadError("请使用图片上传表单。", error="invalid_request")

    fields: dict[str, str] = {}
    image_bytes: bytes | None = None
    seen: set[str] = set()
    try:
        reader = await request.multipart()
        async for part in reader:
            if not isinstance(part, BodyPartReader):
                raise UploadError("不支持嵌套上传表单。", error="invalid_request")
            name = part.name
            if name not in UPLOAD_FIELDS and name != "image":
                raise UploadError("上传表单包含未知字段。", error="invalid_request")
            if name in seen:
                raise UploadError("上传表单包含重复字段。", error="invalid_request")
            seen.add(name)
            limit = MAX_IMAGE_BYTES if name == "image" else MAX_FIELD_BYTES
            value = bytearray()
            while chunk := await part.read_chunk():
                if len(value) + len(chunk) > limit or request.content.total_bytes > MAX_REQUEST_BYTES:
                    message = "图片不能超过 32 MiB。" if name == "image" else "上传表单字段过长。"
                    raise UploadError(message, status=413, error="upload_too_large")
                value.extend(chunk)
            if name == "image":
                image_bytes = bytes(value)
            else:
                fields[name] = value.decode("utf-8").strip()
        if request.content.total_bytes > MAX_REQUEST_BYTES:
            raise UploadError("上传请求过大。", status=413, error="upload_too_large")
    except (AssertionError, ValueError, UnicodeError) as exc:
        raise UploadError("上传表单无效，请重新选择图片。", error="invalid_request") from exc

    if not image_bytes:
        raise UploadError("请上传图片。", error="missing_image")
    if fields.get("mode", "single") not in {"single", "multi", "chat"}:
        raise UploadError("分析模式无效。", error="invalid_request")
    if any(fields.get(name, "false") not in {"true", "false"} for name in ("tta", "cam", "rag")):
        raise UploadError("分析选项无效。", error="invalid_request")
    return image_bytes, fields


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as probe:
            if probe.format not in ALLOWED_IMAGE_FORMATS:
                raise UploadError("仅支持 JPEG、PNG、WEBP、BMP 和 GIF 图片。")
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise UploadError("图片不能超过 2500 万像素。", status=413, error="image_too_large")
            probe.verify()
        image = Image.open(BytesIO(image_bytes))
        try:
            image.load()
        except BaseException:
            image.close()
            raise
        return image
    except Image.DecompressionBombError as exc:
        raise UploadError("图片不能超过 2500 万像素。", status=413, error="image_too_large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise UploadError("图片无法解码，文件可能已损坏或不是支持的图片。") from exc


def analyze_image(image_bytes: bytes, fields: dict[str, str]) -> dict[str, Any]:
    # Image decoding, model inference and knowledge lookup all run in the worker.
    with decode_image(image_bytes) as image:
        mode = fields.get("mode", "single")
        if mode == "chat":
            # 多轮对话模式：Agent 带记忆，每次带当前轮文字 + 图片
            return run_chat(image, fields)
        use_tta = fields.get("tta", "false") == "true"
        use_cam = fields.get("cam", "true") == "true"
        use_rag = fields.get("rag", "true") == "true"
        prompt = fields.get("prompt", "")
        if mode == "multi":
            return run_multi(image, use_tta=use_tta, use_rag=use_rag, prompt=prompt)
        return run_single(image, use_tta=use_tta, use_cam=use_cam, use_rag=use_rag, prompt=prompt)


class AnalysisWorker:
    """One upload/job at a time, without an unbounded executor queue."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zhimo-analysis")
        self.busy = False
        self.closing = False

    def claim(self) -> bool:
        if self.busy or self.closing:
            return False
        self.busy = True
        return True

    def release(self) -> None:
        self.busy = False

    def submit(self, image_bytes: bytes, fields: dict[str, str]) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        job = self.executor.submit(analyze_image, image_bytes, fields)
        # Only the actual thread completion frees the slot, even if the HTTP
        # handler was cancelled while awaiting this job.
        job.add_done_callback(lambda _: loop.call_soon_threadsafe(self.release))
        future = asyncio.wrap_future(job)
        # A disconnected request may never retrieve a worker exception.
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return future

    async def close(self) -> None:
        self.closing = True
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)


ANALYSIS_WORKER_KEY = web.AppKey("analysis_worker", AnalysisWorker)


async def analysis_worker_context(app: web.Application):
    worker = AnalysisWorker()
    app[ANALYSIS_WORKER_KEY] = worker
    try:
        yield
    finally:
        await worker.close()


async def analyze(request: web.Request) -> web.Response:
    worker = request.app[ANALYSIS_WORKER_KEY]
    if not worker.claim():
        return json_response({"error": "server_busy", "message": "已有图片正在处理，请稍后重试。"}, status=503)
    submitted = False
    try:
        image_bytes, fields = await read_upload(request)
        future = worker.submit(image_bytes, fields)
        submitted = True
        result = await asyncio.shield(future)
        return json_response(result)
    except UploadError as exc:
        return json_response({"error": exc.error, "message": str(exc)}, status=exc.status)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return json_response({"error": "server_error", "diagnostic": classify_exception(exc)}, status=500)
    finally:
        if not submitted:
            worker.release()


def create_app() -> web.Application:
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app.cleanup_ctx.append(analysis_worker_context)
    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_post("/api/analyze", analyze)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    return app


def find_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Zhimo local frontend server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    port = find_port(args.port)
    print(f"Zhimo frontend running at http://{args.host}:{port}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Model paths:")
    for path in resolve_model_paths():
        print(f"  - {path}")
    web.run_app(create_app(), host=args.host, port=port)


if __name__ == "__main__":
    main()
