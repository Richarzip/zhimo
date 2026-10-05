"""Application-level result enrichment shared by HTTP and CLI adapters."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from PIL import Image


def classify_exception(exc: BaseException) -> dict[str, Any]:
    text = str(exc)
    if isinstance(exc, FileNotFoundError) or "权重" in text or "convnext.pth" in text or ".pth" in text:
        return {
            "kind": "missing_model_weight",
            "message": "已运行到模型推理入口，但未找到模型权重。补齐权重后可继续推理。",
            "suggestion": "将权重放到 zhimo/checkpoints/convnext.pth，或启动服务时设置 ZHIMO_MODEL_PATH。",
            "raw": text,
        }
    if "Ollama" in text or "bge-m3" in text or "Connection refused" in text:
        return {
            "kind": "rag_unavailable",
            "message": "知识库检索不可用，通常是 Ollama 未启动或 bge-m3 尚未拉取。",
            "suggestion": "运行 ollama pull bge-m3，并确认 Ollama 服务已启动。",
            "raw": text,
        }
    if "pytorch_grad_cam" in text:
        return {
            "kind": "cam_dependency_missing",
            "message": "Grad-CAM 依赖缺失，普通识别不受影响。",
            "suggestion": "在当前环境安装 python -m pip install grad-cam。",
            "raw": text,
        }
    return {
        "kind": "runtime_error",
        "message": "运行阶段出现异常。",
        "suggestion": "查看 raw 字段和服务端日志定位问题。",
        "raw": text,
    }


def reference_examples(project_root: Path, name: str, limit: int = 2) -> list[dict[str, Any]]:
    """Load only audited local assets and never expose arbitrary paths."""
    audit_candidates = [
        project_root / "research" / "baidu_knowledge_audit.json",
        project_root / "research" / "knowledge_audit.json",
        project_root / "knowledge_audit.json",
    ]
    audit_path = next((path for path in audit_candidates if path.exists()), None)
    if audit_path is None:
        return []
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    for entry in audit.get("entries", []):
        if entry.get("name") != name:
            continue
        examples = []
        for item in entry.get("images", [])[:limit]:
            rel = item.get("path", "")
            path = (project_root / rel).resolve()
            if not path.exists() and "assets/calligrapher/" in rel:
                path = (project_root / rel.replace("assets/calligrapher/", "assets/calligraphers/")).resolve()
            try:
                path.relative_to(project_root)
            except ValueError:
                continue
            if path.exists():
                examples.append({
                    "title": item.get("title", "作品示例"),
                    "url": item.get("url") or f"/static/assets/calligraphers/{path.name}",
                    "source_url": item.get("source_url", ""),
                    "license": item.get("license", ""),
                    "path": str(path),
                })
        return examples
    return []


def finalize_outputs(
    result: dict[str, Any],
    original: Image.Image,
    processed: Image.Image,
    *,
    project_root: Path,
    use_examples: bool,
    use_pdf: bool,
) -> dict[str, Any]:
    name = (result.get("recognition") or {}).get("calligrapher")
    examples = reference_examples(project_root, name) if use_examples and name else []
    if use_examples:
        result["reference_examples"] = [
            {key: value for key, value in item.items() if key != "path"}
            for item in examples
        ]
        if name and not examples:
            result["reference_examples_note"] = "未找到已审计的本地作品图片，未展示未经核验的网络图片。"
    if use_pdf:
        from .report import build_pdf_report
        pdf = build_pdf_report(original, processed, result, examples=examples)
        result["pdf"] = {
            "filename": f"zhimo_{name or 'analysis'}.pdf",
            "data_url": f"data:application/pdf;base64,{base64.b64encode(pdf).decode('ascii')}",
        }
    return result


__all__ = ["classify_exception", "reference_examples", "finalize_outputs"]