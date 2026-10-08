from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import time
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl
from urllib.request import urlopen

from aiohttp import BodyPartReader, web
from PIL import Image, ImageOps, UnidentifiedImageError


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
STATIC_DIR = FRONTEND_DIR / "static"
SRC_DIR = PROJECT_ROOT / "src"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_FIELD_BYTES = 16 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "GIF"})
UPLOAD_FIELDS = frozenset({
    "mode", "tta", "cam", "rag", "denoise", "examples", "pdf",
    "prompt", "session_id", "text", "analysis_mode",
})

for import_root in (PROJECT_ROOT, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def json_response(data: dict[str, Any], status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(data, ensure_ascii=False, default=_json_default),
        status=status,
        content_type="application/json",
    )


def _json_default(value: Any) -> Any:
    """Convert common array-library scalars at the HTTP boundary."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


from zhimo.application.outputs import (
    classify_exception as _classify_exception,
    finalize_outputs as _finalize_outputs,
)


def classify_exception(exc: BaseException) -> dict[str, Any]:
    return _classify_exception(exc)


def finalize_outputs(
    result: dict[str, Any],
    original: Image.Image | None,
    processed: Image.Image | None,
    *,
    use_examples: bool,
    use_pdf: bool,
) -> dict[str, Any]:
    return _finalize_outputs(
        result,
        original,
        processed,
        project_root=PROJECT_ROOT,
        use_examples=use_examples,
        use_pdf=use_pdf,
    )

OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
OLLAMA_DEFAULT_MODEL = "bge-m3"


def _ollama_url() -> str:
    value = os.getenv("OLLAMA_HOST", OLLAMA_DEFAULT_URL).strip()
    if not value:
        return OLLAMA_DEFAULT_URL
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")


def _ollama_tags(url: str) -> dict[str, Any] | None:
    try:
        with urlopen(f"{url}/api/tags", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _ollama_has_model(payload: dict[str, Any] | None, model: str) -> bool:
    return any(str(item.get("name", "")).split(":", 1)[0] == model
               for item in (payload or {}).get("models", []))


def _find_ollama() -> str | None:
    executable = shutil.which("ollama")
    if executable:
        return executable
    candidates = [
        Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.getenv("ProgramFiles", "")) / "Ollama" / "ollama.exe",
        Path(os.getenv("ProgramW6432", "")) / "Ollama" / "ollama.exe",
    ]
    return next((str(path) for path in candidates if path.is_file()), None)

def ensure_ollama_ready() -> dict[str, Any]:
    """Start Ollama and ensure the embedding model is available for RAG."""
    if os.getenv("ZHIMO_AUTO_START_OLLAMA", "true").lower() in {"0", "false", "no"}:
        return {"available": False, "skipped": True, "reason": "disabled"}
    url = _ollama_url()
    model = os.getenv("ZHIMO_OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL).strip() or OLLAMA_DEFAULT_MODEL
    tags = _ollama_tags(url)
    started = False
    if tags is None:
        executable = _find_ollama()
        if executable is None:
            print("[RAG] Ollama executable not found; knowledge search will be unavailable.")
            return {"available": False, "reason": "ollama_not_found", "url": url, "model": model}
        print("[RAG] Ollama is not running; starting `ollama serve`...")
        try:
            subprocess.Popen([executable, "serve"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            started = True
        except OSError as exc:
            print(f"[RAG] Could not start Ollama: {exc}")
            return {"available": False, "reason": "ollama_start_failed", "url": url, "model": model}
        deadline = time.monotonic() + float(os.getenv("ZHIMO_OLLAMA_START_TIMEOUT", "20"))
        while time.monotonic() < deadline:
            time.sleep(0.5)
            tags = _ollama_tags(url)
            if tags is not None:
                break
        if tags is None:
            print("[RAG] Ollama did not become ready; knowledge search will be unavailable.")
            return {"available": False, "reason": "ollama_not_ready", "url": url, "model": model}
    if not _ollama_has_model(tags, model):
        executable = _find_ollama()
        if executable is None:
            return {"available": False, "reason": "ollama_not_found", "url": url, "model": model}
        print(f"[RAG] Ollama is ready, but `{model}` is missing; pulling it now...")
        try:
            result = subprocess.run([executable, "pull", model], check=False,
                                    timeout=float(os.getenv("ZHIMO_OLLAMA_PULL_TIMEOUT", "1800")))
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[RAG] Could not pull `{model}`: {exc}")
            return {"available": False, "reason": "model_pull_failed", "url": url, "model": model}
        if result.returncode != 0:
            print(f"[RAG] `ollama pull {model}` failed with exit code {result.returncode}.")
            return {"available": False, "reason": "model_pull_failed", "url": url, "model": model}
    print(f"[RAG] Ollama ready with `{model}`.")
    return {"available": True, "started": started, "url": url, "model": model}

def resolve_model_paths() -> list[Path]:
    from zhimo.application import resolve_model_paths as resolve_configured_paths

    return resolve_configured_paths()


def get_settings():
    from zhimo.config import get_settings as load_settings
    return load_settings()

def get_recognizer():
    from zhimo.application import get_recognizer as build_recognizer

    return build_recognizer()


# Agent session runtime is owned by the application layer.
def get_agent():
    from zhimo.application import get_agent as build_agent
    return build_agent()


def get_knowledge_searcher():
    """Resolve the formal RAG tool while honoring injected legacy test doubles."""
    injected = sys.modules.get("calligrapher_tool")
    if injected is not None and hasattr(injected, "search_knowledge"):
        return injected.search_knowledge
    from zhimo.agent.tools import search_knowledge
    return search_knowledge
def _legacy_or_formal(name: str, attribute: str, formal_module: str):
    injected = sys.modules.get(name)
    if injected is not None and hasattr(injected, attribute):
        return getattr(injected, attribute)
    module = __import__(formal_module, fromlist=[attribute])
    return getattr(module, attribute)


def get_denoiser():
    return _legacy_or_formal("image_denoise", "denoise_image", "zhimo.vision.denoise")


def get_inversion_detector():
    return _legacy_or_formal("image_preprocess", "detect_and_fix_inversion", "zhimo.vision.preprocess")


def get_quality_assessor():
    return _legacy_or_formal("image_quality", "assess_image_quality", "zhimo.vision.quality")


def get_segmenter():
    return _legacy_or_formal("segment", "segment_auto", "zhimo.vision.segmentation")

def run_chat(image: Image.Image | None, fields: dict[str, str]) -> dict[str, Any]:
    """Compatibility adapter for the application chat pipeline."""
    denoise_image = get_denoiser()
    from zhimo.application.pipeline import run_chat as pipeline_run_chat

    return pipeline_run_chat(
        image,
        fields,
        get_agent=get_agent,
        denoise=denoise_image,
        classify=classify_exception,
        finalize=finalize_outputs,
    )


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
) -> dict[str, Any]:
    """Compatibility adapter for the application single-image pipeline."""
    denoise_image = get_denoiser()
    detect_and_fix_inversion = get_inversion_detector()
    assess_image_quality = get_quality_assessor()
    from zhimo.application.pipeline import run_single as pipeline_run_single

    search = None
    if use_rag:
        search_knowledge = get_knowledge_searcher()
        search = lambda *, query: search_knowledge.invoke({"query": query})
    return pipeline_run_single(
        image,
        use_tta=use_tta,
        use_cam=use_cam,
        use_rag=use_rag,
        use_denoise=use_denoise,
        use_examples=use_examples,
        use_pdf=use_pdf,
        prompt=prompt,
        get_recognizer=get_recognizer,
        assess_quality=assess_image_quality,
        detect_inversion=detect_and_fix_inversion,
        denoise=denoise_image,
        search_knowledge=search,
        classify=classify_exception,
        finalize=finalize_outputs,
    )


def run_multi(
    image: Image.Image,
    *,
    use_tta: bool,
    use_rag: bool,
    use_denoise: bool = False,
    use_examples: bool = False,
    use_pdf: bool = False,
    prompt: str = "",
) -> dict[str, Any]:
    """Compatibility adapter for the application multi-image pipeline."""
    denoise_image = get_denoiser()
    detect_and_fix_inversion = get_inversion_detector()
    assess_image_quality = get_quality_assessor()
    segment_auto = get_segmenter()
    from zhimo.application.pipeline import run_multi as pipeline_run_multi

    search = None
    if use_rag:
        search_knowledge = get_knowledge_searcher()
        search = lambda *, query: search_knowledge.invoke({"query": query})
    return pipeline_run_multi(
        image,
        use_tta=use_tta,
        use_rag=use_rag,
        use_denoise=use_denoise,
        use_examples=use_examples,
        use_pdf=use_pdf,
        prompt=prompt,
        get_recognizer=get_recognizer,
        assess_quality=assess_image_quality,
        detect_inversion=detect_and_fix_inversion,
        denoise=denoise_image,
        segment=segment_auto,
        search_knowledge=search,
        classify=classify_exception,
        finalize=finalize_outputs,
    )

async def index(_: web.Request) -> web.FileResponse:
    resp = web.FileResponse(STATIC_DIR / "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@web.middleware
async def no_cache_static(request: web.Request, handler) -> web.StreamResponse:
    """静态资源不缓存：开发期频繁改动 HTML/JS/CSS，避免浏览器加载旧版本。"""
    response = await handler(request)
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


async def health(_: web.Request) -> web.Response:
    settings = get_settings()
    model_paths = resolve_model_paths()
    model_exists = settings.models_exist
    data = {
        "project_root": str(PROJECT_ROOT),
        "model_path": os.pathsep.join(str(path) for path in model_paths),
        "model_paths": [str(path) for path in model_paths],
        "model_exists": model_exists,
        "ensemble_enabled": settings.ensemble_enabled,
        "chroma_db_exists": (settings.chroma_dir / "chroma.sqlite3").exists(),
        "sample_images": sorted(p.name for p in (PROJECT_ROOT / "image_test").glob("*.png")) if (PROJECT_ROOT / "image_test").exists() else [],
        "python": sys.version.split()[0],
    }
    return json_response(data)


class UploadError(Exception):
    def __init__(self, message: str, *, status: int = 400, error: str = "invalid_image") -> None:
        super().__init__(message)
        self.status = status
        self.error = error


async def read_upload(request: web.Request) -> tuple[bytes | None, dict[str, str]]:
    if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
        raise UploadError("上传请求过大，图片不能超过 32 MiB。", status=413, error="upload_too_large")
    if request.content_type not in {"multipart/form-data", "application/x-www-form-urlencoded"}:
        raise UploadError("请使用图片或文字表单。", error="invalid_request")

    fields: dict[str, str] = {}
    image_bytes: bytes | None = None
    seen: set[str] = set()

    def check_field(name: str | None, *, allow_image: bool) -> None:
        if name is None or (name not in UPLOAD_FIELDS and not (allow_image and name == "image")):
            raise UploadError("上传表单包含未知字段。", error="invalid_request")
        if name in seen:
            raise UploadError("上传表单包含重复字段。", error="invalid_request")
        seen.add(name)

    try:
        if request.content_type == "application/x-www-form-urlencoded":
            body = bytearray()
            async for chunk in request.content.iter_chunked(8192):
                if len(body) + len(chunk) > MAX_REQUEST_BYTES or request.content.total_bytes > MAX_REQUEST_BYTES:
                    raise UploadError("上传请求过大。", status=413, error="upload_too_large")
                body.extend(chunk)
            pairs = parse_qsl(
                body.decode("utf-8"), keep_blank_values=True, strict_parsing=True,
                encoding="utf-8", errors="strict", max_num_fields=len(UPLOAD_FIELDS),
            )
            for name, value in pairs:
                check_field(name, allow_image=False)
                if len(value.encode("utf-8")) > MAX_FIELD_BYTES:
                    raise UploadError("上传表单字段过长。", status=413, error="upload_too_large")
                fields[name] = value.strip()
        else:
            reader = await request.multipart()
            async for part in reader:
                if not isinstance(part, BodyPartReader):
                    raise UploadError("不支持嵌套上传表单。", error="invalid_request")
                name = part.name
                check_field(name, allow_image=True)
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
        raise UploadError("上传表单无效，请重新提交。", error="invalid_request") from exc

    if fields.get("analysis_mode", "auto") not in {"auto", "single", "multi"}:
        raise UploadError("分析模式必须为 auto、single 或 multi。", error="invalid_request")
    mode = fields.get("mode", "single")
    if mode not in {"single", "multi", "chat"}:
        raise UploadError("分析模式无效。", error="invalid_request")
    if any(
        fields.get(name, "false") not in {"true", "false"}
        for name in ("tta", "cam", "rag", "denoise", "examples", "pdf")
    ):
        raise UploadError("分析选项无效。", error="invalid_request")
    if image_bytes == b"":
        raise UploadError("图片文件为空，请选择有效图片。")
    if image_bytes is None:
        if mode != "chat":
            raise UploadError("请上传图片。", error="missing_image")
        if not (fields.get("text") or fields.get("prompt")):
            raise UploadError("请输入问题或上传图片。", error="missing_input")
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
            oriented = ImageOps.exif_transpose(image)
        except BaseException:
            image.close()
            raise
        image.close()
        return oriented
    except Image.DecompressionBombError as exc:
        raise UploadError("图片不能超过 2500 万像素。", status=413, error="image_too_large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise UploadError("图片无法解码，文件可能已损坏或不是支持的图片。") from exc


def analyze_image(image_bytes: bytes | None, fields: dict[str, str]) -> dict[str, Any]:
    # Image decoding, model inference and knowledge lookup all run in the worker.
    if image_bytes is None:
        if fields.get("mode", "single") != "chat":
            raise UploadError("请上传图片。", error="missing_image")
        return run_chat(None, fields)
    with decode_image(image_bytes) as image:
        mode = fields.get("mode", "single")
        use_tta = fields.get("tta", "false") == "true"
        use_cam = fields.get("cam", "false") == "true"
        use_rag = fields.get("rag", "true") == "true"
        use_denoise = fields.get("denoise", "true") == "true"
        use_examples = fields.get("examples", "false") == "true"
        use_pdf = fields.get("pdf", "false") == "true"
        prompt = fields.get("prompt", "")
        extra = {}
        extra["use_denoise"] = use_denoise
        if "examples" in fields:
            extra["use_examples"] = use_examples
        if "pdf" in fields:
            extra["use_pdf"] = use_pdf
        from zhimo.application import AnalysisRequest, AnalysisService

        def legacy_pipeline(request: AnalysisRequest) -> dict[str, Any]:
            if request.mode == "chat":
                return run_chat(request.image, fields)
            if request.mode == "multi":
                return run_multi(
                    request.image,
                    use_tta=request.tta,
                    use_rag=request.rag,
                    **extra,
                    prompt=request.prompt,
                )
            return run_single(
                request.image,
                use_tta=request.tta,
                use_cam=request.cam,
                use_rag=request.rag,
                **extra,
                prompt=request.prompt,
            )

        request = AnalysisRequest(
            image=image,
            mode=mode,
            tta=use_tta,
            cam=use_cam,
            rag=use_rag,
            denoise=use_denoise,
            prompt=prompt,
            session_id=fields.get("session_id"),
        )
        return AnalysisService(legacy_pipeline).analyze(request)


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

    def submit(self, image_bytes: bytes | None, fields: dict[str, str]) -> asyncio.Future:
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


MAX_EXPORT_ENTRIES = 200


async def legacy_export_chat(request: web.Request) -> web.Response:
    """导出对话记录为 PDF（由前端按钮或 Agent 的 export_pdf 工具触发）。

    请求体为 JSON：{"entries": [{"role", "text", "image"(可选 base64), "recognition"(可选)}]}
    前端负责剥离 markdown、压缩图片；本接口只做确定性渲染，返回 pdf data_url。
    """
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return json_response({"error": "bad_json", "message": "请求体必须是 JSON。"}, status=400)

    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        return json_response({"error": "empty_entries", "message": "没有可导出的对话内容。"}, status=400)
    if len(entries) > MAX_EXPORT_ENTRIES:
        return json_response({"error": "too_many_entries", "message": f"对话条目过多（超过 {MAX_EXPORT_ENTRIES} 条）。"}, status=400)

    from zhimo.application.report import build_chat_pdf
    try:
        pdf = build_chat_pdf(entries)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return json_response(
            {"error": "pdf_failed", "message": f"PDF 生成失败：{type(exc).__name__}。", "diagnostic": classify_exception(exc)},
            status=500,
        )

    data_url = f"data:application/pdf;base64,{base64.b64encode(pdf).decode('ascii')}"
    return json_response({
        "pdf": {
            "filename": f"zhimo_chat_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            "data_url": data_url,
        }
    })


async def export_chat(request: web.Request) -> web.Response:
    """Generate an identification report for one image and one result."""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return json_response({"error": "bad_json", "message": "Invalid JSON body."}, status=400)

    image_data = payload.get("image") if isinstance(payload, dict) else None
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(image_data, str) or not image_data:
        return json_response({"error": "missing_image", "message": "No image report can be exported."}, status=400)
    if not isinstance(result, dict):
        return json_response({"error": "missing_result", "message": "No identification result can be exported."}, status=400)

    try:
        raw = image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data
        image_bytes = base64.b64decode(raw, validate=True)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        with decode_image(image_bytes) as source:
            image = source.convert("RGB")
            from zhimo.application.report import build_pdf_report
            pdf = build_pdf_report(image, image, result)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return json_response(
            {"error": "pdf_failed", "message": f"Report generation failed: {type(exc).__name__}", "diagnostic": classify_exception(exc)},
            status=400 if isinstance(exc, (ValueError, UploadError)) else 500,
        )

    data_url = f"data:application/pdf;base64,{base64.b64encode(pdf).decode('ascii')}"
    return json_response({
        "pdf": {
            "filename": f"zhimo_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            "data_url": data_url,
        }
    })


def create_app() -> web.Application:
    app = web.Application(client_max_size=MAX_REQUEST_BYTES, middlewares=[no_cache_static])
    app.cleanup_ctx.append(analysis_worker_context)
    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_post("/api/analyze", analyze)
    app.router.add_post("/api/export", export_chat)
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
    ensure_ollama_ready()
    port = find_port(args.port)
    print(f"Zhimo frontend running at http://{args.host}:{port}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Model paths:")
    for path in resolve_model_paths():
        print(f"  - {path}")
    web.run_app(create_app(), host=args.host, port=port)


if __name__ == "__main__":
    main()
