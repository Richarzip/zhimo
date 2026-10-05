from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

from aiohttp import BodyPartReader, web
from PIL import Image, UnidentifiedImageError


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
    "prompt", "session_id", "text",
})

for import_root in (PROJECT_ROOT, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def json_response(data: dict[str, Any], status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(data, ensure_ascii=False),
        status=status,
        content_type="application/json",
    )


from zhimo.application.outputs import (
    classify_exception as _classify_exception,
    finalize_outputs as _finalize_outputs,
)


def classify_exception(exc: BaseException) -> dict[str, Any]:
    return _classify_exception(exc)


def finalize_outputs(
    result: dict[str, Any],
    original: Image.Image,
    processed: Image.Image,
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

def run_chat(image: Image.Image, fields: dict[str, str]) -> dict[str, Any]:
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
    return web.FileResponse(STATIC_DIR / "index.html")


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
        use_tta = fields.get("tta", "false") == "true"
        use_cam = fields.get("cam", "true") == "true"
        use_rag = fields.get("rag", "true") == "true"
        use_denoise = fields.get("denoise", "true") == "true"
        use_examples = fields.get("examples", "false") == "true"
        use_pdf = fields.get("pdf", "false") == "true"
        prompt = fields.get("prompt", "")
        extra = {}
        if "denoise" in fields:
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
