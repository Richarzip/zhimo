"""Frontend API regressions. Uses real HTTP/Pillow; models and RAG are stubbed."""
from __future__ import annotations

import asyncio
import importlib.util
import io
import struct
import sys
import threading
import types
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
SPEC = importlib.util.spec_from_file_location("zhimo_frontend_server_tests", SERVER_PATH)
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


def image_bytes(fmt="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format=fmt)
    return output.getvalue()


def form_for(payload=None, **fields):
    form = FormData()
    form.add_field("image", image_bytes() if payload is None else payload,
                   filename="upload.png", content_type="image/png")
    for name, value in fields.items():
        form.add_field(name, value)
    return form


class APIRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cancelled = asyncio.Event()

        @web.middleware
        async def observe_cancellation(request, handler):
            try:
                return await handler(request)
            except asyncio.CancelledError:
                if request.path == "/api/analyze":
                    self.cancelled.set()
                raise

        self.app = server.create_app()
        self.app.middlewares.append(observe_cancellation)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.mock_single = Mock(return_value={"mode": "single", "summary": "done"})
        self.inference_patch = patch.object(server, "run_single", self.mock_single)
        self.inference_patch.start()
        self.addCleanup(self.inference_patch.stop)

    async def asyncTearDown(self):
        await self.client.close()

    async def assert_response(self, response, status, error=None):
        self.assertEqual(response.status, status, await response.text())
        self.assertEqual(response.content_type, "application/json")
        data = await response.json()
        if error is not None:
            self.assertEqual(data["error"], error)
            self.assertTrue(data["message"])
        return data

    async def assert_ready_again(self):
        response = await self.client.post("/api/analyze", data=form_for())
        await self.assert_response(response, 200)

    async def test_supported_formats_and_inference_options(self):
        for fmt in ("PNG", "JPEG", "WEBP", "BMP", "GIF"):
            with self.subTest(format=fmt):
                response = await self.client.post("/api/analyze", data=form_for(
                    image_bytes(fmt), tta="true", cam="true", rag="false", prompt="原因"))
                await self.assert_response(response, 200)
                self.assertEqual(self.mock_single.call_args.kwargs,
                                 {"use_tta": True, "use_cam": True, "use_rag": False, "prompt": "原因"})

    async def test_rejects_fake_image_truncated_image_and_unsupported_format(self):
        for payload in (b"not an image", image_bytes()[:40], image_bytes("TIFF")):
            with self.subTest(payload_size=len(payload)):
                response = await self.client.post("/api/analyze", data=form_for(payload))
                await self.assert_response(response, 400, "invalid_image")
        self.mock_single.assert_not_called()
        await self.assert_ready_again()

    async def test_rejects_more_than_25_million_pixels_before_decoding(self):
        payload = bytearray(image_bytes())
        payload[16:24] = struct.pack(">II", 5001, 5000)
        payload[29:33] = struct.pack(">I", zlib.crc32(payload[12:29]) & 0xffffffff)
        response = await self.client.post("/api/analyze", data=form_for(bytes(payload)))
        await self.assert_response(response, 413, "image_too_large")
        self.mock_single.assert_not_called()
        await self.assert_ready_again()

    async def test_image_limit_is_enforced_with_and_without_content_length(self):
        payload = image_bytes()
        for chunked in (False, True):
            with self.subTest(chunked=chunked), patch.object(server, "MAX_IMAGE_BYTES", len(payload) - 1):
                response = await self.client.post("/api/analyze", data=form_for(payload), chunked=chunked or None)
                await self.assert_response(response, 413, "upload_too_large")
        self.mock_single.assert_not_called()
        # The file size limit excludes multipart boundaries and headers.
        with patch.object(server, "MAX_IMAGE_BYTES", len(payload)):
            await self.assert_ready_again()

    async def test_total_request_limit_with_and_without_content_length(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked), patch.object(server, "MAX_REQUEST_BYTES", 64):
                response = await self.client.post("/api/analyze", data=form_for(), chunked=chunked or None)
                await self.assert_response(response, 413, "upload_too_large")
        self.mock_single.assert_not_called()
        await self.assert_ready_again()

    async def test_oversized_prompt_is_rejected(self):
        response = await self.client.post("/api/analyze", data=form_for(prompt="x" * (server.MAX_FIELD_BYTES + 1)))
        await self.assert_response(response, 413, "upload_too_large")
        self.mock_single.assert_not_called()
        await self.assert_ready_again()

    async def test_invalid_form_options_are_rejected(self):
        for fields in ({"mode": "wrong"}, {"tta": "1"}, {"rag": "yes"}, {"unknown": "value"}):
            with self.subTest(fields=fields):
                response = await self.client.post("/api/analyze", data=form_for(**fields))
                await self.assert_response(response, 400, "invalid_request")
        duplicate = form_for(rag="true")
        duplicate.add_field("rag", "false")
        response = await self.client.post("/api/analyze", data=duplicate)
        await self.assert_response(response, 400, "invalid_request")
        self.mock_single.assert_not_called()
        await self.assert_ready_again()

    async def test_missing_image_and_non_multipart_are_rejected(self):
        form = FormData()
        form.add_field("prompt", "why", content_type="text/plain")
        response = await self.client.post("/api/analyze", data=form)
        await self.assert_response(response, 400, "missing_image")
        response = await self.client.post("/api/analyze", json={"image": "not a file"})
        await self.assert_response(response, 400, "invalid_request")
        self.mock_single.assert_not_called()
        await self.assert_ready_again()

    async def test_internal_error_returns_json_and_releases_worker(self):
        self.mock_single.side_effect = RuntimeError("inference failed")
        with patch.object(server.traceback, "print_exc"):
            response = await self.client.post("/api/analyze", data=form_for())
        data = await self.assert_response(response, 500)
        self.assertEqual(data["error"], "server_error")
        self.mock_single.side_effect = None
        await self.assert_ready_again()

    async def test_inference_and_decoding_do_not_block_health_and_busy_is_bounded(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        worker_threads = []
        real_decode = server.decode_image

        def decoding(payload):
            worker_threads.append(threading.get_ident())
            return real_decode(payload)

        def slow_inference(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            if not release.wait(5):
                raise RuntimeError("test worker release timed out")
            return {"mode": "single", "summary": "done"}

        self.mock_single.side_effect = slow_inference
        with patch.object(server, "decode_image", decoding):
            first = asyncio.create_task(self.client.post("/api/analyze", data=form_for()))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                health = await asyncio.wait_for(self.client.get("/api/health"), timeout=0.5)
                await self.assert_response(health, 200)
                busy = await asyncio.wait_for(self.client.post("/api/analyze", data=form_for()), timeout=0.5)
                await self.assert_response(busy, 503, "server_busy")
                self.assertEqual(self.mock_single.call_count, 1)
                self.assertTrue(worker_threads)
                self.assertNotIn(loop_thread, worker_threads)
            finally:
                release.set()
                response = await first
                await self.assert_response(response, 200)
        await self.assert_ready_again()

    async def test_cancelled_http_request_keeps_slot_until_worker_finishes(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        active = 0
        max_active = 0

        def slow_inference(*args, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            loop.call_soon_threadsafe(started.set)
            try:
                if not release.wait(5):
                    raise RuntimeError("test worker release timed out")
                return {"mode": "single"}
            finally:
                active -= 1

        self.mock_single.side_effect = slow_inference
        first = asyncio.create_task(self.client.post("/api/analyze", data=form_for()))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            # TestServer enables real HTTP handler cancellation on disconnect.
            await asyncio.wait_for(self.cancelled.wait(), timeout=1)
            busy = await asyncio.wait_for(self.client.post("/api/analyze", data=form_for()), timeout=0.5)
            await self.assert_response(busy, 503, "server_busy")
            self.assertEqual(self.mock_single.call_count, 1)
        finally:
            release.set()
            if not first.done():
                await first

        async def wait_until_idle():
            while self.app[server.ANALYSIS_WORKER_KEY].busy:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_until_idle(), timeout=1)
        await self.assert_ready_again()
        self.assertEqual(max_active, 1)

    async def test_cleanup_shuts_down_worker(self):
        worker = self.app[server.ANALYSIS_WORKER_KEY]
        await self.client.close()
        self.assertFalse(worker.claim())
        with self.assertRaises(RuntimeError):
            worker.executor.submit(lambda: None)


class RecognitionContractTests(unittest.TestCase):
    def setUp(self):
        # Keep NumPy loaded when patch.dict restores sys.modules between tests.
        importlib.import_module("numpy")
        quality = types.ModuleType("image_quality")
        quality.assess_image_quality = Mock(return_value={"overall": 0.1})
        knowledge = types.ModuleType("calligrapher_tool")
        self.search = knowledge.search_knowledge = Mock()
        self.search.invoke.return_value = "风格材料"
        segmentation = types.ModuleType("segment")
        segmentation.segment_auto = Mock(return_value={"boxes": [[0, 0, 4, 8], [4, 0, 4, 8]], "method": "stub"})
        self.modules = patch.dict(sys.modules, {
            "image_quality": quality, "calligrapher_tool": knowledge, "segment": segmentation,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.recognizer = Mock()
        self.recognizer.recognize.return_value = self.recognizer.predict_with_cam.return_value = {
            "calligrapher": "王羲之", "confidence": 0.8, "all_probabilities": {"王羲之": 0.8},
        }
        self.recognizer_patch = patch.object(server, "get_recognizer", return_value=self.recognizer)
        self.recognizer_patch.start()
        self.addCleanup(self.recognizer_patch.stop)
        self.image = Image.new("RGB", (8, 8), "white")
        self.addCleanup(self.image.close)

    def test_tta_flag_reaches_cam_and_normal_inference(self):
        for use_cam in (False, True):
            for use_tta in (False, True):
                with self.subTest(cam=use_cam, tta=use_tta):
                    self.recognizer.reset_mock()
                    result = server.run_single(self.image, use_cam=use_cam, use_tta=use_tta,
                                               use_rag=False, prompt="")
                    self.assertNotIn("blocked", result)
                    selected = self.recognizer.predict_with_cam if use_cam else self.recognizer.recognize
                    unused = self.recognizer.recognize if use_cam else self.recognizer.predict_with_cam
                    self.assertEqual(selected.call_args.kwargs, {"tta": use_tta})
                    selected.assert_called_once()
                    unused.assert_not_called()

    def test_single_and_multi_queries_include_author_and_user_question(self):
        prompt = "这是谁的字？为什么？"
        for mode in ("single", "multi"):
            for author in ("王羲之", "颜真卿"):
                with self.subTest(mode=mode, author=author):
                    result = {"calligrapher": author, "confidence": 0.8, "all_probabilities": {author: 0.8}}
                    self.recognizer.recognize.return_value = result
                    if mode == "single":
                        response = server.run_single(self.image, use_cam=False, use_tta=False, use_rag=True, prompt=prompt)
                    else:
                        response = server.run_multi(self.image, use_tta=False, use_rag=True, prompt=prompt)
                    self.assertNotIn("blocked", response)
                    query = self.search.invoke.call_args.args[0]["query"]
                    self.assertIn(author, query)
                    self.assertIn(prompt, query)
                    self.assertEqual(response["knowledge"], "风格材料")


if __name__ == "__main__":
    unittest.main()
