"""Agent tools share application model configuration without loading weights."""

import base64
from contextlib import ExitStack, contextmanager
import importlib.util
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from zhimo.application import factory
from zhimo.config.settings import PROJECT_ROOT


@contextmanager
def replace_modules(modules):
    """Restore only substitutes, preserving newly imported native modules."""
    missing = object()
    original = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in original.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_agent_tools():
    """Replace external tool/RAG integrations, keeping production tool bodies."""
    langchain = types.ModuleType("langchain")
    langchain_tools = types.ModuleType("langchain.tools")
    langchain_tools.tool = lambda function: function
    langgraph = types.ModuleType("langgraph")
    langgraph_prebuilt = types.ModuleType("langgraph.prebuilt")
    langgraph_prebuilt.InjectedState = type("InjectedState", (), {})
    chroma = types.ModuleType("zhimo.knowledge.chroma")
    chroma.build_vectorstore = Mock()
    spec = importlib.util.spec_from_file_location(
        "zhimo.agent._test_tools_impl", ROOT / "src/zhimo/agent/tools_impl.py"
    )
    module = importlib.util.module_from_spec(spec)
    with replace_modules({
        "langchain": langchain,
        "langchain.tools": langchain_tools,
        "langgraph": langgraph,
        "langgraph.prebuilt": langgraph_prebuilt,
        "zhimo.knowledge.chroma": chroma,
    }):
        spec.loader.exec_module(module)
    return module


class SingleRecognizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.recognize_calls = 0

    def recognize(self, image, *, tta=False):
        self.recognize_calls += 1
        return {
            "calligrapher": "王羲之",
            "confidence": 0.9,
            "all_probabilities": {"王羲之": 0.9, "颜真卿": 0.1},
            "model_backbone": "single-test-model",
        }

    def predict_with_cam(self, image, *, tta=False):
        return {**self.recognize(image, tta=tta), "evidence": {"attention_note": "test CAM"}}


class EnsembleRecognizer(SingleRecognizer):
    def __init__(self, model_paths):
        super().__init__(model_path=model_paths[0])
        self.model_paths = model_paths

    def recognize(self, image, *, tta=False):
        return {
            **super().recognize(image, tta=tta),
            "ensemble": {"strategy": "soft_voting", "members": [], "agreement": True},
        }


class AgentModelTests(unittest.TestCase):
    def setUp(self):
        self.tools = load_agent_tools()
        self.single = Mock(side_effect=SingleRecognizer)
        self.ensemble = Mock(side_effect=EnsembleRecognizer)
        vision = types.ModuleType("zhimo.vision")
        vision.CalligrapherRecognizer = self.single
        vision.EnsembleRecognizer = self.ensemble
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(patch.dict(os.environ, {}, clear=True))
        contexts.enter_context(replace_modules({"zhimo.vision": vision}))

    @staticmethod
    def image_state():
        image = Image.new("RGB", (80, 80), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"messages": [types.SimpleNamespace(
            type="human",
            content=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}],
        )]}

    def test_defaults_match_application_factory_and_reuse_instance(self):
        expected = [str(PROJECT_ROOT / "checkpoints" / name) for name in ("convnext.pth", "swin.pth")]
        with patch.object(factory, "get_recognizer", wraps=factory.get_recognizer) as build:
            recognizer = self.tools.get_recognizer()
            self.assertIs(self.tools.get_recognizer(), recognizer)
            build.assert_called_once_with()
        self.ensemble.assert_called_once_with(model_paths=expected)
        self.single.assert_not_called()

    def test_single_weight_overrides_use_single_model(self):
        for variable in ("ZHIMO_MODEL_PATH", "ZHIMO_MODEL_PATHS"):
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: "weights/custom.pth"}, clear=True):
                recognizer = self.tools.get_recognizer()
                self.assertIs(type(recognizer), SingleRecognizer)
                self.assertEqual(recognizer.model_path, str(PROJECT_ROOT / "weights/custom.pth"))
        self.single.assert_called_once()
        self.ensemble.assert_not_called()

    def test_multiple_weights_preserve_order_and_absolute_paths(self):
        absolute = Path(tempfile.gettempdir()) / "zhimo-custom.pth"
        os.environ["ZHIMO_MODEL_PATHS"] = os.pathsep.join(("weights/first.pth", str(absolute)))
        recognizer = self.tools.get_recognizer()
        expected = [str(PROJECT_ROOT / "weights/first.pth"), str(absolute)]
        self.assertEqual(recognizer.model_paths, expected)
        self.ensemble.assert_called_once_with(model_paths=expected)
        self.single.assert_not_called()

    def test_model_cache_does_not_reuse_a_different_configuration(self):
        os.environ["ZHIMO_MODEL_PATHS"] = "weights/first.pth"
        first = self.tools.get_recognizer()
        os.environ["ZHIMO_MODEL_PATHS"] = os.pathsep.join(("weights/first.pth", "weights/second.pth"))
        second = self.tools.get_recognizer()
        self.assertIsNot(first, second)
        self.assertIs(self.tools.get_recognizer(), second)
        os.environ["ZHIMO_MODEL_PATHS"] = "weights/third.pth"
        third = self.tools.get_recognizer()
        self.assertIsNot(third, first)
        self.assertIsNot(third, second)
        self.assertEqual(third.model_path, str(PROJECT_ROOT / "weights/third.pth"))
        self.assertEqual(self.single.call_count, 2)
        self.ensemble.assert_called_once()

    def test_model_paths_do_not_depend_on_working_directory(self):
        os.environ["ZHIMO_MODEL_PATH"] = "weights/custom.pth"
        initial = self.tools.get_recognizer()
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as other_cwd:
            try:
                os.chdir(other_cwd)
                self.assertIs(self.tools.get_recognizer(), initial)
                fresh_tools = load_agent_tools()
                self.assertEqual(fresh_tools.get_recognizer().model_path, initial.model_path)
            finally:
                os.chdir(original_cwd)
        self.assertEqual(initial.model_path, str(PROJECT_ROOT / "weights/custom.pth"))

    def test_single_model_identification_and_analysis_accept_missing_ensemble(self):
        os.environ["ZHIMO_MODEL_PATH"] = "weights/custom.pth"
        state = self.image_state()
        for name in ("identify_calligrapher", "analyze_calligraphy"):
            with self.subTest(tool=name):
                result = getattr(self.tools, name)(state)
                self.assertEqual(result["calligrapher"], "王羲之")
                self.assertEqual(result["top_k"][0], {"name": "王羲之", "confidence": 0.9})
                self.assertIsNone(result["ensemble"])
                self.assertEqual(result["model_backbone"], "single-test-model")
        self.single.assert_called_once_with(model_path=str(PROJECT_ROOT / "weights/custom.pth"))
        self.assertEqual(self.tools.get_recognizer().recognize_calls, 2)

    def test_ensemble_details_are_preserved(self):
        for name in ("identify_calligrapher", "analyze_calligraphy"):
            with self.subTest(tool=name):
                result = getattr(self.tools, name)(self.image_state())
                self.assertEqual(result["ensemble"]["strategy"], "soft_voting")
                self.assertTrue(result["ensemble"]["agreement"])
        self.ensemble.assert_called_once()

    def test_multi_character_tool_uses_configured_single_model(self):
        os.environ["ZHIMO_MODEL_PATH"] = "weights/custom.pth"
        segmentation = types.ModuleType("zhimo.vision.segmentation_impl")
        segmentation.segment_auto = Mock(return_value={
            "boxes": [(0, 0, 30, 30), (40, 40, 30, 30)], "method": "test",
        })
        with replace_modules({"zhimo.vision.segmentation_impl": segmentation}):
            result = self.tools.analyze_multi_char(self.image_state())
        self.assertEqual(result["calligrapher"], "王羲之")
        self.assertEqual(result["num_characters"], 2)
        self.assertEqual(self.tools.get_recognizer().recognize_calls, 2)
        self.single.assert_called_once()
        self.ensemble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
