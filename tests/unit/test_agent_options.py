"""Tool option enforcement without model weights or external services."""
import os
import types
import unittest
from unittest.mock import Mock, patch

from tests.unit import test_agent_models as model_tests
from tests.unit.test_agent_models import load_agent_tools, replace_modules


class AgentOptionTests(unittest.TestCase):
    def setUp(self):
        self.tools = load_agent_tools()
        self.image = model_tests.AgentModelTests.image_state()
        self.recognizer = Mock()
        self.recognizer.recognize.return_value = {
            "calligrapher": "王羲之", "confidence": 0.9,
            "all_probabilities": {"王羲之": 0.9}, "model_backbone": "test",
        }
        self.recognizer.predict_with_cam.return_value = {
            **self.recognizer.recognize.return_value, "evidence": {"heatmap": "test"},
        }
        self.tools.get_recognizer = Mock(return_value=self.recognizer)

    def test_cam_and_tta_choose_the_requested_path(self):
        for cam in (True, False):
            for tta in (True, False):
                with self.subTest(cam=cam, tta=tta):
                    self.recognizer.reset_mock()
                    self.image["analysis_options"] = {"cam": cam, "tta": tta}
                    result = self.tools.analyze_calligraphy(self.image)
                    selected = self.recognizer.predict_with_cam if cam else self.recognizer.recognize
                    unused = self.recognizer.recognize if cam else self.recognizer.predict_with_cam
                    selected.assert_called_once()
                    self.assertEqual(selected.call_args.kwargs, {"tta": tta})
                    unused.assert_not_called()
                    self.assertEqual(bool(result["evidence"]), cam)

    def test_identify_and_each_character_receive_tta(self):
        segment = types.ModuleType("zhimo.vision.segmentation_impl")
        segment.segment_auto = Mock(return_value={"boxes": [(0, 0, 20, 20), (40, 40, 20, 20)], "method": "test"})
        for tta in (True, False):
            self.recognizer.reset_mock()
            self.image["analysis_options"] = {"tta": tta}
            self.tools.identify_calligrapher(self.image)
            with replace_modules({segment.__name__: segment}):
                self.tools.analyze_multi_char(self.image)
            self.assertEqual(self.recognizer.recognize.call_count, 3)
            self.assertTrue(all(call.kwargs == {"tta": tta} for call in self.recognizer.recognize.call_args_list))

    def test_disallowed_mode_cannot_load_a_model_or_segment(self):
        for mode, tools in (("multi", ("identify_calligrapher", "analyze_calligraphy")),
                            ("single", ("analyze_multi_char",))):
            for tool in tools:
                with self.subTest(mode=mode, tool=tool):
                    self.image["analysis_options"] = {"analysis_mode": mode}
                    result = getattr(self.tools, tool)(self.image)
                    self.assertIn("error", result)
        self.tools.get_recognizer.assert_not_called()

    def test_rag_disabled_does_not_initialize_or_query_vectorstore(self):
        self.tools.get_vectorstore = Mock()
        result = self.tools.search_knowledge("王羲之", {"analysis_options": {"rag": False}})
        self.assertIn("关闭", result)
        self.tools.get_vectorstore.assert_not_called()
        self.tools.get_vectorstore.return_value.similarity_search.return_value = []
        for state in (None, {"analysis_options": {"rag": True}}):
            self.tools.search_knowledge("王羲之", state)
        self.assertEqual(self.tools.get_vectorstore.call_count, 2)

    def test_vectorstore_cache_tracks_configured_directory(self):
        self.tools.build_vectorstore = Mock(side_effect=[object(), object()])
        with patch.dict(os.environ, {"ZHIMO_CHROMA_DIR": "first-db"}):
            first = self.tools.get_vectorstore()
            self.assertIs(self.tools.get_vectorstore(), first)
            os.environ["ZHIMO_CHROMA_DIR"] = "second-db"
            self.assertIsNot(self.tools.get_vectorstore(), first)
        self.assertEqual(self.tools.build_vectorstore.call_count, 2)
