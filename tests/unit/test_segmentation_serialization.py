"""Segmentation coordinates must survive real LangChain ToolMessage encoding."""

import base64
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from zhimo.vision import segmentation_impl as segmentation


class SegmentationJsonTests(unittest.TestCase):
    def test_public_segmentation_results_use_json_safe_python_coordinates(self):
        with Image.open(ROOT / "image_test/test1.png") as image:
            array = np.asarray(image.convert("RGB"))
        result = segmentation.segment_auto(array)
        self.assertEqual(result["num_boxes"], 4)
        for boxes in (result["boxes"], segmentation.segment_by_connected_components(array),
                      segmentation.segment_by_projection(array)):
            self.assertTrue(boxes)
            self.assertTrue(all(type(value) is int for box in boxes for value in box))
            self.assertEqual(json.loads(json.dumps(boxes)), [list(box) for box in boxes])
        self.assertEqual(len(json.loads(json.dumps(result))["boxes"]), 4)


@unittest.skipUnless(importlib.util.find_spec("langchain"), "optional LangChain dependencies not installed")
class MultiCharacterToolSerializationTests(unittest.TestCase):
    def test_numpy_coordinates_are_normalized_before_tool_message_serialization(self):
        from tests.unit.test_agent_models import load_agent_tools, replace_modules

        tools = load_agent_tools()
        segmentation_module = types.ModuleType("zhimo.vision.segmentation_impl")
        segmentation_module.segment_auto = Mock(return_value={
            "boxes": [(np.int64(2), np.int64(3), np.int64(20), np.int64(21))],
            "method": "numpy-test",
        })
        tools.get_recognizer = Mock(return_value=Mock(recognize=Mock(return_value={
            "calligrapher": "test", "confidence": 0.8,
        })))
        image_buffer = BytesIO()
        Image.new("RGB", (8, 8), "white").save(image_buffer, format="PNG")
        state = {
            "messages": [],
            "analysis_options": {"analysis_mode": "multi", "rag": False},
        }
        from langchain_core.messages import HumanMessage
        from langchain_core.messages import convert_to_messages
        state["messages"] = convert_to_messages([HumanMessage(content=[
            {"type": "text", "text": "test"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
             base64.b64encode(image_buffer.getvalue()).decode("ascii")}},
        ])])
        with replace_modules({segmentation_module.__name__: segmentation_module}):
            result = tools.analyze_multi_char(state)
        self.assertEqual(result["per_char_results"][0]["bbox"], [2, 3, 20, 21])
        self.assertTrue(all(type(value) is int for value in result["per_char_results"][0]["bbox"]))
        json.dumps(result)

    def test_real_tool_message_retains_four_characters_in_chat_pipeline(self):
        from typing import Annotated, TypedDict
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode
        from tests.unit.test_agent_models import replace_modules
        from zhimo.application.pipeline import parse_tool_result, run_chat

        # Keep the real tool decorator and graph. Only recognition and the unused
        # knowledge dependency are replaced; segmentation uses the actual image.
        chroma = types.ModuleType("zhimo.knowledge.chroma")
        chroma.build_vectorstore = Mock(side_effect=AssertionError("no real DB in this test"))
        module_name = "zhimo.agent._test_multi_serialization_tools"
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "src/zhimo/agent/tools_impl.py")
        tools = importlib.util.module_from_spec(spec)
        with replace_modules({module_name: tools, chroma.__name__: chroma}):
            spec.loader.exec_module(tools)

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            analysis_options: dict

        builder = StateGraph(State)
        builder.add_node("tools", ToolNode([tools.analyze_multi_char]))
        builder.set_entry_point("tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        tool_messages = []

        class LocalToolAgent:
            def get_state(self, config):
                return types.SimpleNamespace(values={})

            def invoke(self, state, config):
                call = AIMessage(content="", tool_calls=[{
                    "id": "multi-test", "name": "analyze_multi_char", "args": {},
                }])
                result = graph.invoke({**state, "messages": [*state["messages"], call]})
                tool_messages.extend(message for message in result["messages"] if isinstance(message, ToolMessage))
                return result

        recognizer = Mock()
        recognizer.recognize.return_value = {"calligrapher": "沙孟海", "confidence": 0.8}
        with Image.open(ROOT / "image_test/test1.png") as source:
            image = source.convert("RGB")
        with patch.object(tools, "get_recognizer", return_value=recognizer):
            response = run_chat(
                image, {"text": "分析四字作品", "analysis_mode": "multi", "denoise": "false",
                        "rag": "false", "cam": "false", "tta": "false"},
                get_agent=LocalToolAgent, denoise=Mock(), classify=lambda error: {"error": str(error)},
                finalize=lambda data, *args, **kwargs: data,
            )

        self.assertEqual(len(tool_messages), 1, response)
        parsed = parse_tool_result(tool_messages[0].content)
        self.assertNotIn("raw", parsed)
        self.assertEqual(parsed["num_characters"], 4)
        self.assertEqual(len(parsed["per_char_results"]), 4)
        self.assertTrue(all(type(value) is int for item in parsed["per_char_results"] for value in item["bbox"]))
        self.assertEqual(recognizer.recognize.call_count, 4)
        self.assertEqual(response["mode"], "multi", response)
        self.assertEqual(response["recognition"]["calligrapher"], "沙孟海")
        self.assertEqual(response["recognition"]["num_characters"], 4)
        self.assertEqual(len(response["recognition"]["per_char_results"]), 4)


if __name__ == "__main__":
    unittest.main()
