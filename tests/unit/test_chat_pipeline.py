"""Chat regressions without a model service or LangChain installation."""

import base64
import json
from io import BytesIO
import sys
import types
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from zhimo.application.outputs import classify_exception, finalize_outputs
from zhimo.application.pipeline import run_chat
from zhimo.vision.denoise import denoise_image
# Load PDF dependencies before patch.dict restores sys.modules between tests.
from zhimo.application import report


def message(kind, content, identifier, **kwargs):
    return types.SimpleNamespace(type=kind, content=content, id=identifier, **kwargs)


class HumanMessage:
    def __init__(self, *, content, id=None):
        self.type = "human"
        self.content = content
        self.id = id


class ChatPipelineTests(unittest.TestCase):
    def setUp(self):
        messages = types.ModuleType("langchain.messages")
        messages.HumanMessage = HumanMessage
        patcher = patch.dict(sys.modules, {
            "langchain": types.ModuleType("langchain"),
            "langchain.messages": messages,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.agent = Mock()
        self.agent.get_state.return_value = types.SimpleNamespace(values={"messages": []})
        self.agent.invoke.side_effect = lambda inputs, config: {
            "messages": inputs["messages"] + [message("ai", "本轮回答", "answer")]
        }
        self.denoise = Mock(side_effect=lambda image: (image, {"enabled": True}))

    def run_chat(self, image=None, **fields):
        return run_chat(
            image,
            {"session_id": "existing-session", **fields},
            get_agent=lambda: self.agent,
            denoise=self.denoise,
            classify=classify_exception,
            finalize=partial(finalize_outputs, project_root=ROOT),
        )

    def test_text_only_turn_uses_existing_session_without_image_processing(self):
        result = self.run_chat(text="  王羲之的代表作有哪些？  ")
        self.assertEqual(result["reply"], "本轮回答")
        self.assertEqual(result["mode"], "chat")
        self.assertEqual(result["session_id"], "existing-session")
        self.assertEqual(result["denoise"]["enabled"], False)
        self.denoise.assert_not_called()
        inputs = self.agent.invoke.call_args.args[0]
        self.assertEqual(inputs["messages"][0].content, [
            {"type": "text", "text": "王羲之的代表作有哪些？"},
        ])
        self.assertEqual(self.agent.invoke.call_args.kwargs["config"],
                         {"configurable": {"thread_id": "existing-session"}})

    def test_analysis_options_are_sent_for_each_turn_and_reset_to_defaults(self):
        selections = [
            {"rag": "false", "cam": "false", "tta": "true", "analysis_mode": "multi"},
            {"rag": "true", "cam": "true", "tta": "false", "analysis_mode": "single"},
            {},
        ]
        expected = [
            {"rag": False, "cam": False, "tta": True, "analysis_mode": "multi"},
            {"rag": True, "cam": True, "tta": False, "analysis_mode": "single"},
            {"rag": True, "cam": True, "tta": False, "analysis_mode": "auto"},
        ]
        for fields, options in zip(selections, expected):
            self.run_chat(text="重新分析", **fields)
            self.assertEqual(self.agent.invoke.call_args.args[0]["analysis_options"], options)

    def test_prompt_alias_and_text_only_pdf_are_supported(self):
        result = self.run_chat(text="  ", prompt="介绍一下书法风格", pdf="true", examples="true")
        self.assertEqual(result["reply"], "本轮回答")
        self.assertEqual(result["reference_examples"], [])
        self.assertTrue(base64.b64decode(result["pdf"]["data_url"].split(",", 1)[1]).startswith(b"%PDF-"))
        self.denoise.assert_not_called()

    def test_empty_turn_does_not_invoke_agent(self):
        with self.assertRaisesRegex(ValueError, "输入问题"):
            self.run_chat(text="  ")
        self.agent.invoke.assert_not_called()

    def test_image_turn_still_sends_denoised_image(self):
        with Image.new("RGB", (8, 8), "white") as image:
            result = self.run_chat(image=image)
        self.assertEqual(result["reply"], "本轮回答")
        self.denoise.assert_called_once()
        content = self.agent.invoke.call_args.args[0]["messages"][0].content
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_compacted_history_keeps_this_turn_reply_and_recognition(self):
        history = [message("ai", "旧回答", f"old-{i}") for i in range(30)]
        self.agent.get_state.return_value.values["messages"] = history
        tool = message("tool", json.dumps({"calligrapher": "王羲之", "confidence": 0.8}),
                       "current-tool", name="identify_calligrapher")
        self.agent.invoke.side_effect = lambda inputs, config: {"messages": [
            message("human", "历史摘要", "summary"),
            *history[-2:], *inputs["messages"], tool,
            message("ai", "压缩后仍应返回本轮回答", "current-answer"),
        ]}
        result = self.run_chat(text="这幅字是谁写的？")
        self.assertEqual(result["reply"], "压缩后仍应返回本轮回答")
        self.assertEqual(result["recognition"]["calligrapher"], "王羲之")
        self.assertEqual(result["mode"], "single")
        self.assertEqual(len(result["steps"]), 1)
        self.assertLess(result["message_count"], len(history))

    def test_real_adaptive_output_and_diagnostics_reach_image_chat(self):
        image = Image.new("RGB", (128, 128), "white")
        for x in range(12, 110):
            image.putpixel((x, 30), (0, 0, 0))
        for x in range(20, 110, 5):
            for y in range(70, 110, 5):
                image.putpixel((x, y), (0, 0, 0))
        self.denoise.side_effect = denoise_image
        result = self.run_chat(image=image, denoise="true")
        self.assertEqual(result["denoise"]["mode"], "adaptive")
        self.assertTrue(result["denoise"]["applied"])
        payload = self.agent.invoke.call_args.args[0]["messages"][0].content[0]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(payload.split(",", 1)[1]))) as sent:
            self.assertEqual(sent.getpixel((20, 70)), (255, 255, 255))
            self.assertEqual(sent.getpixel((20, 30)), (0, 0, 0))
        self.assertEqual(image.getpixel((20, 70)), (0, 0, 0))
        self.assertIn("quality_before", result["denoise"])
        self.assertIn("quality_after", result["denoise"])

        self.denoise.reset_mock()
        disabled = self.run_chat(image=image, denoise="false")
        self.denoise.assert_not_called()
        self.assertEqual(disabled["denoise"]["method"], "disabled")

    def test_compaction_can_remove_current_human_message_without_replaying_old_tools(self):
        old_tool = message("tool", json.dumps({"calligrapher": "旧作者"}), "old-tool", name="old-tool")
        self.agent.get_state.return_value.values["messages"] = [old_tool] + [
            message("ai", "旧回答", f"old-{i}") for i in range(30)
        ]
        # A checkpoint round-trip recreates message objects but retains their IDs.
        old_tool_copy = message("tool", old_tool.content, "old-tool", name="old-tool")
        self.agent.invoke.return_value = {"messages": [
            message("human", "摘要替代了当前用户消息", "new-summary"),
            old_tool_copy, message("ai", "本轮纯文字解释", "new-answer"),
        ]}
        self.agent.invoke.side_effect = None
        result = self.run_chat(text="请解释上一次的结果")
        self.assertEqual(result["reply"], "本轮纯文字解释")
        self.assertEqual(result["recognition"], {})
        self.assertEqual(result["steps"], [])

    def test_normal_followup_does_not_reuse_prior_recognition(self):
        history = [message("tool", json.dumps({"calligrapher": "旧作者"}), "old-tool", name="identify")]
        self.agent.get_state.return_value.values["messages"] = history
        self.agent.invoke.side_effect = lambda inputs, config: {"messages": [
            *history, *inputs["messages"], message("ai", "这是对上一轮的解释", "followup"),
        ]}
        result = self.run_chat(text="为什么？")
        self.assertEqual(result["reply"], "这是对上一轮的解释")
        self.assertEqual(result["recognition"], {})
        self.assertEqual(result["steps"], [])

    def test_text_only_agent_failure_retains_structured_error(self):
        self.agent.invoke.side_effect = RuntimeError("temporary failure")
        result = self.run_chat(text="介绍一下书法")
        self.assertEqual(result["error"], "agent_error")
        self.assertEqual(result["session_id"], "existing-session")
        self.assertIn("temporary failure", result["diagnostic"]["raw"])


if __name__ == "__main__":
    unittest.main()
