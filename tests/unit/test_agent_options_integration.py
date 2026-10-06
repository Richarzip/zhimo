"""Real LangChain graph/tool injection; optional when Agent dependencies are absent."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import uuid
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("langchain"), "optional LangChain dependencies not installed")
class AgentGraphOptionTests(unittest.TestCase):
    def test_real_agent_applies_new_options_without_reusing_checkpoint_values(self):
        from PIL import Image
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver
        from tests.unit.test_agent_models import replace_modules

        prompts = []

        class ToolCallingModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, *args, **kwargs):
                prompts.append(messages[0].content)
                result = super()._generate(messages, *args, **kwargs)
                result.generations[0].message = result.generations[0].message.model_copy(update={"id": uuid.uuid4().hex})
                return result

        # Tool requests intentionally ignore RAG/CAM preferences. Tools must still
        # enforce injected state, independent of model cooperation.
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[
                {"id": "analyze", "name": "analyze_calligraphy", "args": {}},
                {"id": "search", "name": "search_knowledge", "args": {"query": "王羲之"}},
            ]),
            AIMessage(content="本轮回答"),
        ])
        chroma = types.ModuleType("zhimo.knowledge.chroma")
        chroma.build_vectorstore = Mock(side_effect=AssertionError("must not open a real DB"))
        with replace_modules({chroma.__name__: chroma}):
            from zhimo.agent import tools_impl, builder_impl
        recognizer = Mock()
        recognizer.recognize.return_value = {
            "calligrapher": "王羲之", "confidence": 0.9,
            "all_probabilities": {"王羲之": 0.9}, "model_backbone": "test",
        }
        recognizer.predict_with_cam.return_value = {
            **recognizer.recognize.return_value, "evidence": {"heatmap": "test"},
        }
        vectorstore = Mock()
        vectorstore.similarity_search.return_value = []
        from zhimo.application.pipeline import run_chat
        with patch.object(builder_impl, "init_chat_model", return_value=model), \
             patch.object(tools_impl, "get_recognizer", return_value=recognizer), \
             patch.object(tools_impl, "get_vectorstore", return_value=vectorstore) as get_store:
            agent = builder_impl.create_calligraphy_agent(checkpointer=MemorySaver())
            for rag, cam, tta in ((False, False, True), (True, True, False)):
                recognizer.reset_mock()
                get_store.reset_mock()
                response = run_chat(
                    Image.new("RGB", (32, 32), "white"),
                    {"text": "重新分析", "session_id": "same-thread", "denoise": "false",
                     "rag": str(rag).lower(), "cam": str(cam).lower(), "tta": str(tta).lower(),
                     "analysis_mode": "single"},
                    get_agent=lambda: agent, denoise=Mock(), classify=lambda e: {"error": str(e)},
                    finalize=lambda data, *args, **kwargs: data,
                )
                self.assertNotIn("error", response, response)
                self.assertEqual(response["reply"], "本轮回答")
                selected = recognizer.predict_with_cam if cam else recognizer.recognize
                unused = recognizer.recognize if cam else recognizer.predict_with_cam
                self.assertEqual(selected.call_args.kwargs, {"tta": tta})
                unused.assert_not_called()
                self.assertEqual(get_store.call_count, int(rag))
                self.assertIn("RAG 已开启" if rag else "RAG 已关闭", prompts[-1])
                self.assertIn("CAM 已开启" if cam else "CAM 已关闭", prompts[-1])
                self.assertIn("用户选择单字分析", prompts[-1])
                snapshot = agent.get_state({"configurable": {"thread_id": "same-thread"}})
                self.assertEqual(snapshot.values["analysis_options"]["tta"], tta)
        # Existing deterministic HTTP single/multi pipelines invoke retrieval
        # directly, outside ToolNode, and must retain the default enabled behavior.
        with patch.object(tools_impl, "get_vectorstore", return_value=vectorstore):
            self.assertIn("没有找到", tools_impl.search_knowledge.invoke({"query": "王羲之"}))
        self.assertNotIn("state", tools_impl.search_knowledge.tool_call_schema.model_json_schema()["properties"])
