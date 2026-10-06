"""Persistence configuration must match health checks from any launch directory."""
import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from tests.unit.test_agent_models import replace_modules
from zhimo.config.settings import PROJECT_ROOT, get_settings


class KnowledgeConfigTests(unittest.TestCase):
    def test_vectorstore_uses_normalized_current_settings(self):
        chroma = types.ModuleType("langchain_chroma")
        chroma.Chroma = Mock()
        ollama = types.ModuleType("langchain_ollama")
        ollama.OllamaEmbeddings = Mock()
        spec = importlib.util.spec_from_file_location(
            "zhimo.knowledge._test_chroma", PROJECT_ROOT / "src/zhimo/knowledge/chroma.py")
        module = importlib.util.module_from_spec(spec)
        with replace_modules({chroma.__name__: chroma, ollama.__name__: ollama}):
            spec.loader.exec_module(module)
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as other:
            try:
                os.chdir(other)
                for configured, expected in ((None, PROJECT_ROOT / "chroma_db"),
                                              ("custom/db", PROJECT_ROOT / "custom/db"),
                                              (other, Path(other)),
                                              ("~/zhimo-test-db", Path.home() / "zhimo-test-db")):
                    with self.subTest(configured=configured), patch.dict(os.environ, {}, clear=True):
                        if configured is not None:
                            os.environ["ZHIMO_CHROMA_DIR"] = configured
                        module.build_vectorstore()
                        actual = chroma.Chroma.call_args.kwargs["persist_directory"]
                        self.assertEqual(actual, str(expected.resolve()))
                        self.assertEqual(actual, str(get_settings().chroma_dir))
            finally:
                os.chdir(original_cwd)
