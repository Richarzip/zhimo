import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from zhimo.application.contracts import AnalysisRequest, AnalysisResponse
from zhimo.config.settings import get_settings


class InterfaceTests(unittest.TestCase):
    def test_default_settings_expose_two_model_paths(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        self.assertEqual(len(settings.model_paths), 2)
        self.assertEqual(settings.model_paths[0].name, "convnext.pth")
        self.assertEqual(settings.model_paths[1].name, "swin.pth")

    def test_model_paths_can_be_overridden(self):
        with patch.dict(os.environ, {"ZHIMO_MODEL_PATHS": "a.pth;b.pth"}, clear=True):
            settings = get_settings()
        self.assertEqual([path.name for path in settings.model_paths], ["a.pth", "b.pth"])

    def test_response_contract_is_serializable(self):
        response = AnalysisResponse(mode="single", recognition={"calligrapher": "王羲之"})
        self.assertEqual(response.to_dict()["mode"], "single")
        self.assertEqual(AnalysisRequest(image="image").mode, "single")


if __name__ == "__main__":
    unittest.main()
