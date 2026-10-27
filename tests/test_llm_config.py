"""Offline checks for the shared Gemini model configuration."""

import os
import unittest
from unittest.mock import patch

from llm.llm_config import DEFAULT_GEMINI_MODEL, get_gemini_model, get_ragas_model


class LLMConfigTests(unittest.TestCase):
    def test_explicit_override_has_highest_priority(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "env-model"}):
            self.assertEqual(get_gemini_model("cli-model"), "cli-model")

    def test_environment_then_project_default(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "env-model"}):
            self.assertEqual(get_gemini_model(), "env-model")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_gemini_model(), DEFAULT_GEMINI_MODEL)

    def test_ragas_override_can_differ_or_follow_application_model(self):
        with patch.dict(
            os.environ,
            {"GEMINI_MODEL": "app-model", "RAGAS_MODEL": "judge-model"},
        ):
            self.assertEqual(get_ragas_model(), "judge-model")
            self.assertEqual(get_ragas_model("cli-judge"), "cli-judge")
        with patch.dict(os.environ, {"GEMINI_MODEL": "app-model"}, clear=True):
            self.assertEqual(get_ragas_model(), "app-model")

    def test_blank_explicit_override_is_rejected(self):
        with self.assertRaises(ValueError):
            get_gemini_model("  ")


if __name__ == "__main__":
    unittest.main()
