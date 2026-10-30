"""Smoke test for the classic Streamlit UI (it must load without the ML stack)."""

from __future__ import annotations

import unittest

from src.config import ROOT

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # Streamlit is optional for the web studio
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class ClassicApp(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)

    def test_loads_without_exceptions(self):
        self.app.run()
        self.assertFalse(self.app.exception, self.app.exception)
        self.assertEqual(len(self.app.tabs), 5)

    def test_all_translation_targets_are_offered(self):
        self.app.run()
        target = self.app.sidebar.selectbox[0]
        self.assertIn("Telugu", target.options)
        self.assertIn("Bengali", target.options)

    def test_changing_language_clears_a_stale_translation(self):
        self.app.run()
        self.app.session_state["translated_notes"] = "हिन्दी"
        self.app.session_state["translated_lang"] = "Hindi"
        self.app.sidebar.selectbox[0].set_value("Tamil").run()
        self.assertFalse(self.app.exception, self.app.exception)
        self.assertIsNone(self.app.session_state["translated_notes"])


if __name__ == "__main__":
    unittest.main()
