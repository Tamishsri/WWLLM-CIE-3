"""
languages.py — Language tables shared by the pipelines and the UI
=================================================================
Dependency-free so the API can describe languages without importing
PyTorch.  ``translator.py`` re-exports ``SUPPORTED_TARGETS``.
"""

from __future__ import annotations

from typing import Dict

ASR_LANGUAGES: Dict[str, str] = {
    "Hindi": "hi",
    "English": "en",
    "Tamil": "ta",
    "Kannada": "kn",
    "Telugu": "te",
    "Malayalam": "ml",
}
"""Source languages with an IndicConformer checkpoint (name → ISO 639-1)."""

WHISPER_MODELS: Dict[str, str] = {
    "Hindi": "vasista22/whisper-hindi-small",
    "Tamil": "vasista22/whisper-tamil-small",
    "Kannada": "vasista22/whisper-kannada-small",
    "Telugu": "vasista22/whisper-telugu-small",
}
"""Open checkpoints fine-tuned per language.  Others use ``GENERAL_WHISPER``."""

GENERAL_WHISPER: str = "openai/whisper-small"

SUPPORTED_TARGETS: Dict[str, str] = {
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Odia": "ory_Orya",
}
"""IndicTrans2 target languages and their FLORES-200 codes (the model covers 22)."""

NATIVE_NAMES: Dict[str, str] = {
    "Hindi": "हिन्दी",
    "English": "English",
    "Tamil": "தமிழ்",
    "Kannada": "ಕನ್ನಡ",
    "Telugu": "తెలుగు",
    "Malayalam": "മലയാളം",
    "Bengali": "বাংলা",
    "Marathi": "मराठी",
    "Gujarati": "ગુજરાતી",
    "Punjabi": "ਪੰਜਾਬੀ",
    "Odia": "ଓଡ଼ିଆ",
}
"""Endonyms for display."""
