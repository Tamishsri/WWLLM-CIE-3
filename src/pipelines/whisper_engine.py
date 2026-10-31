"""
whisper_engine.py — Open ASR that runs on a CPU with no login
=============================================================
IndicConformer is the project's first-choice ASR, but its checkpoints are
gated on Hugging Face and need the NeMo toolkit.  This engine exposes the same
``transcribe(chunks)`` interface using openly licensed Whisper models,
fine-tuned per language where a fine-tuned model exists.

Usage::

    engine = WhisperEngine("Hindi")
    texts = engine.transcribe(chunks)   # one string per VAD chunk
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from loguru import logger
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

from src.pipelines.languages import GENERAL_WHISPER, WHISPER_MODELS

SAMPLE_RATE: int = 16_000


class WhisperEngine:
    """Whisper inference over VAD chunks.

    Parameters
    ----------
    language : str
        Language name, e.g. ``"Hindi"``.
    model_id : str | None
        Overrides the per-language default.
    device : str | None
        ``"cuda"`` or ``"cpu"``.  Auto-detects if *None*.
    batch_size : int
        Chunks per ``generate()`` call.
    """

    def __init__(
        self,
        language: str = "Hindi",
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
    ) -> None:
        self.language = language.lower()
        self.model_id = model_id or WHISPER_MODELS.get(language, GENERAL_WHISPER)
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        logger.info(f"Loading Whisper: {self.model_id} …")
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        # Fine-tuned checkpoints predate language= in generate(); they share
        # Whisper's vocabulary, so the stock generation config applies to them.
        self.model.generation_config = GenerationConfig.from_pretrained(GENERAL_WHISPER)
        logger.success(f"Whisper ready on {self.device} — {self.model_id}")

    def transcribe(self, chunks: List[torch.Tensor]) -> List[str]:
        """Transcribe 16 kHz mono waveforms (each ≤ 30 s), one string per chunk."""
        if not chunks:
            return []
        audio = [chunk.detach().cpu().numpy() for chunk in chunks]
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = inputs.input_features.to(self.device)
        with torch.no_grad():
            generated = self.model.generate(features, language=self.language, task="transcribe")
        return [text.strip() for text in self.processor.batch_decode(generated, skip_special_tokens=True)]

    @property
    def model_info(self) -> Dict[str, str]:
        """Return a summary dict for UI display."""
        return {"model_id": self.model_id, "device": str(self.device), "language": self.language}
