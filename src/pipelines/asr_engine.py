"""
asr_engine.py — AI4Bharat IndicConformer ASR Inference Engine
=============================================================
Production-grade wrapper around the **IndicConformer** (hybrid CTC-RNNT)
automatic speech recognition model released by AI4Bharat.

Architecture Overview
---------------------
IndicConformer is a 600 M-parameter Conformer-Large encoder with a
hybrid CTC/RNNT decoder head, trained on 40 000+ hours of Indian-language
speech data.  It ships as a ``.nemo`` checkpoint and requires NVIDIA
NeMo's ASR collection for loading and inference.

This module provides:
    1. **Lazy singleton loading** — the model is downloaded / restored
       once and cached across Streamlit re-runs via ``st.session_state``.
    2. **Batch inference with automatic chunking** — audio chunks from
       the VAD pipeline are grouped into micro-batches to maximise GPU
       utilisation without exceeding VRAM.
    3. **Graceful CPU fallback** — if a CUDA OOM error is raised mid-
       inference, the failing batch is re-tried on CPU and a warning is
       logged.
    4. **Robust exception handling** — tensor-size mismatches, empty
       audio, and codec errors are caught and logged, never silently
       swallowed.

Usage::

    engine = ASREngine(model_id="ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
    transcripts = engine.transcribe(chunks)   # list[str]

Environment Prerequisites
-------------------------
* Install AI4Bharat's NeMo fork::

      git clone --branch nemo-v2 https://github.com/AI4Bharat/NeMo.git
      cd NeMo && bash reinstall.sh

* CUDA 11.8+ with PyTorch 2.1+ compiled for the matching CUDA toolkit.
* ≥ 8 GB VRAM recommended for the Large variant.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Union

import torch
from loguru import logger

from src.pipelines.audio_utils import write_wav

# ---------------------------------------------------------------------------
# NeMo import — guarded so the rest of the codebase remains importable even
# if NeMo is not yet installed (useful for linting / CI).
# ---------------------------------------------------------------------------
try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import ASRModel

    _NEMO_AVAILABLE = True
except ImportError:
    _NEMO_AVAILABLE = False
    ASRModel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID: str = "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"
"""
Default HuggingFace / NeMo model identifier.

Available variants (swap ``hi`` for target language ISO code):
    • ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large   — Hindi
    • ai4bharat/indicconformer_stt_en_hybrid_rnnt_large   — English
    • ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large   — Tamil
    • ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large   — Kannada
    • ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large   — Malayalam
    • ai4bharat/indicconformer_stt_te_hybrid_rnnt_large   — Telugu

You may also pass a local ``.nemo`` checkpoint path.
"""

SAMPLE_RATE: int = 16_000
"""IndicConformer requires 16 kHz mono WAV input."""


# ---------------------------------------------------------------------------
# ASREngine
# ---------------------------------------------------------------------------
class ASREngine:
    """High-level IndicConformer inference engine.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier *or* local ``.nemo`` file path.
    device : str | None
        ``"cuda"`` or ``"cpu"``.  If *None*, auto-detects CUDA.
    batch_size : int
        Number of audio chunks to feed per ``model.transcribe()`` call.
        Reduce if you encounter OOM on smaller GPUs.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        if not _NEMO_AVAILABLE:
            raise ImportError(
                "NVIDIA NeMo ASR is required for IndicConformer.  "
                "Install AI4Bharat's fork: "
                "git clone --branch nemo-v2 https://github.com/AI4Bharat/NeMo.git && "
                "cd NeMo && bash reinstall.sh"
            )

        self.model_id = model_id
        self.batch_size = batch_size

        # ── Device selection ────────────────────────────────────────────
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        logger.info(f"ASREngine device → {self.device}")

        # ── Model loading ───────────────────────────────────────────────
        self._model: ASRModel = self._load_model(model_id)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self, model_id: str) -> ASRModel:
        """Load the IndicConformer model from a .nemo file or HF Hub.

        Strategy:
            1. If *model_id* points to an existing ``.nemo`` file on disk,
               use ``ASRModel.restore_from()``.
            2. Otherwise, treat it as a HuggingFace model identifier and
               call ``ASRModel.from_pretrained()``.
            3. Freeze all parameters (inference-only) and move to the
               selected device.

        Returns
        -------
        ASRModel
            Ready-to-infer NeMo ASR model.

        Raises
        ------
        RuntimeError
            If neither loading strategy succeeds.
        """
        logger.info(f"Loading IndicConformer: {model_id} …")

        try:
            if Path(model_id).is_file() and model_id.endswith(".nemo"):
                # ── Local checkpoint ────────────────────────────────────
                model = ASRModel.restore_from(restore_path=model_id)
                logger.info("Loaded from local .nemo checkpoint.")
            else:
                # ── HuggingFace Hub ─────────────────────────────────────
                model = ASRModel.from_pretrained(model_name=model_id)
                logger.info(f"Loaded from HuggingFace: {model_id}")
        except Exception as exc:
            logger.error(f"Failed to load model '{model_id}': {exc}")
            raise RuntimeError(
                f"Could not load IndicConformer model '{model_id}'.  "
                "Ensure the AI4Bharat NeMo fork is installed and the "
                "model identifier is correct."
            ) from exc

        # Freeze for inference — no gradient computation needed
        model.freeze()
        model = model.to(self.device)
        model.eval()
        logger.success(
            f"IndicConformer ready on {self.device} "
            f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params)"
        )
        return model

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_chunks: List[torch.Tensor],
    ) -> List[str]:
        """Transcribe a list of 16 kHz mono waveform tensors.

        The NeMo ``transcribe()`` API expects a list of **file paths**,
        so each tensor is written to a temporary WAV file.  This is the
        recommended upstream pattern and avoids internal resampling
        mismatches.

        Parameters
        ----------
        audio_chunks : List[torch.Tensor]
            1-D float32 tensors at 16 kHz (output of ``AudioProcessor``).

        Returns
        -------
        List[str]
            One transcript string per input chunk, in order.

        Notes
        -----
        * Empty or near-silent chunks produce ``""``.
        * If a CUDA OOM occurs, the failing micro-batch is retried on
          CPU and a warning is logged.
        """
        if not audio_chunks:
            logger.warning("transcribe() called with empty chunk list.")
            return []

        # ── Write chunks to temporary WAV files ────────────────────────
        tmp_paths: List[str] = []
        valid_indices: List[int] = []

        for idx, chunk in enumerate(audio_chunks):
            if chunk.numel() < SAMPLE_RATE * 0.1:  # < 100 ms
                logger.debug(f"Chunk {idx} too short ({chunk.numel()} samples), skipping.")
                continue
            try:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix=f"asr_chunk_{idx}_"
                )
                tmp.close()  # Windows can't reopen a file that is still open
                write_wav(tmp.name, chunk, SAMPLE_RATE)
                tmp_paths.append(tmp.name)
                valid_indices.append(idx)
            except Exception as exc:
                logger.warning(f"Could not write chunk {idx} to WAV: {exc}")

        if not tmp_paths:
            logger.warning("No valid chunks to transcribe after filtering.")
            return [""] * len(audio_chunks)

        logger.info(
            f"Transcribing {len(tmp_paths)} chunks "
            f"(batch_size={self.batch_size}) …"
        )

        # ── Batch inference ────────────────────────────────────────────
        all_transcripts: List[str] = []
        for batch_start in range(0, len(tmp_paths), self.batch_size):
            batch_paths = tmp_paths[batch_start : batch_start + self.batch_size]
            batch_texts = self._transcribe_batch(batch_paths)
            all_transcripts.extend(batch_texts)

        # ── Clean up temp files ────────────────────────────────────────
        for p in tmp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

        # ── Re-map into the original chunk ordering ────────────────────
        result: List[str] = [""] * len(audio_chunks)
        for pos, orig_idx in enumerate(valid_indices):
            if pos < len(all_transcripts):
                result[orig_idx] = all_transcripts[pos]

        return result

    def _transcribe_batch(self, file_paths: List[str]) -> List[str]:
        """Run model.transcribe() on a micro-batch of WAV file paths.

        Implements a two-tier strategy:
            1. **GPU inference** — fast path.
            2. **CPU fallback** — triggered automatically on CUDA OOM.

        Parameters
        ----------
        file_paths : List[str]
            Paths to 16 kHz mono WAV files.

        Returns
        -------
        List[str]
            Transcripts for the batch.
        """
        # ── Attempt 1: GPU ─────────────────────────────────────────────
        if self.device.type == "cuda":
            try:
                with torch.cuda.amp.autocast():
                    outputs = self._model.transcribe(
                        file_paths, batch_size=len(file_paths)
                    )
                return self._extract_texts(outputs)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "CUDA OOM on batch — clearing cache and retrying on CPU."
                )
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "CUDA" in str(exc) or "out of memory" in str(exc).lower():
                    logger.warning(f"CUDA runtime error: {exc}. Falling back to CPU.")
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"Unexpected RuntimeError during GPU transcription:\n{traceback.format_exc()}")
                    return ["[TRANSCRIPTION ERROR]"] * len(file_paths)

        # ── Attempt 2: CPU fallback ────────────────────────────────────
        try:
            logger.info("Running ASR inference on CPU (slower) …")
            self._model = self._model.to("cpu")
            outputs = self._model.transcribe(
                file_paths, batch_size=len(file_paths)
            )
            # Move model back to GPU for subsequent batches
            if torch.cuda.is_available():
                self._model = self._model.to(self.device)
            return self._extract_texts(outputs)
        except Exception as exc:
            logger.error(
                f"CPU transcription also failed:\n{traceback.format_exc()}"
            )
            return ["[TRANSCRIPTION ERROR]"] * len(file_paths)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_texts(outputs) -> List[str]:
        """Normalise NeMo transcribe() output into a flat list of strings.

        NeMo's ``transcribe()`` may return:
            • ``list[str]`` — simple CTC / greedy decoding
            • ``list[Hypothesis]`` — beam search (each has a ``.text`` attr)
            • A nested structure like ``(hypotheses_list, None)``

        This helper handles all known variants.

        Parameters
        ----------
        outputs
            Raw return value from ``model.transcribe()``.

        Returns
        -------
        List[str]
        """
        # Case 1: tuple — NeMo sometimes wraps results in (hyps, None)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        texts: List[str] = []
        for item in outputs:
            if isinstance(item, str):
                texts.append(item.strip())
            elif hasattr(item, "text"):
                # nemo Hypothesis object
                texts.append(item.text.strip())
            elif isinstance(item, list) and len(item) > 0:
                # Nested list of hypotheses — take best (index 0)
                best = item[0]
                texts.append(
                    best.text.strip() if hasattr(best, "text") else str(best).strip()
                )
            else:
                texts.append(str(item).strip())

        return texts

    def transcribe_file(self, file_path: str | Path) -> str:
        """Convenience: transcribe a single audio file directly.

        Parameters
        ----------
        file_path : str | Path
            Path to a 16 kHz mono WAV.

        Returns
        -------
        str
            Full transcript.
        """
        path_str = str(file_path)
        try:
            outputs = self._model.transcribe([path_str], batch_size=1)
            texts = self._extract_texts(outputs)
            return texts[0] if texts else ""
        except Exception as exc:
            logger.error(f"transcribe_file failed for {path_str}: {exc}")
            return "[TRANSCRIPTION ERROR]"

    @property
    def model_info(self) -> dict[str, str]:
        """Return a summary dict for UI display."""
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "parameters": f"{sum(p.numel() for p in self._model.parameters()) / 1e6:.1f}M",
            "batch_size": str(self.batch_size),
        }
