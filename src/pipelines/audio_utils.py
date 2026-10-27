"""
audio_utils.py — Data Ingestion & Voice Activity Detection Pipeline
=====================================================================
Handles audio extraction from .wav / .mp3 / .mp4 files, resamples to
16 kHz mono, and applies Silero-VAD to produce intelligent speech chunks
(15–30 s) while dropping extended silences.

Architecture Notes:
    • Uses ``torchaudio`` for I/O and resampling — no subprocess calls to
      ffmpeg unless the input is .mp4 (video), in which case ``pydub`` is
      used for demuxing.
    • Silero-VAD runs on CPU (single-threaded, as recommended by upstream)
      regardless of the global device setting.
    • Adjacent VAD speech segments are merged up to ``max_chunk_sec`` to
      keep downstream ASR batches GPU-friendly and prevent OOM.

Typical usage::

    processor = AudioProcessor(max_chunk_sec=25.0)
    chunks = processor.process("lecture.mp4")
    # chunks: list[torch.Tensor], each 16 kHz mono float32
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from loguru import logger

# Silero-VAD helper functions
from silero_vad import load_silero_vad, get_speech_timestamps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int = 16_000
"""All audio is resampled to 16 kHz mono before processing."""

_SUPPORTED_AUDIO_EXTS: set[str] = {".wav", ".mp3", ".flac", ".ogg"}
_SUPPORTED_VIDEO_EXTS: set[str] = {".mp4", ".mkv", ".webm", ".avi"}


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------
class AudioProcessor:
    """End-to-end audio ingestion → VAD chunking pipeline.

    Parameters
    ----------
    max_chunk_sec : float
        Maximum duration (seconds) of a single speech chunk.  Adjacent
        VAD segments are merged until this ceiling is hit.  Default 25 s.
    min_chunk_sec : float
        Minimum duration (seconds).  Chunks shorter than this are merged
        with their neighbour if possible.  Default 2 s.
    vad_threshold : float
        Silero-VAD speech probability threshold (0–1).  Higher values
        are stricter.  Default 0.45.
    min_silence_ms : int
        Minimum silence gap (ms) before the VAD considers a segment
        boundary.  Default 300 ms.
    """

    def __init__(
        self,
        max_chunk_sec: float = 25.0,
        min_chunk_sec: float = 2.0,
        vad_threshold: float = 0.45,
        min_silence_ms: int = 300,
    ) -> None:
        self.max_chunk_sec = max_chunk_sec
        self.min_chunk_sec = min_chunk_sec
        self.vad_threshold = vad_threshold
        self.min_silence_ms = min_silence_ms

        # Silero recommends single-threaded CPU execution
        torch.set_num_threads(1)
        logger.info("Loading Silero-VAD model …")
        self._vad_model = load_silero_vad()
        logger.success("Silero-VAD ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, file_path: str | Path) -> List[torch.Tensor]:
        """Ingest a media file and return a list of speech-chunk tensors.

        Each returned tensor is a 1-D float32 waveform at 16 kHz.

        Parameters
        ----------
        file_path : str | Path
            Path to a .wav, .mp3, .mp4, etc.

        Returns
        -------
        List[torch.Tensor]
            Ordered speech chunks ready for ASR inference.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the file extension is unsupported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio/video file not found: {path}")

        ext = path.suffix.lower()
        if ext in _SUPPORTED_VIDEO_EXTS:
            waveform = self._extract_audio_from_video(path)
        elif ext in _SUPPORTED_AUDIO_EXTS:
            waveform = self._load_audio(path)
        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Accepted: {_SUPPORTED_AUDIO_EXTS | _SUPPORTED_VIDEO_EXTS}"
            )

        # Resample to 16 kHz mono
        waveform = self._normalise(waveform, TARGET_SAMPLE_RATE)
        logger.info(
            f"Waveform ready — {waveform.shape[-1] / TARGET_SAMPLE_RATE:.1f} s "
            f"@ {TARGET_SAMPLE_RATE} Hz"
        )

        # Run VAD → raw speech timestamps
        raw_stamps = self._run_vad(waveform)
        logger.info(f"VAD detected {len(raw_stamps)} raw speech segments.")

        # Merge into 15-30 s chunks
        chunks = self._merge_segments(waveform, raw_stamps)
        logger.success(f"Produced {len(chunks)} speech chunks for ASR.")
        return chunks

    def process_uploaded_bytes(
        self, data: bytes, filename: str
    ) -> List[torch.Tensor]:
        """Process raw bytes from a Streamlit ``UploadedFile``.

        Writes to a temporary file, delegates to :meth:`process`, then
        cleans up.

        Parameters
        ----------
        data : bytes
            Raw file content.
        filename : str
            Original filename (used to infer extension).

        Returns
        -------
        List[torch.Tensor]
        """
        ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return self.process(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
        """Load an audio file via torchaudio.

        Returns
        -------
        tuple[torch.Tensor, int]
            (waveform, sample_rate)
        """
        waveform, sr = torchaudio.load(str(path))
        return waveform, sr

    @staticmethod
    def _extract_audio_from_video(path: Path) -> tuple[torch.Tensor, int]:
        """Extract audio track from a video container.

        Falls back to ``pydub`` (which shells out to ffmpeg) because
        torchaudio's video backend is not universally available.

        Returns
        -------
        tuple[torch.Tensor, int]
            (waveform, sample_rate)
        """
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise ImportError(
                "pydub is required for video → audio extraction.  "
                "Install it: pip install pydub"
            ) from exc

        logger.info(f"Extracting audio from video: {path.name}")
        audio = AudioSegment.from_file(str(path))

        # Export to in-memory WAV for torchaudio loading
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)

        waveform, sr = torchaudio.load(buf)
        return waveform, sr

    @staticmethod
    def _normalise(
        wav_and_sr: tuple[torch.Tensor, int],
        target_sr: int,
    ) -> torch.Tensor:
        """Down-mix to mono and resample to *target_sr*.

        Parameters
        ----------
        wav_and_sr : tuple[torch.Tensor, int]
            (waveform [C, T], original sample rate)
        target_sr : int
            Desired sample rate.

        Returns
        -------
        torch.Tensor
            1-D float32 tensor.
        """
        waveform, sr = wav_and_sr

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # [T]

    def _run_vad(
        self, waveform: torch.Tensor
    ) -> list[dict[str, int]]:
        """Run Silero-VAD and return speech timestamps in *samples*.

        Parameters
        ----------
        waveform : torch.Tensor
            1-D float32 tensor at ``TARGET_SAMPLE_RATE``.

        Returns
        -------
        list[dict[str, int]]
            Each dict has ``{'start': int, 'end': int}`` in samples.
        """
        stamps = get_speech_timestamps(
            waveform,
            self._vad_model,
            sampling_rate=TARGET_SAMPLE_RATE,
            threshold=self.vad_threshold,
            min_speech_duration_ms=250,
            max_speech_duration_s=float(self.max_chunk_sec),
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=False,  # sample-level precision
        )
        return stamps

    def _merge_segments(
        self,
        waveform: torch.Tensor,
        stamps: list[dict[str, int]],
    ) -> List[torch.Tensor]:
        """Merge adjacent VAD segments into chunks of ≤ *max_chunk_sec*.

        Strategy:
            1. Walk through sorted timestamps.
            2. Accumulate contiguous segments until adding the next one
               would exceed ``max_chunk_sec``.
            3. Flush the accumulated range as a single chunk.
            4. Drop any final chunk shorter than ``min_chunk_sec``
               (unless it's the only chunk).

        Parameters
        ----------
        waveform : torch.Tensor
            Full 1-D waveform.
        stamps : list[dict[str, int]]
            VAD timestamps (sample-level).

        Returns
        -------
        List[torch.Tensor]
            Merged speech chunks.
        """
        if not stamps:
            logger.warning("No speech detected by VAD — returning full waveform.")
            return [waveform]

        max_samples = int(self.max_chunk_sec * TARGET_SAMPLE_RATE)
        min_samples = int(self.min_chunk_sec * TARGET_SAMPLE_RATE)

        chunks: List[torch.Tensor] = []
        run_start: int = stamps[0]["start"]
        run_end: int = stamps[0]["end"]

        for seg in stamps[1:]:
            proposed_end = seg["end"]
            if (proposed_end - run_start) <= max_samples:
                # Extend the current run
                run_end = proposed_end
            else:
                # Flush current run
                chunks.append(waveform[run_start:run_end])
                run_start = seg["start"]
                run_end = seg["end"]

        # Flush final run
        final_chunk = waveform[run_start:run_end]
        if len(final_chunk) >= min_samples or not chunks:
            chunks.append(final_chunk)
        else:
            # Merge tiny trailing fragment into the last chunk
            chunks[-1] = torch.cat([chunks[-1], final_chunk])

        return chunks
