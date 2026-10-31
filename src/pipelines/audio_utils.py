"""
audio_utils.py — Data Ingestion & Voice Activity Detection Pipeline
=====================================================================
Handles audio extraction from .wav / .mp3 / .mp4 files, resamples to
16 kHz mono, and applies Silero-VAD to produce intelligent speech chunks
(15–30 s) while dropping extended silences.

Architecture Notes:
    • Uses ``soundfile`` / ``torchaudio`` for I/O and resampling; anything
      they can't read directly (video, .m4a) is decoded with ``ffmpeg``.
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
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from loguru import logger

# Silero-VAD helper functions.  Importing the package pins the whole process to
# one thread, which would slow every model that runs afterwards.
_threads = torch.get_num_threads()
from silero_vad import load_silero_vad, get_speech_timestamps  # noqa: E402

torch.set_num_threads(_threads)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int = 16_000
"""All audio is resampled to 16 kHz mono before processing."""

_SUPPORTED_AUDIO_EXTS: set[str] = {".wav", ".mp3", ".flac", ".ogg"}
_SUPPORTED_VIDEO_EXTS: set[str] = {".mp4", ".mkv", ".webm", ".avi", ".m4a"}


def read_audio(source: str | Path | io.BytesIO) -> tuple[torch.Tensor, int]:
    """Decode audio into ``(waveform [C, T], sample_rate)``.

    Recent ``torchaudio`` releases delegate decoding to the optional
    TorchCodec package and raise ``ImportError`` without it, so ``soundfile``
    is the fallback.
    """
    try:
        return torchaudio.load(source if isinstance(source, io.BytesIO) else str(source))
    except (ImportError, RuntimeError, OSError) as exc:
        logger.debug(f"torchaudio.load failed ({exc}); using soundfile.")

    import soundfile as sf

    if isinstance(source, io.BytesIO):
        source.seek(0)
    else:
        source = str(source)
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


def write_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a mono waveform as 16-bit PCM WAV (no codec dependency)."""
    pcm = (waveform.detach().cpu().flatten().clamp(-1.0, 1.0) * 32767).short()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.numpy().tobytes())


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------
class AudioProcessor:
    """End-to-end audio ingestion → VAD chunking pipeline.

    Parameters
    ----------
    max_chunk_sec : float
        Maximum duration (seconds) of a single speech chunk.  Default 25 s.
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
        vad_threshold: float = 0.45,
        min_silence_ms: int = 300,
    ) -> None:
        self.max_chunk_sec = max_chunk_sec
        self.vad_threshold = vad_threshold
        self.min_silence_ms = min_silence_ms

        logger.info("Loading Silero-VAD model …")
        self._vad_model = load_silero_vad()
        logger.success("Silero-VAD ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, file_path: str | Path) -> List[torch.Tensor]:
        """Ingest a media file and return a list of speech-chunk tensors.

        Each returned tensor is a 1-D float32 waveform at 16 kHz.  See
        :meth:`process_with_spans` to also get each chunk's time range.
        """
        chunks, _ = self.process_with_spans(file_path)
        return chunks

    def process_with_spans(
        self, file_path: str | Path
    ) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """Like :meth:`process`, but also return ``(start_s, end_s)`` per chunk.

        The spans let the UI paint VAD chunks on the waveform and sync the
        transcript to playback.

        Parameters
        ----------
        file_path : str | Path
            Path to a .wav, .mp3, .mp4, etc.

        Returns
        -------
        Tuple[List[torch.Tensor], List[Tuple[float, float]]]
            Ordered speech chunks ready for ASR inference, and their spans.

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

        chunks, spans = self._cut_segments(waveform, raw_stamps)
        logger.success(f"Produced {len(chunks)} speech chunks for ASR.")
        return chunks, spans

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
        return read_audio(path)

    @staticmethod
    def _extract_audio_from_video(path: Path) -> tuple[torch.Tensor, int]:
        """Extract the audio track from a video container with ffmpeg.

        Returns
        -------
        tuple[torch.Tensor, int]
            (waveform, sample_rate)
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(f"ffmpeg is needed to read {path.suffix} files — install it and add it to PATH.")

        logger.info(f"Extracting audio with ffmpeg: {path.name}")
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path), "-ac", "1",
             "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
            capture_output=True,
            check=True,
        )
        return read_audio(io.BytesIO(result.stdout))

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
        # Silero recommends one thread, but the setting is process-wide, so it
        # must not leak into the ASR and LLM models that run afterwards.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return get_speech_timestamps(
                waveform,
                self._vad_model,
                sampling_rate=TARGET_SAMPLE_RATE,
                threshold=self.vad_threshold,
                min_speech_duration_ms=250,
                max_speech_duration_s=float(self.max_chunk_sec),
                min_silence_duration_ms=self.min_silence_ms,
                return_seconds=False,  # sample-level precision
            )
        finally:
            torch.set_num_threads(threads)

    @staticmethod
    def _cut_segments(
        waveform: torch.Tensor,
        stamps: list[dict[str, int]],
    ) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """Turn VAD timestamps into chunks, one per stretch of speech.

        Stretches are deliberately not merged.  Whisper's fine-tuned models
        learned on single utterances and stop at the first long pause, so a
        merged chunk would lose everything after its first sentence.

        Parameters
        ----------
        waveform : torch.Tensor
            Full 1-D waveform.
        stamps : list[dict[str, int]]
            VAD timestamps (sample-level).

        Returns
        -------
        Tuple[List[torch.Tensor], List[Tuple[float, float]]]
            Speech chunks and their ``(start_s, end_s)`` spans.
        """
        sr = TARGET_SAMPLE_RATE
        if not stamps:
            logger.warning("No speech detected by VAD — returning full waveform.")
            return [waveform], [(0.0, waveform.shape[-1] / sr)]

        chunks = [waveform[seg["start"] : seg["end"]] for seg in stamps]
        spans = [(round(seg["start"] / sr, 3), round(seg["end"] / sr, 3)) for seg in stamps]
        return chunks, spans
