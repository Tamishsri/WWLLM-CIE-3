"""
capabilities.py — Which pipeline stages can run for real on this machine
========================================================================
Detection uses ``importlib.util.find_spec`` so nothing heavy is imported
at startup.  GPU details need ``import torch`` (slow), so they are probed
once in a background thread.

Set ``WWLLM_DEMO=1`` (or ``python server.py --demo``) to force the offline
engines even when the full stack is installed.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

from loguru import logger

STAGE_REQUIREMENTS: Dict[str, List[str]] = {
    "vad": ["torch", "torchaudio", "silero_vad"],
    "asr": ["torch", "torchaudio", "silero_vad", "nemo"],
    "llm": ["torch", "transformers", "accelerate"],
    "mt": ["torch", "transformers", "IndicTransToolkit"],
}
"""Python modules each stage needs to run its real model."""

_OPTIONAL = ["bitsandbytes", "pydub", "jiwer", "bert_score", "rouge_score"]


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Capabilities:
    """Snapshot of installed modules plus lazily probed device info."""

    modules: Dict[str, bool]
    forced_demo: bool = False
    device: Dict[str, Any] = field(default_factory=lambda: {"probed": False})

    def live(self, stage: str) -> bool:
        """True if *stage* can run its real model."""
        if self.forced_demo:
            return False
        return all(self.modules.get(m, False) for m in STAGE_REQUIREMENTS[stage])

    def missing(self, stage: str) -> List[str]:
        return [m for m in STAGE_REQUIREMENTS[stage] if not self.modules.get(m, False)]

    def has(self, module: str) -> bool:
        return self.modules.get(module, False)

    @property
    def mode(self) -> str:
        """``live`` (all stages real), ``demo`` (none) or ``hybrid``."""
        states = [self.live(s) for s in STAGE_REQUIREMENTS]
        if all(states):
            return "live"
        return "hybrid" if any(states) else "demo"

    def probe_device_async(self) -> None:
        """Import torch in the background to report CUDA / GPU details."""
        if not self.modules.get("torch"):
            self.device = {"probed": True, "torch": None, "cuda": False, "gpu": None}
            return

        def _probe() -> None:
            try:
                import torch

                cuda = torch.cuda.is_available()
                self.device = {
                    "probed": True,
                    "torch": torch.__version__,
                    "cuda": cuda,
                    "cuda_version": torch.version.cuda if cuda else None,
                    "gpu": torch.cuda.get_device_name(0) if cuda else None,
                    "vram_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                    )
                    if cuda
                    else None,
                }
            except Exception as exc:  # broken CUDA installs shouldn't kill the app
                logger.warning(f"Device probe failed: {exc}")
                self.device = {"probed": True, "torch": None, "cuda": False, "gpu": None}

        threading.Thread(target=_probe, name="device-probe", daemon=True).start()


def detect_capabilities() -> Capabilities:
    """Inspect the environment once at startup."""
    names = sorted({m for reqs in STAGE_REQUIREMENTS.values() for m in reqs} | set(_OPTIONAL))
    forced = os.environ.get("WWLLM_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}
    caps = Capabilities(modules={n: _has_module(n) for n in names}, forced_demo=forced)
    logger.info(
        "Stage engines: "
        + ", ".join(f"{s}={'live' if caps.live(s) else 'fallback'}" for s in STAGE_REQUIREMENTS)
        + (" (demo forced)" if forced else "")
    )
    return caps
