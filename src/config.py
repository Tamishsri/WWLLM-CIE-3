"""
config.py — Shared configuration loader
=======================================
Loads ``configs/config.yaml`` and applies environment overrides of the
form ``WWLLM_<SECTION>_<KEY>`` (e.g. ``WWLLM_ASR_BATCH_SIZE=4``).  Override
values are parsed as YAML, so ``true``, ``4`` and ``0.3`` keep their types.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT: Path = Path(__file__).resolve().parent.parent
"""Project root (the directory containing ``app.py`` and ``server.py``)."""

CONFIG_PATH: Path = ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Return the YAML config with ``WWLLM_*`` environment overrides applied."""
    cfg: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            override = os.environ.get(f"WWLLM_{section}_{key}".upper())
            if override is not None:
                values[key] = yaml.safe_load(override)
    return cfg
