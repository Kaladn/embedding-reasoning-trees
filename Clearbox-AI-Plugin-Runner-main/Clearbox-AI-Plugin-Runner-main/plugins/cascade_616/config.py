"""Cascade 6-1-6 configuration."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "store_path": "D:/Clearbox AI Data Root/cascade_616/evidence.bin",
    "lexicon_path": "",     # optional — symbols baked into evidence store
    "default_k": 5,
    "max_chain_steps": 12,
}


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load config. File values override defaults."""
    config = dict(DEFAULTS)

    if config_path is None:
        for candidate in [
            Path("clearbox.config.json"),
            Path("forest.config.json"),
        ]:
            if candidate.exists():
                config_path = candidate
                break

    if config_path and config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                full = json.load(f)
            section = full.get("cascade_616", {})
            config.update(section)
        except Exception as e:
            logger.warning("Failed to load cascade_616 config: %s", e)

    return config
