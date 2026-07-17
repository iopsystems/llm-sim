"""Shared test fixtures/helpers.

EngineCore tests need facebook/opt-125m's config in the local HF cache
(weights are dummy-loaded, never downloaded). They skip cleanly when absent.
"""

import functools
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "facebook/opt-125m"


@functools.lru_cache
def model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL, local_files_only=True, allow_patterns=["config.json"])
        return True
    except Exception:
        return False
