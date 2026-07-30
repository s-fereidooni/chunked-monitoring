"""Content-addressed on-disk cache for LLM responses.

Inference is the dominant cost of this project, and the chunk-wise stage
multiplies call count by the number of chunks. Re-running a stage after an
unrelated code change should cost nothing.

The key is a digest of everything that can change the response: model, sampling
parameters, the schema, and the exact prompt text. Anything that would produce a
different answer produces a different key, so a stale hit is not possible by
construction. Entries are sharded two levels deep by digest prefix to keep
directory sizes reasonable at tens of thousands of entries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from monitor_localization.paths import CACHE_DIR
from monitor_localization.utils.io import read_json, write_json
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)


def cache_key(payload: dict[str, Any]) -> str:
    """Stable digest of a request payload.

    `sort_keys=True` matters: dict ordering must not affect the key, or the cache
    silently misses whenever an unrelated refactor reorders a literal.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    """Read-through cache mapping request digests to stored responses."""

    def __init__(self, root: str | Path | None = None, *, enabled: bool = True) -> None:
        self.root = Path(root) if root is not None else CACHE_DIR / "responses"
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            entry = read_json(path)
        except (OSError, json.JSONDecodeError):
            # A truncated entry from an interrupted write is a miss, not a crash.
            logger.warning("Discarding unreadable cache entry %s", path)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        write_json(self.path_for(key), value)
        self.writes += 1

    @property
    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
