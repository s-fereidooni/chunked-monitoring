"""Minimal `.env` loading.

Avoids a python-dotenv dependency for what is a dozen lines. Existing
environment variables always win, so an explicitly exported value is never
silently overridden by a stale `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

from monitor_localization.paths import ROOT


def load_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load `KEY=value` pairs from `.env` into `os.environ`.

    Returns the variables that were actually applied. Missing file is not an
    error — credentials may come from the ambient environment or `ant auth login`.
    """
    path = Path(path) if path is not None else ROOT / ".env"
    if not path.exists():
        return {}

    applied: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
