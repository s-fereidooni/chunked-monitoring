"""Versioned monitor prompts.

Prompts live as text files under `prompts/<version>/<name>.md` so they diff
cleanly in review. Each load carries a sha256 of the exact text, which gets
recorded alongside results — a prompt edited after a run can then be detected
rather than quietly invalidating the numbers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPTS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Prompt:
    """One versioned prompt, with a digest for provenance."""

    name: str
    version: str
    text: str
    sha256: str

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]

    def describe(self) -> dict[str, str]:
        """Provenance record to store with experiment outputs."""
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


def available_versions() -> list[str]:
    return sorted(
        p.name
        for p in PROMPTS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def available_prompts(version: str) -> list[str]:
    directory = PROMPTS_ROOT / version
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.md"))


@cache
def load_prompt(name: str, version: str = "v1") -> Prompt:
    """Load `prompts/<version>/<name>.md`.

    Cached, so repeated loads during a run cannot pick up a mid-run edit and
    produce results attributed to two different prompt texts.
    """
    path = PROMPTS_ROOT / version / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"No prompt {name!r} at version {version!r}. "
            f"Available: {available_prompts(version) or available_versions()}"
        )
    text = path.read_text().strip()
    digest = hashlib.sha256(text.encode()).hexdigest()
    return Prompt(name=name, version=version, text=text, sha256=digest)


__all__ = [
    "PROMPTS_ROOT",
    "Prompt",
    "available_prompts",
    "available_versions",
    "load_prompt",
]
