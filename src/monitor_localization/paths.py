"""Canonical project paths.

Everything resolves relative to the repository root so scripts and notebooks
behave identically regardless of the working directory they are launched from.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_SUBSET = DATA_DIR / "evaluation_subset.json"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
ANALYSIS_DIR = ROOT / "analysis"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if absent, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
