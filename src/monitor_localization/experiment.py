"""Experiment run directories.

Every run records its configuration, timestamps, and usage next to its outputs so
a results file is never orphaned from the settings that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monitor_localization.config import ExperimentConfig
from monitor_localization.paths import RESULTS_DIR, ensure_dir
from monitor_localization.utils.io import write_json


@dataclass
class ExperimentRun:
    """One stage's output directory and its manifest."""

    config: ExperimentConfig
    stage: str
    root: Path = field(init=False)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base = Path(self.config.output_dir)
        if not base.is_absolute():
            base = RESULTS_DIR.parent / base
        self.root = ensure_dir(base / self.config.name / self.stage)

    @property
    def results_path(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "run.json"

    def write_manifest(self, **fields: Any) -> Path:
        """Persist configuration plus whatever the stage wants to record."""
        manifest = {
            "stage": self.stage,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "config": self.config.to_dict(),
            **self.extra,
            **fields,
        }
        return write_json(self.manifest_path, manifest)
