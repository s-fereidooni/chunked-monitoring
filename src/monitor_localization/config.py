"""Typed experiment configuration, loaded from YAML.

Every knob the notes call out as configurable lives here: model, temperature,
chunk size, overlap, aggregation method, output directory. Nothing that varies
between experiments should be hardcoded elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from monitor_localization.paths import CONFIGS_DIR, RESULTS_DIR

Effort = Literal["low", "medium", "high", "xhigh", "max"]
ThinkingMode = Literal["adaptive", "disabled"]
AggregationMethod = Literal["max", "mean", "noisy_or", "topk_mean"]
ChunkUnit = Literal["messages", "characters"]


class StrictModel(BaseModel):
    """Reject unknown keys so a typo in YAML fails loudly instead of silently."""

    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    """Which MALT variant to load.

    `hf_config` selects between the base transcripts (`default`) and MALT's
    chain-of-thought obfuscation variants. The variants are deliberately out of
    scope for the first milestone but cost nothing to leave addressable.
    """

    hf_repo: str = "metr-evals/malt-public"
    hf_config: Literal[
        "default", "irrelevant_detail", "language_mixing", "summarize", "vague_cot"
    ] = "default"
    split: str = "public"
    cache_dir: Path | None = None


class SubsetConfig(StrictModel):
    """How to draw the reproducible evaluation subset.

    MALT's public manually-reviewed pool is ~1.8k runs, but ~92% of its
    behavior-positive runs are `prompted` — a uniform draw would make the
    evaluation mostly a test of prompted claude-3-7-sonnet behavior. Stratifying
    keeps the composition explicit and controllable.
    """

    n_samples: int = 500
    seed: int = 0
    manually_reviewed_only: bool = True
    stratify_by: list[str] = Field(default_factory=lambda: ["is_positive", "run_source"])
    positive_fraction: float | None = 0.5
    # Ceiling on the share of the positive stratum drawn from `prompted` runs.
    # Without it, ~92% of reviewed positives are prompted and natural instances
    # (which METR flags as the more realistic signal) get drowned out.
    max_prompted_fraction: float | None = 0.5
    max_characters: int | None = None

    @model_validator(mode="after")
    def _check_fractions(self) -> SubsetConfig:
        for field in ("positive_fraction", "max_prompted_fraction"):
            value = getattr(self, field)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {value}")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        return self


class ModelConfig(StrictModel):
    """Monitor LLM settings.

    `temperature` defaults to None and is omitted from the request entirely when
    unset. This is deliberate: claude-opus-5 and the other current models reject
    `temperature` with a 400, so a non-None value is only valid on older models.
    """

    name: str = "claude-opus-5"
    max_tokens: int = 4096
    effort: Effort = "high"
    thinking: ThinkingMode = "adaptive"
    temperature: float | None = None
    max_concurrency: int = 8
    max_retries: int = 4

    @model_validator(mode="after")
    def _warn_on_temperature(self) -> ModelConfig:
        if self.temperature is not None and self.thinking == "adaptive":
            raise ValueError(
                "temperature cannot be combined with adaptive thinking. Current Claude "
                "models reject `temperature` outright; set thinking='disabled' and use an "
                "older model, or leave temperature unset."
            )
        return self


class ChunkConfig(StrictModel):
    """Transcript chunking. The first milestone uses fixed size, no overlap."""

    size: int = 10
    overlap: int = 0
    unit: ChunkUnit = "messages"

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkConfig:
        if self.overlap >= self.size:
            raise ValueError(f"overlap ({self.overlap}) must be < size ({self.size})")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        return self


class AggregationConfig(StrictModel):
    """How per-chunk scores collapse into one transcript-level score."""

    method: AggregationMethod = "max"
    top_k: int = 3


class ExperimentConfig(StrictModel):
    """One fully-specified experiment."""

    name: str
    description: str = ""
    prompt_version: str = "v1"
    output_dir: Path = RESULTS_DIR
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    subset: SubsetConfig = Field(default_factory=SubsetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config by path, or by bare name from `configs/`."""
        path = Path(path)
        if not path.exists() and not path.is_absolute():
            candidate = CONFIGS_DIR / path
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".yaml")
            path = candidate
        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-safe dict, for recording alongside experiment outputs."""
        return self.model_dump(mode="json")
