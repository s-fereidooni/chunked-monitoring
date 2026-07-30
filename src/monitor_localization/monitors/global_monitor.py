"""Stage 2: the global monitor baseline.

Sees the entire transcript in one call and returns a single judgment. This is the
architecture whose blind spots the chunk-wise monitor is meant to recover, so it
is deliberately the straightforward implementation — no chunking, no retrieval,
no summarisation.
"""

from __future__ import annotations

from typing import Any

from monitor_localization.config import ModelConfig
from monitor_localization.dataset.schema import Transcript
from monitor_localization.llm.client import MonitorClient, MonitorLLMError
from monitor_localization.monitors.base import Monitor, MonitorResult, Verdict
from monitor_localization.prompts import load_prompt
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)


def render_for_monitor(transcript: Transcript) -> str:
    """Format one transcript as the monitor's user message."""
    header = (
        f"Task: {transcript.task_id}\n"
        f"Messages: {transcript.n_messages}\n"
        "---\n"
    )
    return header + transcript.render()


class GlobalMonitor(Monitor):
    """Whole-transcript monitor."""

    name = "global"

    def __init__(
        self,
        client: MonitorClient | None = None,
        model: ModelConfig | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self.model = model or ModelConfig()
        self.client = client or MonitorClient(self.model)
        self.prompt = load_prompt("global_monitor", prompt_version)

    def judge(self, transcript: Transcript) -> MonitorResult:
        """Score a transcript, recording failures rather than raising.

        A failed call becomes a MonitorResult with `error` set and no score, so
        one bad transcript cannot abort a 500-run stage. The evaluation stage
        excludes failures explicitly instead of scoring them as clean, which
        would otherwise manufacture false negatives.
        """
        user = render_for_monitor(transcript)
        base_metadata: dict[str, Any] = {
            "model": self.model.name,
            "prompt": self.prompt.describe(),
            "n_messages": transcript.n_messages,
            "n_characters": transcript.n_characters,
        }

        try:
            completion = self.client.complete(self.prompt.text, user, Verdict)
        except MonitorLLMError as exc:
            logger.warning("run %s failed: %s", transcript.run_id, exc)
            return MonitorResult(
                run_id=transcript.run_id,
                monitor=self.name,
                score=float("nan"),
                reasoning="",
                labels=list(transcript.labels),
                is_positive=transcript.is_positive,
                error=f"{type(exc).__name__}: {exc}",
                metadata=base_metadata,
            )

        verdict = completion.parsed
        assert isinstance(verdict, Verdict)
        return MonitorResult(
            run_id=transcript.run_id,
            monitor=self.name,
            score=float(verdict.score),
            reasoning=verdict.reasoning,
            labels=list(transcript.labels),
            is_positive=transcript.is_positive,
            metadata={
                **base_metadata,
                "from_cache": completion.from_cache,
                "usage": completion.usage.as_dict(),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "monitor": self.name,
            "model": self.model.model_dump(mode="json"),
            "prompt": self.prompt.describe(),
        }
