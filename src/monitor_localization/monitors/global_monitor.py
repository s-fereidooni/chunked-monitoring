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
from monitor_localization.monitors.chunking import render_with_context
from monitor_localization.llm.client import MonitorClient, MonitorLLMError
from monitor_localization.monitors.base import Monitor, MonitorResult, Verdict
from monitor_localization.prompts import load_prompt
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)


def render_for_monitor(transcript: Transcript, context_messages: int = 5) -> str:
    """Format one transcript as the monitor's user message.

    Uses METR's MALT layout — the opening messages in `<first_few_messages>`,
    the content under review in `<transcript_chunk>`, separated by a literal
    ellipsis. The global monitor puts the whole transcript in the chunk slot, so
    the only difference from the chunked condition is what fills that slot.

    No task_id or message-count header: those are metadata the monitor would not
    have in deployment, and a message count is a length cue we do not want
    influencing a comparison where length is already a confound.

    `context_messages` must match `ChunkConfig.context_messages`, or the two arms
    differ by prompt shape as well as by chunking. Callers pass it from config;
    the default here only matches METR's baseline by coincidence.
    """
    # The body is the whole transcript, so it already contains the opening
    # messages; repeating them would make the global arm read its head twice.
    # render_with_context applies the same suppression rule the chunked arm uses,
    # which makes this exactly the single-chunk case.
    del context_messages  # kept in the signature for callers; see chunking.py
    return render_with_context([], transcript.messages)


class GlobalMonitor(Monitor):
    """Whole-transcript monitor."""

    name = "global"

    def __init__(
        self,
        client: MonitorClient | None = None,
        model: ModelConfig | None = None,
        prompt_version: str = "v1",
        target: str | None = None,
    ) -> None:
        self.model = model or ModelConfig()
        self.client = client or MonitorClient(self.model)
        self.target = target
        # v3 onward carries a prompt per monitored behavior, following METR's
        # separate reward-hacking and sandbagging monitors.
        name = f"{target}_global" if target else "global_monitor"
        self.prompt = load_prompt(name, prompt_version)

    def judge(self, transcript: Transcript) -> MonitorResult:
        """Score a transcript, recording failures rather than raising.

        A failed call becomes a MonitorResult with `error` set and no score, so
        one bad transcript cannot abort a 500-run stage. The evaluation stage
        excludes failures explicitly instead of scoring them as clean, which
        would otherwise manufacture false negatives.
        """
        user = render_for_monitor(transcript)
        # Two distinct models are in play: the monitor, and the model that
        # produced the transcript. Both are recorded under unambiguous keys — the
        # failure analysis breaks recall down by the monitored run's provenance
        # (notably prompted vs natural), which needs that metadata carried through.
        base_metadata: dict[str, Any] = {
            "monitor_model": self.model.name,
            "prompt": self.prompt.describe(),
            "monitored_model": transcript.model,
            "run_source": transcript.run_source,
            "has_chain_of_thought": transcript.has_chain_of_thought,
            "manually_reviewed": transcript.manually_reviewed,
            "task_id": transcript.task_id,
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
