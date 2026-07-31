"""Stage 4: the chunk-wise monitor.

Splits a transcript, scores each chunk independently, and aggregates. Every
per-chunk verdict is retained on the result, so aggregation can be re-run
offline with a different method without paying for inference again — and so the
per-chunk scores are available as an approximate localization of the evidence.
"""

from __future__ import annotations

from typing import Any

from monitor_localization.config import AggregationConfig, ChunkConfig, ModelConfig
from monitor_localization.dataset.schema import Transcript
from monitor_localization.llm.client import MonitorClient, MonitorLLMError
from monitor_localization.monitors.aggregation import aggregate_with_evidence
from monitor_localization.monitors.base import (
    ChunkVerdict,
    Monitor,
    MonitorResult,
    Verdict,
)
from monitor_localization.monitors.chunking import split_transcript
from monitor_localization.prompts import load_prompt
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkMonitor(Monitor):
    """Scores a transcript by scoring its chunks and aggregating."""

    name = "chunk"

    def __init__(
        self,
        client: MonitorClient | None = None,
        model: ModelConfig | None = None,
        chunk: ChunkConfig | None = None,
        aggregation: AggregationConfig | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self.model = model or ModelConfig()
        self.chunk_cfg = chunk or ChunkConfig()
        self.aggregation_cfg = aggregation or AggregationConfig()
        self.client = client or MonitorClient(self.model)
        self.prompt = load_prompt("chunk_monitor", prompt_version)

    def judge(self, transcript: Transcript) -> MonitorResult:
        """Score every chunk, then aggregate.

        A chunk whose call fails contributes NaN and is dropped by the
        aggregator rather than counted as clean — the same reasoning as the
        global monitor's NaN handling. Only if *every* chunk fails does the
        transcript become an error.
        """
        chunks = split_transcript(transcript, self.chunk_cfg)
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
            "chunk": self.chunk_cfg.model_dump(mode="json"),
            "aggregation": self.aggregation_cfg.model_dump(mode="json"),
            "n_chunks": len(chunks),
        }

        if not chunks:
            return MonitorResult(
                run_id=transcript.run_id,
                monitor=self.name,
                score=float("nan"),
                reasoning="",
                labels=list(transcript.labels),
                is_positive=transcript.is_positive,
                error="EmptyTranscript: no messages to chunk",
                metadata=base_metadata,
            )

        verdicts: list[ChunkVerdict] = []
        failures = 0
        for chunk in chunks:
            try:
                completion = self.client.complete(
                    self.prompt.text, chunk.render(), Verdict
                )
            except MonitorLLMError as exc:
                failures += 1
                logger.warning(
                    "run %s chunk %s failed: %s", transcript.run_id, chunk.index, exc
                )
                verdicts.append(
                    ChunkVerdict(
                        index=chunk.index,
                        score=float("nan"),
                        reasoning=f"{type(exc).__name__}: {exc}",
                        start_message=chunk.start,
                        end_message=chunk.end,
                    )
                )
                continue
            verdict = completion.parsed
            assert isinstance(verdict, Verdict)
            verdicts.append(
                ChunkVerdict(
                    index=chunk.index,
                    score=float(verdict.score),
                    reasoning=verdict.reasoning,
                    start_message=chunk.start,
                    end_message=chunk.end,
                )
            )

        scores = [v.score for v in verdicts]
        score, top_index = aggregate_with_evidence(scores, self.aggregation_cfg)

        if score != score:
            return MonitorResult(
                run_id=transcript.run_id,
                monitor=self.name,
                score=score,
                reasoning="",
                labels=list(transcript.labels),
                is_positive=transcript.is_positive,
                error=f"AllChunksFailed: {failures}/{len(chunks)} chunk calls failed",
                chunk_verdicts=verdicts,
                metadata=base_metadata,
            )

        # Surface the driving chunk's reasoning as the transcript-level rationale,
        # so the failure digest reads the same way for both monitors.
        reasoning = ""
        if top_index is not None:
            driver = verdicts[top_index]
            reasoning = (
                f"[chunk {driver.index}, messages "
                f"{driver.start_message}-{driver.end_message - 1}] {driver.reasoning}"
            )

        return MonitorResult(
            run_id=transcript.run_id,
            monitor=self.name,
            score=score,
            reasoning=reasoning,
            labels=list(transcript.labels),
            is_positive=transcript.is_positive,
            chunk_verdicts=verdicts,
            metadata={
                **base_metadata,
                "n_failed_chunks": failures,
                "top_chunk": top_index,
                "chunk_scores": scores,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "monitor": self.name,
            "model": self.model.model_dump(mode="json"),
            "chunk": self.chunk_cfg.model_dump(mode="json"),
            "aggregation": self.aggregation_cfg.model_dump(mode="json"),
            "prompt": self.prompt.describe(),
        }
