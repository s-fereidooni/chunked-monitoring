"""Splitting a transcript into chunks for localized monitoring.

Follows METR's MALT baseline: the transcript is chunked, and the opening
messages are repeated across chunks so every chunk can see what the agent was
asked to do. Without that, a monitor scoring chunk 7 of 12 has no task
description and every excerpt looks equally unexplained.

The repeated prefix is marked as context and excluded from the chunk's own
message range, so a score always refers to the excerpt rather than the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from monitor_localization.config import ChunkConfig
from monitor_localization.dataset.schema import Message, Transcript


@dataclass(slots=True)
class Chunk:
    """One excerpt, plus the repeated opening context it is shown with."""

    index: int
    start: int
    end: int
    messages: list[Message]
    context: list[Message] = field(default_factory=list)

    @property
    def n_messages(self) -> int:
        return len(self.messages)

    def render(self) -> str:
        """Format for the monitor, keeping context visually separate."""
        parts: list[str] = []
        if self.context:
            body = "\n\n".join(m.render() for m in self.context)
            parts.append(
                "=== CONTEXT: opening messages of the transcript "
                "(for reference, do not score) ===\n" + body
            )
        excerpt = "\n\n".join(m.render() for m in self.messages)
        parts.append(
            f"=== EXCERPT: messages {self.start}-{self.end - 1} "
            f"(score this) ===\n{excerpt}"
        )
        return "\n\n".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "n_messages": self.n_messages,
            "n_context": len(self.context),
        }


def _windows(total: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open [start, end) windows covering `total` items.

    Stride is `size - overlap`. The final window is clamped rather than padded,
    so the last chunk may be shorter but no message is dropped or duplicated
    beyond the configured overlap.
    """
    if total <= 0:
        return []
    stride = size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < size ({size})")
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + size, total)
        spans.append((start, end))
        if end == total:
            break
        start += stride
    return spans


def split_transcript(
    transcript: Transcript,
    cfg: ChunkConfig | None = None,
) -> list[Chunk]:
    """Split a transcript into chunks according to `cfg`.

    `unit="messages"` splits on message boundaries; `unit="characters"` packs
    messages until the character budget is reached, never splitting a message —
    a half message is not something a monitor can sensibly judge.
    """
    cfg = cfg or ChunkConfig()
    messages = transcript.messages
    if not messages:
        return []

    context = messages[: cfg.context_messages] if cfg.context_messages else []

    if cfg.unit == "messages":
        spans = _windows(len(messages), cfg.size, cfg.overlap)
    else:
        spans = _character_windows(messages, cfg.size, cfg.overlap)

    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        # Only attach context when the chunk does not already contain it.
        needs_context = bool(context) and start >= len(context)
        chunks.append(
            Chunk(
                index=index,
                start=start,
                end=end,
                messages=messages[start:end],
                context=list(context) if needs_context else [],
            )
        )
    return chunks


def _character_windows(
    messages: list[Message],
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Windows over message indices, bounded by a character budget.

    Messages are never split. A single message exceeding the budget becomes its
    own chunk rather than being dropped.
    """
    if size <= 0:
        raise ValueError("size must be > 0 for character chunking")
    lengths = [len(m.content or "") for m in messages]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(messages):
        total = 0
        end = start
        while end < len(messages) and (total + lengths[end] <= size or end == start):
            total += lengths[end]
            end += 1
        spans.append((start, end))
        if end >= len(messages):
            break
        # Step back by roughly `overlap` characters worth of messages.
        if overlap > 0:
            back = 0
            new_start = end
            while new_start > start + 1 and back + lengths[new_start - 1] <= overlap:
                new_start -= 1
                back += lengths[new_start]
            start = new_start
        else:
            start = end
    return spans


def chunk_count(transcript: Transcript, cfg: ChunkConfig | None = None) -> int:
    """Number of chunks without materialising them — used for cost estimates."""
    return len(split_transcript(transcript, cfg))
