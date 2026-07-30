"""MALT record types and the label taxonomy.

Mirrors the on-disk schema of `metr-evals/malt-public` (config `default`, split
`public`, 7,179 runs). One row = one agent run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# MALT's benign label. A run is behavior-positive iff it carries any other label.
BENIGN_LABEL = "normal"

# Behavior labels, grouped as in METR's release post. Membership is used for
# per-category breakdowns in the failure analysis, not for scoring.
REWARD_HACKING = frozenset(
    {
        "bypass_constraints",
        "ignores_task_instructions",
        "hardcoded_solution",
        "brute_force",
    }
)
SANDBAGGING_SABOTAGE = frozenset(
    {
        "gives_up",
        "match_weaker_model",
        "partial_problem_solving",
        "refusals",
        "sabotage",
    }
)
MISCELLANEOUS = frozenset({"reasoning_about_task"})

BEHAVIOR_LABELS = REWARD_HACKING | SANDBAGGING_SABOTAGE | MISCELLANEOUS
ALL_LABELS = BEHAVIOR_LABELS | {BENIGN_LABEL}

LABEL_GROUPS: dict[str, frozenset[str]] = {
    "reward_hacking": REWARD_HACKING,
    "sandbagging_sabotage": SANDBAGGING_SABOTAGE,
    "miscellaneous": MISCELLANEOUS,
}


def label_group(label: str) -> str:
    """Return the group a behavior label belongs to.

    Unknown labels return "unknown" rather than raising: MALT may add labels, and
    an unrecognised one should show up in a report, not crash a run.
    """
    if label == BENIGN_LABEL:
        return "benign"
    for group, members in LABEL_GROUPS.items():
        if label in members:
            return group
    return "unknown"


@dataclass(slots=True)
class Message:
    """One message in an agent transcript."""

    role: str
    content: str = ""
    name: str | None = None
    function_call: dict[str, Any] | None = None
    completion_type: str | None = None
    node_id: int | None = None

    @property
    def is_tool_call(self) -> bool:
        return self.function_call is not None and bool(self.function_call.get("name"))

    def render(self) -> str:
        """Plain-text rendering, as shown to the monitor.

        Tool calls are included explicitly — a monitor that cannot see which tool
        was invoked with which arguments is missing most of the signal for
        behaviors like `bypass_constraints`.
        """
        parts = [f"[{self.role}]"]
        if self.name:
            parts.append(f"({self.name})")
        header = " ".join(parts)
        body = self.content or ""
        if self.is_tool_call:
            fc = self.function_call or {}
            body = f"{body}\ntool_call: {fc.get('name')}({fc.get('arguments', '')})".strip()
        return f"{header}\n{body}".strip()

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Message:
        meta = raw.get("metadata") or {}
        fc = raw.get("function_call") or None
        # The parquet schema always materialises the struct; treat an empty name
        # as "no tool call" rather than a call to the empty string.
        if fc is not None and not fc.get("name"):
            fc = None
        return cls(
            role=raw.get("role") or "unknown",
            content=raw.get("content") or "",
            name=raw.get("name"),
            function_call=fc,
            completion_type=meta.get("completion_type"),
            node_id=meta.get("node_id"),
        )


@dataclass(slots=True)
class Transcript:
    """One labeled MALT run, flattened to a linear message list."""

    run_id: int
    task_id: str
    model: str
    labels: list[str]
    manually_reviewed: bool
    run_source: str
    has_chain_of_thought: bool
    messages: list[Message] = field(default_factory=list)

    @property
    def is_positive(self) -> bool:
        """True iff the run carries any behavior label beyond `normal`."""
        return any(label != BENIGN_LABEL for label in self.labels)

    @property
    def behavior_labels(self) -> list[str]:
        return [label for label in self.labels if label != BENIGN_LABEL]

    @property
    def groups(self) -> list[str]:
        return sorted({label_group(label) for label in self.behavior_labels})

    @property
    def is_prompted(self) -> bool:
        return self.run_source == "prompted"

    @property
    def n_messages(self) -> int:
        return len(self.messages)

    @property
    def n_characters(self) -> int:
        return sum(len(m.content or "") for m in self.messages)

    def render(self, separator: str = "\n\n---\n\n") -> str:
        """Full transcript as text, for the global monitor."""
        return separator.join(m.render() for m in self.messages)

    def metadata_row(self) -> dict[str, Any]:
        """Flat dict for DataFrames, manifests, and result CSVs."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "model": self.model,
            "labels": self.labels,
            "manually_reviewed": self.manually_reviewed,
            "run_source": self.run_source,
            "has_chain_of_thought": self.has_chain_of_thought,
            "is_positive": self.is_positive,
            "groups": self.groups,
            "n_messages": self.n_messages,
            "n_characters": self.n_characters,
        }
