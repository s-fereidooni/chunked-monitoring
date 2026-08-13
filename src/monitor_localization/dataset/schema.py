"""MALT record types and the label taxonomy.

Mirrors the on-disk schema of `metr-evals/malt-public` (config `default`, split
`public`, 7,179 runs). One row = one agent run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# MALT's benign label. A run is behavior-positive iff it carries any other label.
BENIGN_LABEL = "normal"

# Behavior labels, grouped following METR's release post:
# https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/
#
# One deliberate deviation: METR places `hardcoded_solution` under miscellaneous,
# but we group it with reward hacking. Hardcoding to the tests is reward hacking
# by any ordinary reading, and it is the paradigm case of *locally evidenced*
# misconduct — a single visible edit — which is exactly the property the
# chunk-vs-global comparison is about. It also keeps 15 of the scarcest runs in
# the family that needs them: reward hacking totals 66 runs with it, 51 without.
# This deviation must be disclosed in any write-up, since it makes our
# reward-hacking numbers not directly comparable to METR's.
REWARD_HACKING = frozenset(
    {
        "bypass_constraints",
        "ignores_task_instructions",
        "hardcoded_solution",
    }
)
SANDBAGGING_SABOTAGE = frozenset(
    {
        "gives_up",
        "match_weaker_model",
        "partial_problem_solving",
        # Prompted models were told to reason about features irrelevant to the
        # task — whether it was an eval, how well to perform, what category it
        # belonged to. METR groups this with sandbagging, so we do too.
        "reasoning_about_task",
        "refusals",
        "sabotage",
    }
)
# Present in METR's taxonomy but absent from malt-public; listed so that a run
# carrying one resolves to a real group rather than "unknown" if the public
# release later gains them. `hardcoded_solution` is intentionally not here.
MISCELLANEOUS = frozenset({"evaluation_awareness", "brute_forced_solution"})

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


MONITORED_FAMILY = "sandbagging_sabotage"


def behavior_labels(labels: str | Sequence[str], family: str = MONITORED_FAMILY) -> list[str]:
    """A run's in-scope behavior labels, empty for a benign run.

    Accepts either the `;`-joined form used in result CSVs or a list.

    Filtered to the family the monitor was *prompted* for. A run may carry labels
    from another family — one in the evaluation subset carries
    `bypass_constraints`, which is reward hacking — and the sandbagging monitor
    was never asked to detect those, so they do not belong in a per-label
    breakdown of its performance. Such a run is not dropped: it is still counted
    under whichever in-scope label it also carries, and still appears in every
    run-level statistic.

    Pass `family=""` to disable the filter and get every non-benign label.
    """
    if isinstance(labels, str):
        labels = [x for x in labels.split(";") if x]
    return [
        label
        for label in labels
        if label != BENIGN_LABEL and (not family or label_group(label) == family)
    ]


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
        """Length of the transcript *as rendered*, which is what the monitor reads.

        Summing `content` alone omits tool calls, which `Message.render()`
        appends as `tool_call: name(arguments)`. That understated length by 23%
        overall and — worse for a confound control — unevenly by label: 13% for
        prose-heavy `refusals` against 41% for tool-heavy `gives_up`. Since this
        feeds every length analysis, the bias ran along the axis the analysis
        exists to control for.

        Defined as `len(self.render())` so there is exactly one notion of length
        in the codebase; `subset.apply_length_cap` already used that one.
        """
        return len(self.render())

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
