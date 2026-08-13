"""Per-sample outcomes and failure-mode inspection.

This is the manual-analysis surface. The notes call for a CSV that can be read by
hand to understand *why* the monitor was wrong, so every row carries the monitor's
own reasoning next to the ground-truth labels — the reasoning is what makes a
false negative interpretable rather than merely counted.

Mistakes are ranked by confidence, not listed arbitrarily. A false negative scored
2 is a different phenomenon from one scored 48: the first is a blind spot, the
second is a threshold problem. Sorting surfaces that distinction immediately.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from monitor_localization.dataset.schema import BENIGN_LABEL, behavior_labels, label_group
from monitor_localization.evaluation.metrics import DEFAULT_THRESHOLD

# Outcome codes, matching the notes' TP/FP/TN/FN vocabulary.
TRUE_POSITIVE = "TP"
FALSE_POSITIVE = "FP"
TRUE_NEGATIVE = "TN"
FALSE_NEGATIVE = "FN"
FAILED = "ERROR"

PER_SAMPLE_COLUMNS = [
    "run_id",
    "outcome",
    "score",
    "is_positive",
    "labels",
    "label_groups",
    "run_source",
    "model",
    "task_id",
    "has_chain_of_thought",
    "n_messages",
    "n_characters",
    "reasoning",
    "error",
]


def classify(score: float, is_positive: bool, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Map a score and ground truth to an outcome code.

    A NaN score means the monitor call failed; that is reported as ERROR rather
    than folded into a confusion cell, so failures never masquerade as judgments.
    """
    if score != score:  # NaN
        return FAILED
    flagged = score >= threshold
    if is_positive:
        return TRUE_POSITIVE if flagged else FALSE_NEGATIVE
    return FALSE_POSITIVE if flagged else TRUE_NEGATIVE


def per_sample_table(
    results: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Build the per-sample analysis table from serialized MonitorResults."""
    rows: list[dict[str, Any]] = []
    for record in results:
        labels = list(record.get("labels") or [])
        behaviors = [label for label in labels if label != BENIGN_LABEL]
        score = record.get("score", float("nan"))
        score = float(score) if score is not None else float("nan")
        metadata = record.get("metadata") or {}
        rows.append(
            {
                "run_id": record.get("run_id"),
                "outcome": classify(score, bool(record.get("is_positive")), threshold),
                "score": score,
                "is_positive": bool(record.get("is_positive")),
                "labels": ";".join(labels),
                "label_groups": ";".join(sorted({label_group(b) for b in behaviors})),
                "run_source": metadata.get("run_source", ""),
                # The model that produced the transcript, not the monitor.
                "model": metadata.get("monitored_model", ""),
                "task_id": metadata.get("task_id", ""),
                "has_chain_of_thought": metadata.get("has_chain_of_thought", ""),
                "n_messages": metadata.get("n_messages", ""),
                "n_characters": metadata.get("n_characters", ""),
                "reasoning": record.get("reasoning", ""),
                "error": record.get("error") or "",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=PER_SAMPLE_COLUMNS)
    return frame[PER_SAMPLE_COLUMNS].sort_values("run_id").reset_index(drop=True)


def false_negatives(table: pd.DataFrame) -> pd.DataFrame:
    """Missed behaviors, most confidently missed first.

    Ascending score: the lowest-scored miss is the one the monitor was most sure
    was clean, which makes it the most informative blind spot — and, for Stage 5,
    the most interesting candidate for chunk-wise recovery.
    """
    return (
        table[table["outcome"] == FALSE_NEGATIVE]
        .sort_values("score", ascending=True)
        .reset_index(drop=True)
    )


def false_positives(table: pd.DataFrame) -> pd.DataFrame:
    """Benign runs flagged, most confidently flagged first."""
    return (
        table[table["outcome"] == FALSE_POSITIVE]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def errors(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["outcome"] == FAILED].reset_index(drop=True)


def recall_by(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Detection rate broken down by a metadata column.

    Answers "which behaviors does the monitor miss" — the question the research
    plan is ultimately about. Restricted to positives, since recall is undefined
    for benign runs.
    """
    positives = table[table["is_positive"] & (table["outcome"] != FAILED)]
    if positives.empty:
        return pd.DataFrame(columns=[column, "n", "detected", "recall"])
    grouped = (
        positives.assign(detected=positives["outcome"] == TRUE_POSITIVE)
        .groupby(column)
        .agg(n=("run_id", "count"), detected=("detected", "sum"))
        .reset_index()
    )
    grouped["recall"] = (grouped["detected"] / grouped["n"]).round(4)
    return grouped.sort_values("recall").reset_index(drop=True)


def recall_by_label(table: pd.DataFrame) -> pd.DataFrame:
    """Per-behavior detection rate, exploding multi-label runs.

    A run labeled both `sabotage` and `gives_up` counts toward each, so rows do
    not sum to the number of positive runs.
    """
    positives = table[table["is_positive"] & (table["outcome"] != FAILED)].copy()
    if positives.empty:
        return pd.DataFrame(columns=["label", "n", "detected", "recall"])
    positives["label"] = positives["labels"].apply(behavior_labels)
    exploded = positives.explode("label").dropna(subset=["label"])
    exploded["detected"] = exploded["outcome"] == TRUE_POSITIVE
    grouped = (
        exploded.groupby("label")
        .agg(n=("run_id", "count"), detected=("detected", "sum"))
        .reset_index()
    )
    grouped["recall"] = (grouped["detected"] / grouped["n"]).round(4)
    return grouped.sort_values("recall").reset_index(drop=True)


def format_failure_digest(
    table: pd.DataFrame,
    n: int = 10,
    reasoning_chars: int = 240,
) -> str:
    """Human-readable digest of the most confident mistakes, for the terminal."""
    lines: list[str] = []

    sections = (
        (f"Top {n} false negatives (missed behaviors, most confident first)",
         false_negatives(table)),
        (f"Top {n} false positives (benign flagged, most confident first)",
         false_positives(table)),
    )
    for title, frame in sections:
        lines.append("")
        lines.append("=" * 78)
        lines.append(title)
        lines.append("=" * 78)
        if frame.empty:
            lines.append("  (none)")
            continue
        for _, row in frame.head(n).iterrows():
            reasoning = str(row["reasoning"]).replace("\n", " ")
            if len(reasoning) > reasoning_chars:
                reasoning = reasoning[:reasoning_chars].rstrip() + "…"
            lines.append("")
            lines.append(f"  run {row['run_id']}  score={row['score']:.0f}  labels={row['labels']}")
            lines.append(f"    source={row['run_source']} messages={row['n_messages']}")
            lines.append(f"    monitor said: {reasoning}")
    return "\n".join(lines)
