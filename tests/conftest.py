"""Synthetic MALT-shaped fixtures.

MALT is gated and ~18 GB, so unit tests run against hand-built rows that mirror
the real parquet schema rather than the dataset itself.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def make_message(
    role: str = "assistant",
    content: str = "hello",
    tool: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """One raw message, shaped like the parquet schema (structs always present)."""
    return {
        "role": role,
        "content": content,
        "name": None,
        "function_call": {"name": tool or "", "arguments": '{"x": 1}' if tool else ""},
        "metadata": {
            "completion_type": metadata.get("completion_type", "chat"),
            "contains_function_call": tool is not None,
            "function_call_result": False,
            "node_id": metadata.get("node_id", 0),
            "parent_node_id": None,
            "branch_id": 0,
            "timestamp": 0,
        },
    }


def make_sample(n_input: int, output_content: str | None = None) -> dict[str, Any]:
    """One (input, output) pair. `output` is list-of-choices, each a message list.

    The completion is distinct per sample by default. A shared literal here once
    hid a real bug: three samples all completing with "done" collapsed to one
    message under a content-keyed dedup, and the test asserted the collapsed
    count as correct.
    """
    content = output_content if output_content is not None else f"done {n_input}"
    return {
        "input": [make_message(content=f"msg {i}", node_id=i) for i in range(n_input)],
        "output": [[make_message(content=content, node_id=n_input)]],
        "metadata": {"unmatched": False, "has_chain_of_thought": True},
    }


def make_row(
    run_id: int = 1,
    labels: list[str] | None = None,
    manually_reviewed: bool = True,
    run_source: str = "unprompted",
    n_samples: int = 3,
) -> dict[str, Any]:
    """One raw MALT row: growing samples plus the run's metadata struct."""
    return {
        "samples": [make_sample(n_input=i + 1) for i in range(n_samples)],
        "metadata": {
            "has_chain_of_thought": True,
            "labels": labels if labels is not None else ["normal"],
            "manually_reviewed": manually_reviewed,
            "model": "claude-sonnet-4-20250514",
            "public": True,
            "run_id": run_id,
            "run_source": run_source,
            "task_id": "task_family/task_a",
        },
    }


def make_metadata(
    n_natural_positive: int = 20,
    n_prompted_positive: int = 400,
    n_negative: int = 300,
    n_unreviewed: int = 1000,
) -> pd.DataFrame:
    """Metadata table with MALT's real skew: positives overwhelmingly prompted."""
    rows: list[dict[str, Any]] = []
    run_id = 0

    def add(labels: list[str], reviewed: bool, source: str, model: str) -> None:
        nonlocal run_id
        run_id += 1
        rows.append(
            {
                "run_id": run_id,
                "task_id": f"family/task_{run_id % 17}",
                "model": model,
                "labels": labels,
                "manually_reviewed": reviewed,
                "run_source": source,
                "has_chain_of_thought": run_id % 2 == 0,
            }
        )

    for _ in range(n_natural_positive):
        add(["bypass_constraints"], True, "unprompted", "claude-opus-4-20250514")
    for _ in range(n_prompted_positive):
        add(["sabotage"], True, "prompted", "claude-3-7-sonnet-20250219")
    for _ in range(n_negative):
        add(["normal"], True, "unprompted", "claude-opus-4-20250514")
    for _ in range(n_unreviewed):
        add(["normal"], False, "unprompted", "o1-preview")

    df = pd.DataFrame(rows)
    df["is_positive"] = df["labels"].apply(lambda ls: any(x != "normal" for x in ls))
    df["n_labels"] = df["labels"].apply(len)
    df["is_prompted"] = df["run_source"] == "prompted"
    return df


@pytest.fixture
def metadata() -> pd.DataFrame:
    return make_metadata()
