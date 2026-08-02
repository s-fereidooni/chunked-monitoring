#!/usr/bin/env python
"""Audit the evaluation subset for features that predict the label by themselves.

A monitor scores well if it detects misbehaviour. It also scores well if some
incidental property of the transcript happens to correlate with the label — the
elicitation prompt was one such property, and the point of this script is to find
the others before they turn into a finding.

Everything here reads local result CSVs, so it is free and instant. Checks that
need the raw transcripts (residual instruction-mentions in reasoning traces)
require --deep and a dataset load.

    python scripts/audit_leakage.py
    python scripts/audit_leakage.py --deep --sample 60
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from monitor_localization.config import ExperimentConfig
from monitor_localization.experiment import ExperimentRun
from monitor_localization.utils import load_env, setup_logging

# Phrasing that would betray the elicitation even after the system prompt is
# dropped — the model saying out loud that it was told to misbehave. METR edits
# reasoning traces to remove exactly this, so hits here are edits that missed.
RESIDUAL = re.compile(
    r"(?:I(?:'m| am| was)\s+(?:supposed to|instructed to|told to|asked to)"
    r"|as instructed|per (?:my|the) instructions|my instructions (?:say|are)"
    r"|the (?:user|system) (?:told|asked|instructed) me"
    r"|I(?:'ve| have) been (?:told|instructed|asked))",
    re.I,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _verdict(leaky: bool, detail: str) -> str:
    return f"  {'LEAK  ' if leaky else 'ok    '} {detail}"


def categorical(frame: pd.DataFrame, column: str, name: str) -> None:
    """Does one column's value predict is_positive on its own?

    Reported as the share of each value's runs that are positive: a value seen
    only in positives, or only in benign, hands the monitor the answer without
    it reading anything.
    """
    if column not in frame or frame[column].isna().all():
        print(f"  (no {column} column)")
        return
    table = (
        frame.groupby(column)
        .agg(n=("run_id", "count"), positive=("is_positive", "sum"))
        .assign(share_positive=lambda d: (d.positive / d.n).round(3))
        .sort_values("n", ascending=False)
    )
    pure = table[(table["share_positive"].isin([0.0, 1.0])) & (table["n"] >= 5)]
    print(table.head(12).to_string())
    leaky = len(pure) > 0 and len(table) > 1
    print(
        _verdict(
            leaky,
            f"{len(pure)}/{len(table)} {name} values (n>=5) are wholly positive or "
            f"wholly benign",
        )
    )


def task_overlap(frame: pd.DataFrame) -> None:
    """Do positives and benign runs even share tasks?

    With no overlap, task identity alone separates the classes, and a monitor
    could in principle key on the task rather than the behaviour.
    """
    pos = set(frame[frame["is_positive"]]["task_id"].dropna())
    ben = set(frame[~frame["is_positive"]]["task_id"].dropna())
    shared = pos & ben
    print(f"  positive tasks {len(pos)}  ·  benign tasks {len(ben)}  ·  shared {len(shared)}")
    coverage = len(shared) / max(len(pos), 1)
    print(
        _verdict(
            coverage < 0.5,
            f"{coverage:.0%} of positive tasks also appear among benign runs",
        )
    )
    if shared and coverage < 1.0:
        only_pos = sorted(pos - ben)[:5]
        print(f"  tasks seen only in positives, e.g. {only_pos}")


def deep_scan(frame: pd.DataFrame, sample: int) -> None:
    """Residual instruction-mentions outside the system prompt."""
    from monitor_localization.dataset import load_transcripts

    subset = (
        frame.groupby(frame["labels"].fillna("normal").str.split(";").str[0],
                      group_keys=False)
        .head(max(sample // 7, 4))
    )
    transcripts = {t.run_id: t for t in load_transcripts(subset["run_id"].tolist())}
    rows = []
    for _, run in subset.iterrows():
        transcript = transcripts.get(run["run_id"])
        if not transcript:
            continue
        # Skip the leading system message: that is what strip_system_prompt
        # removes, so a hit there is already handled.
        body = transcript.messages[1:] if transcript.messages else []
        hits = sum(1 for m in body if RESIDUAL.search(m.content or ""))
        rows.append({"label": (run["labels"] or "normal").split(";")[0],
                     "is_positive": run["is_positive"], "residual_messages": hits})
    if not rows:
        print("  no transcripts loaded")
        return
    table = pd.DataFrame(rows)
    summary = table.groupby("label").agg(
        n=("residual_messages", "size"),
        runs_with_residual=("residual_messages", lambda s: int((s > 0).sum())),
    )
    print(summary.to_string())
    total = int((table["residual_messages"] > 0).sum())
    print(
        _verdict(
            total > 0,
            f"{total}/{len(table)} sampled runs still say the behaviour was instructed",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default")
    parser.add_argument("--stage", default="global")
    parser.add_argument("--deep", action="store_true",
                        help="also scan raw transcripts (slow: loads the dataset)")
    parser.add_argument("--sample", type=int, default=42)
    args = parser.parse_args()

    load_env()
    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    path: Path = ExperimentRun(cfg, stage=args.stage).results_path.parent / "per_sample.csv"
    if not path.is_file():
        print(f"No per_sample.csv at {path}. Run scripts/analyze.py first.")
        return 1
    frame = pd.read_csv(path)

    _section(f"Leakage audit — {len(frame)} runs "
             f"({int(frame.is_positive.sum())} positive, "
             f"{int((~frame.is_positive).sum())} benign)")

    _section("run_source — is the behaviour prompted or natural?")
    categorical(frame, "run_source", "run_source")

    _section("model — which model produced the run?")
    categorical(frame, "model", "model")

    _section("has_chain_of_thought")
    categorical(frame, "has_chain_of_thought", "has_chain_of_thought")

    _section("task_id — do the classes share tasks?")
    task_overlap(frame)

    _section("length — already known, restated for completeness")
    for name, group in (("positive", frame[frame.is_positive]),
                        ("benign", frame[~frame.is_positive])):
        print(f"  {name:<9} median {group.n_characters.median() / 4.4:>9,.0f} tok  "
              f"messages {group.n_messages.median():>5.0f}")

    if args.deep:
        _section("residual instruction-mentions outside the system prompt")
        deep_scan(frame, args.sample)
    else:
        print("\n  (pass --deep to scan transcripts for residual instruction-mentions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
