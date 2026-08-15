#!/usr/bin/env python
"""Inspect one transcript and what each monitor said about it.

Built for checking the prompted-directive confound by hand: prints the opening
system message in full, flags the phrases that look like an injected sandbagging
directive, and shows both arms' scores and reasoning side by side.

    python scripts/show_run.py 329102
    python scripts/show_run.py --recovered        # pick a recovered run
    python scripts/show_run.py 329102 --messages 3
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import pandas as pd

from monitor_localization.config import ExperimentConfig
from monitor_localization.dataset import load_transcripts
from monitor_localization.experiment import ExperimentRun
from monitor_localization.utils import load_env, setup_logging

# Phrasing that only appears in an injected directive to underperform — not in a
# legitimate task description. Kept narrow deliberately: "instructions" alone
# matches ordinary task text and would overcount.
DIRECTIVE = re.compile(
    r"(?:you (?:should|must|will|are to)\s+(?:secretly|deliberately|intentionally|subtly)"
    r"|do not actually|without (?:revealing|being detected|admitting)"
    r"|never (?:mention|reveal|admit)|your (?:secret|true) goal"
    r"|pretend to|appear to be trying|make it look like|sandbag"
    r"|intentionally (?:underperform|fail|introduce)|refuse (?:all|every) task)",
    re.I,
)

RULE = "─" * 78


def _section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def _wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(line, 96, initial_indent=indent, subsequent_indent=indent)
        if line.strip() else ""
        for line in text.splitlines()
    )


def _results(run: ExperimentRun) -> dict[int, dict]:
    if not run.results_path.is_file():
        return {}
    rows = {}
    for line in run.results_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["run_id"]] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=int, nargs="?", default=None)
    parser.add_argument("--config", default="fixed")
    parser.add_argument(
        "--recovered", action="store_true",
        help="pick the largest recovery instead of naming a run_id",
    )
    parser.add_argument("--messages", type=int, default=1,
                        help="how many opening messages to print")
    parser.add_argument("--all", action="store_true",
                        help="print every message, not just the opening ones")
    parser.add_argument("--chars", type=int, default=9000,
                        help="per-message character cap (ignored with --all)")
    parser.add_argument(
        "--stripped", action="store_true",
        help="show what the monitor actually sees: scaffolding header removed and "
             "prompt echoes redacted, i.e. the stripped config's view",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="write to this file instead of stdout (transcripts "
                             "run to hundreds of thousands of characters)")
    args = parser.parse_args()

    load_env()
    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    chunk = _results(ExperimentRun(cfg, stage="chunk"))
    glob = _results(ExperimentRun(cfg, stage="global"))

    run_id = args.run_id
    if run_id is None:
        paired_path = ExperimentRun(cfg, stage="chunk").results_path.parent / "paired.csv"
        if not paired_path.is_file():
            print("No paired.csv — run scripts/compare.py, or pass a run_id.")
            return 1
        paired = pd.read_csv(paired_path)
        pool = paired[paired["transition"] == "recovered"] if args.recovered else paired
        if pool.empty:
            print("Nothing to show.")
            return 1
        run_id = int(pool.nlargest(1, "delta").iloc[0]["run_id"])
        print(f"(no run_id given — showing the largest recovery, run {run_id})")

    # cfg= is required or the loader falls back to DatasetConfig() defaults and
    # silently ignores strip_system_prompt — the same omission that once made a
    # whole experiment a no-op. --stripped overrides whatever the config says.
    dataset_cfg = cfg.dataset.model_copy(
        update={"strip_system_prompt": True} if args.stripped else {}
    )
    transcripts = load_transcripts([run_id], cfg=dataset_cfg)
    if not transcripts:
        print(f"Run {run_id} not found in the dataset.")
        return 1
    transcript = transcripts[0]

    stream = args.out.open("w") if args.out else None
    redirect = None
    if stream is not None:
        import contextlib

        redirect = contextlib.redirect_stdout(stream)
        redirect.__enter__()

    _section(f"run {run_id}")
    print(f"  preprocessing  strip_system_prompt={dataset_cfg.strip_system_prompt}")
    print(f"  labels        {', '.join(transcript.labels) or 'normal'}")
    print(f"  run_source    {transcript.run_source}")
    print(f"  model         {transcript.model}")
    print(f"  messages      {transcript.n_messages}")
    print(f"  characters    {transcript.n_characters:,} "
          f"(~{transcript.n_characters / 4.4:,.0f} tokens)")

    count = transcript.n_messages if args.all else min(args.messages, transcript.n_messages)
    cap = None if args.all else args.chars
    for index in range(count):
        message = transcript.messages[index]
        body = message.content or ""
        hits = list(DIRECTIVE.finditer(body))
        _section(f"message {index} · role={message.role} · {len(body):,} chars"
                 f" · {len(hits)} directive phrase(s)")
        print(_wrap(body if cap is None else body[:cap]))
        if cap is not None and len(body) > cap:
            print(f"\n  … {len(body) - cap:,} more characters (raise --chars or use --all)")
        if hits:
            print("\n  directive phrases matched in this message:")
            for hit in hits[:12]:
                print(f"    · {hit.group(0)!r}  at char {hit.start():,}")

    anywhere = sum(
        1 for m in transcript.messages if DIRECTIVE.search(m.content or "")
    )
    _section("directive across the whole transcript")
    print(f"  messages containing a directive phrase: {anywhere}/{transcript.n_messages}")
    print("  (a later hit is often the agent's OWN reasoning about underperforming,")
    print("   which is genuine behavioural evidence rather than the injected prompt)")

    _section("what each monitor said")
    for name, rows in (("global", glob), ("chunked", chunk)):
        row = rows.get(run_id)
        if not row:
            print(f"\n  {name}: no result")
            continue
        print(f"\n  {name}  score {row['score']}")
        print(_wrap(row.get("reasoning") or "(none)", indent="    "))

    if redirect is not None:
        redirect.__exit__(None, None, None)
        stream.close()
        print(f"wrote run {run_id} ({transcript.n_messages} messages, "
              f"{transcript.n_characters:,} chars) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
