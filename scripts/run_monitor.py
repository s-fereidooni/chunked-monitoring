#!/usr/bin/env python
"""Stage 2: run the global monitor over the evaluation subset.

Results stream to `results/<experiment>/global/results.jsonl` as they complete, so
an interrupted run keeps everything already scored — and because the response
cache is keyed on the request, resuming re-reads those calls for free.

    python scripts/run_monitor.py --limit 5        # smoke test
    python scripts/run_monitor.py --calibrate 20   # size max_tokens from data
    python scripts/run_monitor.py                  # full subset
    python scripts/run_monitor.py --resume         # score only what is missing
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from monitor_localization.config import ExperimentConfig
from monitor_localization.dataset import (
    GatedDatasetError,
    load_transcripts,
    subset_run_ids,
)
from monitor_localization.experiment import ExperimentRun
from monitor_localization.llm import MonitorClient, ResponseCache
from monitor_localization.monitors import ChunkMonitor, GlobalMonitor
from monitor_localization.utils import append_jsonl, load_env, setup_logging
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)


def calibrate(monitor: GlobalMonitor, transcripts: list, cap: int) -> int:
    """Measure completion-token usage so `model.max_tokens` is set from data.

    Runs with a deliberately loose cap and reports the distribution. Setting the
    cap too high costs nothing (billing is per token generated, not per cap),
    while setting it too low silently kills runs via finish_reason="length" — so
    the cap is a runaway guard, and the right value is comfortably above the
    observed maximum.
    """
    original = monitor.client.cfg.max_tokens
    monitor.client.cfg.max_tokens = cap
    counts: list[int] = []
    try:
        for transcript in transcripts:
            result = monitor.judge(transcript)
            if result.failed:
                logger.warning("run %s failed during calibration", transcript.run_id)
                continue
            counts.append(result.metadata["usage"]["completion_tokens"])
    finally:
        monitor.client.cfg.max_tokens = original

    if not counts:
        print("No successful calls; cannot calibrate.")
        return 1

    counts.sort()
    print(f"\nCompletion tokens over {len(counts)} transcripts (cap {cap}):")
    print(f"  min      {counts[0]}")
    print(f"  median   {int(statistics.median(counts))}")
    print(f"  max      {counts[-1]}")
    suggested = int(counts[-1] * 1.5)
    print(f"\nSuggested model.max_tokens: {suggested}  (observed max x1.5)")
    if suggested > original:
        print(f"  Current value is {original} — raise it, or truncation will lose runs.")
    else:
        print(f"  Current value of {original} has adequate headroom.")
    return 0


def read_results(path: Path) -> list[dict]:
    """Rows of a results file, tolerating a partial final line.

    An interrupted append can leave the file ending mid-object; that line is
    dropped rather than aborting the resume.
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping malformed line in %s", path)
    return rows


def _incomplete(row: dict, retry_partial: bool) -> bool:
    """Is this row missing data that a resume should refetch?

    A run-level error always is. With `retry_partial`, so is a chunked run that
    lost individual chunks to transient errors: those chunks are dropped by the
    aggregator, so the run scored on fewer draws than its length warrants —
    which biases a max aggregate downward and quietly flatters the arm.
    """
    if row.get("error"):
        return True
    return retry_partial and bool((row.get("metadata") or {}).get("n_failed_chunks"))


def prune_failed(path: Path, retry_partial: bool = False) -> set[int]:
    """Drop incomplete rows from a results file; return the run ids that remain.

    The stale row has to go, or the retried run would appear twice and
    downstream analysis would see a phantom failure alongside its own success.
    Re-running a partial row is nearly free: the response cache is keyed per
    chunk, so only the chunks that actually failed cost a call.
    """
    rows = read_results(path)
    kept = [r for r in rows if not _incomplete(r, retry_partial)]
    if len(kept) != len(rows):
        logger.info("dropping %d incomplete rows before resuming", len(rows) - len(kept))
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    return {r["run_id"] for r in kept}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="fixed")
    parser.add_argument(
        "--monitor",
        choices=("global", "chunk"),
        default="global",
        help="which monitor architecture to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N runs")
    parser.add_argument(
        "--run-ids",
        default=None,
        help="restrict to these run_ids: a comma-separated list, or a path to a "
             "file with one id per line. Unlike --limit this preserves whichever "
             "composition the caller chose, so a pilot can stay stratified.",
    )
    parser.add_argument(
        "--calibrate",
        type=int,
        metavar="N",
        default=None,
        help="measure completion-token usage on N runs instead of scoring the subset",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to an existing results file, scoring only the runs missing from it",
    )
    parser.add_argument(
        "--retry-partial",
        action="store_true",
        help="with --resume, also re-run chunked rows that lost chunks to transient errors",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="override model.max_concurrency (lower it to stay under a TPM ceiling)",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, make no calls")
    args = parser.parse_args()

    load_env()
    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.concurrency:
        cfg.model.max_concurrency = args.concurrency

    try:
        run_ids = subset_run_ids()
    except FileNotFoundError:
        print("No evaluation subset found. Run scripts/build_subset.py first.")
        return 1

    if args.run_ids:
        source = Path(args.run_ids)
        raw = source.read_text().split() if source.is_file() else args.run_ids.split(",")
        wanted = {int(value.strip()) for value in raw if value.strip()}
        missing = wanted - set(run_ids)
        if missing:
            print(f"{len(missing)} run_id(s) are not in the subset, e.g. "
                  f"{sorted(missing)[:5]}")
            return 1
        run_ids = [rid for rid in run_ids if rid in wanted]
    if args.limit:
        run_ids = run_ids[: args.limit]
    if args.calibrate:
        run_ids = run_ids[: args.calibrate]

    run = ExperimentRun(cfg, stage=args.monitor)
    all_run_ids = list(run_ids)
    already_scored: set[int] = set()
    if args.resume:
        already_scored = prune_failed(run.results_path, retry_partial=args.retry_partial)
        run_ids = [i for i in run_ids if i not in already_scored]
        logger.info(
            "Resuming: %d already scored, %d to go", len(already_scored), len(run_ids)
        )
        if not run_ids:
            print(f"Nothing to resume — all {len(already_scored)} runs are scored.")
            return 0

    logger.info(
        "Loading %d transcripts (strip_system_prompt=%s)",
        len(run_ids),
        cfg.dataset.strip_system_prompt,
    )
    try:
        # cfg.dataset must be passed explicitly: without it the loader falls back
        # to DatasetConfig() defaults and silently ignores strip_system_prompt,
        # producing byte-identical input that then serves entirely from cache.
        transcripts = load_transcripts(run_ids, cfg=cfg.dataset)
    except GatedDatasetError as exc:
        print(f"\n{exc}\n")
        return 1

    if not transcripts:
        print("No transcripts loaded — check the subset manifest.")
        return 1

    client = MonitorClient(cfg.model, cache=ResponseCache(enabled=not args.no_cache))
    if args.monitor == "chunk":
        monitor = ChunkMonitor(
            client=client,
            model=cfg.model,
            chunk=cfg.chunk,
            aggregation=cfg.aggregation,
            prompt_version=cfg.prompt_version,
            target=cfg.monitor_target,
        )
    else:
        monitor = GlobalMonitor(
            client=client, model=cfg.model, prompt_version=cfg.prompt_version,
            target=cfg.monitor_target,
        )

    if args.dry_run:
        from monitor_localization.monitors import chunk_count

        chars = sum(t.n_characters for t in transcripts)
        # 4.4 chars/token measured on real MALT transcripts (o200k_base).
        tokens = int(chars / 4.4)
        calls = len(transcripts)
        if args.monitor == "chunk":
            calls = sum(chunk_count(t, cfg.chunk) for t in transcripts)
        print(f"\nWould score {len(transcripts)} transcripts with {cfg.model.name}")
        print(f"  monitor: {monitor.name}")
        print(f"  prompt: {monitor.prompt.name} {monitor.prompt.version} "
              f"({monitor.prompt.short_sha})")
        print(f"  API calls: {calls:,}")
        print(f"  transcript characters: {chars:,} (~{tokens:,} tokens)")
        if args.monitor == "chunk":
            print(f"  chunking: size={cfg.chunk.size} overlap={cfg.chunk.overlap} "
                  f"context={cfg.chunk.context_messages} unit={cfg.chunk.unit}")
            print(f"  aggregation: {cfg.aggregation.method}")
        print(f"  output cap: {cfg.model.max_tokens} tokens/call")
        print(f"  concurrency: {cfg.model.max_concurrency}")
        return 0

    if args.calibrate:
        return calibrate(monitor, transcripts, cap=4096)

    if not args.resume:
        run.results_path.unlink(missing_ok=True)
    logger.info("Writing results to %s", run.results_path)

    failures = 0
    with ThreadPoolExecutor(max_workers=cfg.model.max_concurrency) as pool:
        futures = {pool.submit(monitor.judge, t): t for t in transcripts}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result.failed:
                failures += 1
            append_jsonl(run.results_path, result.as_dict())
            if done % 25 == 0 or done == len(transcripts):
                logger.info("%d/%d scored (%d failed)", done, len(transcripts), failures)

    # On a resume the manifest describes the whole results file, not just this
    # invocation — usage and cache stats are per-invocation and labelled as such.
    scored_total = sum(1 for r in read_results(run.results_path) if not r.get("error"))
    run.write_manifest(
        monitor=monitor.describe(),
        n_transcripts=scored_total,
        n_failures=failures,
        resumed=args.resume,
        n_scored_this_invocation=len(transcripts) - failures,
        usage=client.usage.as_dict(),
        cache=client.cache.stats,
        run_ids=all_run_ids,
    )

    print(f"\nScored {len(transcripts) - failures}/{len(transcripts)} transcripts")
    if args.resume:
        print(f"  results file now holds {scored_total}/{len(all_run_ids)} runs")
    if failures:
        print(f"  {failures} failed — see the `error` field in results.jsonl")
    print(f"  usage: {client.usage.as_dict()}")
    print(f"  cache: {client.cache.stats}")
    print(f"  results:  {run.results_path}")
    print(f"  manifest: {run.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
