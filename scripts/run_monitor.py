#!/usr/bin/env python
"""Stage 2: run the global monitor over the evaluation subset.

Results stream to `results/<experiment>/global/results.jsonl` as they complete, so
an interrupted run keeps everything already scored — and because the response
cache is keyed on the request, resuming re-reads those calls for free.

    python scripts/run_monitor.py --limit 5        # smoke test
    python scripts/run_monitor.py --calibrate 20   # size max_tokens from data
    python scripts/run_monitor.py                  # full subset
"""

from __future__ import annotations

import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default")
    parser.add_argument(
        "--monitor",
        choices=("global", "chunk"),
        default="global",
        help="which monitor architecture to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N runs")
    parser.add_argument(
        "--calibrate",
        type=int,
        metavar="N",
        default=None,
        help="measure completion-token usage on N runs instead of scoring the subset",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, make no calls")
    args = parser.parse_args()

    load_env()
    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)

    try:
        run_ids = subset_run_ids()
    except FileNotFoundError:
        print("No evaluation subset found. Run scripts/build_subset.py first.")
        return 1

    if args.limit:
        run_ids = run_ids[: args.limit]
    if args.calibrate:
        run_ids = run_ids[: args.calibrate]

    logger.info("Loading %d transcripts", len(run_ids))
    try:
        transcripts = load_transcripts(run_ids)
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

    run = ExperimentRun(cfg, stage=args.monitor)
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

    run.write_manifest(
        monitor=monitor.describe(),
        n_transcripts=len(transcripts),
        n_failures=failures,
        usage=client.usage.as_dict(),
        cache=client.cache.stats,
        run_ids=run_ids,
    )

    print(f"\nScored {len(transcripts) - failures}/{len(transcripts)} transcripts")
    if failures:
        print(f"  {failures} failed — see the `error` field in results.jsonl")
    print(f"  usage: {client.usage.as_dict()}")
    print(f"  cache: {client.cache.stats}")
    print(f"  results:  {run.results_path}")
    print(f"  manifest: {run.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
