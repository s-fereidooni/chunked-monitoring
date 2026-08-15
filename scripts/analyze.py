#!/usr/bin/env python
"""Stage 3: score a monitor's output and extract its failure cases.

Reads a stage's results.jsonl and writes, next to it:

  metrics.json          confusion matrices, ROC AUC, detection rate at target FPR
  per_sample.csv        every run with its outcome (TP/FP/TN/FN) and the
                        monitor's own reasoning — the manual-inspection surface
  false_negatives.csv   missed behaviors, most confidently missed first
  false_positives.csv   benign runs flagged, most confidently flagged first
  recall_by_label.csv   detection rate per behavior label

    python scripts/analyze.py                      # global stage of the default config
    python scripts/analyze.py --stage chunk
    python scripts/analyze.py --threshold 60 --top 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from monitor_localization.config import ExperimentConfig
from monitor_localization.evaluation import (
    DEFAULT_TARGET_FPR,
    DEFAULT_THRESHOLD,
    errors,
    evaluate,
    false_negatives,
    false_positives,
    format_failure_digest,
    fpr_by_length_decile,
    length_by_label,
    per_sample_table,
    recall_by,
    recall_by_label,
    score_vs_length,
)
from monitor_localization.experiment import ExperimentRun
from monitor_localization.utils import read_jsonl, setup_logging, write_json


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="fixed")
    parser.add_argument("--stage", default="global", help="which stage's results to analyze")
    parser.add_argument("--results", type=Path, default=None, help="explicit results.jsonl path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    parser.add_argument("--top", type=int, default=10, help="failures to show in the digest")
    args = parser.parse_args()

    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    run = ExperimentRun(cfg, stage=args.stage)
    results_path = args.results or run.results_path

    if not results_path.is_file():
        print(f"No results at {results_path}. Run scripts/run_monitor.py first.")
        return 1

    records = list(read_jsonl(results_path))
    if not records:
        print(f"{results_path} is empty.")
        return 1

    table = per_sample_table(records, threshold=args.threshold)
    scores = table["score"].tolist()
    truth = table["is_positive"].tolist()

    report = evaluate(scores, truth, threshold=args.threshold, target_fpr=args.target_fpr)

    _section(f"{args.stage} monitor — {len(records)} runs")
    print(f"  scored:           {report['n_scored']}")
    if report["n_excluded_failures"]:
        print(f"  excluded (failed):{report['n_excluded_failures']}  <- not counted as clean")
    auc = report["roc_auc"]
    print(f"  ROC AUC:          {auc if auc is not None else 'n/a (one class only)'}")

    fixed = report["fixed_threshold"]
    _section(f"At fixed threshold {args.threshold:g}")
    print(f"  TP {fixed['tp']:<5} FP {fixed['fp']:<5} TN {fixed['tn']:<5} FN {fixed['fn']:<5}")
    print(f"  precision {fixed['precision']}  recall {fixed['recall']}  F1 {fixed['f1']}")
    print(f"  false positive rate {fixed['false_positive_rate']}")

    at_fpr = report["at_target_fpr"]
    _section(f"At {args.target_fpr:.0%} false-positive budget (METR's protocol)")
    if at_fpr["detection_rate"] is None:
        print("  n/a — needs both positive and benign runs")
    else:
        print(f"  threshold        {at_fpr['threshold']:.1f}")
        print(f"  detection rate   {at_fpr['detection_rate']}")
        print(f"  achieved FPR     {at_fpr['achieved_fpr']}")

    dist = report["score_distribution"]
    _section("Score distribution")
    for name in ("positive", "negative"):
        stats = dist[name]
        if stats.get("n"):
            print(
                f"  {name:<9} n={stats['n']:<5} median={stats['median']:<6} "
                f"mean={stats['mean']:<6} p10={stats['p10']:<6} p90={stats['p90']}"
            )

    by_label = recall_by_label(table)
    if not by_label.empty:
        _section("Recall by behavior label (worst first)")
        print(by_label.to_string(index=False))

    by_source = recall_by(table, "run_source")
    if not by_source.empty:
        _section("Recall by run_source")
        print(by_source.to_string(index=False))
        print("\n  Natural runs are the realistic signal; prompted runs are easier.")
        print("  A large gap here means headline recall is flattered by prompted runs.")

    # --- length controls -----------------------------------------------------
    lengths = length_by_label(table)
    if not lengths.empty:
        _section("Transcript length by label (tokens)")
        print(lengths.to_string(index=False))
        print("\n  `spread` is p90/p10 — how much length varies WITHIN the label.")
        print("  Length is confounded with label, so a label's AUROC is partly a")
        print("  function of its length distribution. Labels with low spread cannot")
        print("  separate the two.")

    correlations = score_vs_length(table)
    if not correlations.empty:
        _section("Within-label: does length predict the score?")
        print(correlations.to_string(index=False))
        benign_row = correlations[correlations["label"] == "normal"]
        if not benign_row.empty and float(benign_row.iloc[0]["rho"]) > 0:
            print("\n  Positive rho on benign runs means the monitor grows more")
            print("  suspicious of longer transcripts regardless of behavior — a")
            print("  false-positive tendency that max aggregation over more chunks")
            print("  would amplify.")

    fpr_deciles = fpr_by_length_decile(table, threshold=args.threshold)
    if not fpr_deciles.empty:
        _section("False-positive rate by length decile (benign only)")
        print(fpr_deciles.to_string(index=False))

    print(format_failure_digest(table, n=args.top))

    out = results_path.parent
    write_json(out / "metrics.json", {"threshold": args.threshold, **report})
    table.to_csv(out / "per_sample.csv", index=False)
    false_negatives(table).to_csv(out / "false_negatives.csv", index=False)
    false_positives(table).to_csv(out / "false_positives.csv", index=False)
    by_label.to_csv(out / "recall_by_label.csv", index=False)
    lengths.to_csv(out / "length_by_label.csv", index=False)
    correlations.to_csv(out / "score_vs_length.csv", index=False)
    fpr_deciles.to_csv(out / "fpr_by_length_decile.csv", index=False)

    failed = errors(table)
    if not failed.empty:
        failed.to_csv(out / "failed_runs.csv", index=False)

    _section("Written")
    for name in (
        "metrics.json",
        "per_sample.csv",
        "false_negatives.csv",
        "false_positives.csv",
        "recall_by_label.csv",
    ):
        print(f"  {out / name}")
    if not failed.empty:
        print(f"  {out / 'failed_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
