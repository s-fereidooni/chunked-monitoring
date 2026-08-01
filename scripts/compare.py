#!/usr/bin/env python
"""Stage 5: paired comparison of the global and chunked monitors.

Both arms score the same transcripts, so the comparison is paired: run-level
variance cancels, and McNemar's exact test on the discordant pairs is the right
significance test rather than comparing two independent recall estimates.

Writes, next to the variant's results:

  comparison.json       McNemar, both arms' headline metrics, the length control
  paired.csv            every run with both scores, its transition, and length
  transitions.csv       per-label recovered / lost / both-caught / both-missed
  delta_vs_length.csv   within-label correlation of the score delta with length

    python scripts/compare.py
    python scripts/compare.py --threshold 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from monitor_localization.config import ExperimentConfig
from monitor_localization.evaluation import (
    DEFAULT_TARGET_FPR,
    DEFAULT_THRESHOLD,
    LOST,
    RECOVERED,
    delta_vs_length,
    evaluate,
    mcnemar,
    pair_results,
    transition_counts,
)
from monitor_localization.experiment import ExperimentRun
from monitor_localization.utils import read_jsonl, setup_logging, write_json


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _arm(paired, suffix: str, threshold: float, target_fpr: float) -> dict:
    """Headline metrics for one arm, computed on the paired rows only.

    Restricting to paired rows matters: an arm evaluated on runs the other arm
    never scored is not comparable, and the missing runs are rarely a random
    sample of the subset.
    """
    return evaluate(
        paired[f"score_{suffix}"].tolist(),
        paired["is_positive"].tolist(),
        threshold=threshold,
        target_fpr=target_fpr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default")
    parser.add_argument("--baseline", default="global")
    parser.add_argument("--variant", default="chunk")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    parser.add_argument("--top", type=int, default=10, help="recoveries to show")
    args = parser.parse_args()

    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    runs = {
        name: ExperimentRun(cfg, stage=name)
        for name in (args.baseline, args.variant)
    }
    for name, run in runs.items():
        if not run.results_path.is_file():
            print(f"No {name} results at {run.results_path}. Run scripts/run_monitor.py.")
            return 1

    baseline = list(read_jsonl(runs[args.baseline].results_path))
    variant = list(read_jsonl(runs[args.variant].results_path))
    paired = pair_results(baseline, variant, threshold=args.threshold)

    if paired.empty:
        print("No runs scored by both monitors.")
        return 1

    _section(f"Paired: {args.baseline} vs {args.variant}")
    print(f"  {args.baseline:<10} scored {len(baseline)}")
    print(f"  {args.variant:<10} scored {len(variant)}")
    print(f"  paired (both, neither failed): {len(paired)}")
    dropped = max(len(baseline), len(variant)) - len(paired)
    if dropped:
        print(f"  {dropped} run(s) not in both arms — excluded, not counted as misses")

    base_report = _arm(paired, "baseline", args.threshold, args.target_fpr)
    var_report = _arm(paired, "variant", args.threshold, args.target_fpr)

    _section("Headline metrics (paired rows only)")
    rows = [
        ("ROC AUC", base_report["roc_auc"], var_report["roc_auc"]),
        (
            f"recall @ {args.threshold:g}",
            base_report["fixed_threshold"]["recall"],
            var_report["fixed_threshold"]["recall"],
        ),
        (
            f"FPR @ {args.threshold:g}",
            base_report["fixed_threshold"]["false_positive_rate"],
            var_report["fixed_threshold"]["false_positive_rate"],
        ),
        (
            f"TPR @ {args.target_fpr:.0%} FPR",
            base_report["at_target_fpr"]["detection_rate"],
            var_report["at_target_fpr"]["detection_rate"],
        ),
    ]
    print(f"  {'metric':<20}{args.baseline:>12}{args.variant:>12}{'delta':>12}")
    for label, b, v in rows:
        delta = f"{v - b:+.3f}" if b is not None and v is not None else "n/a"
        print(f"  {label:<20}{_fmt(b):>12}{_fmt(v):>12}{delta:>12}")

    test = mcnemar(paired)
    _section("McNemar's exact test on positives")
    print(f"  positives           {test['n_positive']}")
    print(f"  recovered           {test['recovered']}  ({args.baseline} missed, "
          f"{args.variant} caught)")
    print(f"  lost                {test['lost']}  ({args.baseline} caught, "
          f"{args.variant} missed)")
    print(f"  discordant pairs    {test['discordant']}  <- the only informative runs")
    print(f"  net                 {test['net']:+d}")
    # Rounded to 4dp a strong result prints as "0.0", which reads as absent
    # rather than tiny.
    p = test["p_value"]
    print(f"  p                   {'<0.0001' if p < 0.0001 else f'{p:.4f}'}")
    print("\n  Concordant runs carry no information; raw recall differences")
    print("  overstate the evidence relative to this test.")

    transitions = transition_counts(paired)
    _section("Transitions by label")
    print(transitions.to_string(index=False))

    # --- the control for "chunking just takes more draws" ---------------------
    lengths = delta_vs_length(paired)
    _section("Within-label: does length explain the score delta?")
    if lengths.empty:
        print("  Not enough runs per label to test.")
    else:
        print(lengths.to_string(index=False))
        print("\n  Max aggregation takes more draws on longer transcripts, so a")
        print("  delta that rises with length within every label — including")
        print("  benign, where any rise is a false positive — is the signature of")
        print("  more draws rather than better localization.")

    _section(f"Recoveries (top {args.top} by delta)")
    recovered = paired[paired["transition"] == RECOVERED].nlargest(args.top, "delta")
    for _, row in recovered.iterrows():
        print(f"\n  run {row['run_id']}  [{row['labels']}]  "
              f"{row['score_baseline']:.0f} -> {row['score_variant']:.0f}  "
              f"(~{row['tokens']:,.0f} tokens)")
        print(f"    {row['reasoning_variant']}")

    lost = paired[paired["transition"] == LOST]
    if not lost.empty:
        _section(f"Regressions ({len(lost)})")
        for _, row in lost.iterrows():
            print(f"\n  run {row['run_id']}  [{row['labels']}]  "
                  f"{row['score_baseline']:.0f} -> {row['score_variant']:.0f}  "
                  f"(~{row['tokens']:,.0f} tokens)")
            print(f"    {row['reasoning_variant']}")

    out: Path = runs[args.variant].results_path.parent
    write_json(
        out / "comparison.json",
        {
            "baseline": args.baseline,
            "variant": args.variant,
            "threshold": args.threshold,
            "target_fpr": args.target_fpr,
            "n_paired": len(paired),
            "mcnemar": test,
            "metrics": {args.baseline: base_report, args.variant: var_report},
        },
    )
    paired.to_csv(out / "paired.csv", index=False)
    transitions.to_csv(out / "transitions.csv", index=False)
    lengths.to_csv(out / "delta_vs_length.csv", index=False)

    _section("Written")
    for name in ("comparison.json", "paired.csv", "transitions.csv", "delta_vs_length.csv"):
        print(f"  {out / name}")
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
