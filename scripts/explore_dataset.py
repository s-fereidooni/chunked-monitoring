#!/usr/bin/env python
"""Stage 1: dataset exploration.

Corpus-wide statistics come from the `metadata` column alone, so this is cheap
(a few MB) even though the full split is ~18.6 GB. Transcript length statistics
need the text, so they are computed only for the evaluation subset and only when
`--with-lengths` is passed.

    python scripts/explore_dataset.py
    python scripts/explore_dataset.py --with-lengths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from monitor_localization.config import ExperimentConfig
from monitor_localization.dataset import (
    GatedDatasetError,
    load_transcripts,
    metadata_table,
    subset_run_ids,
)
from monitor_localization.paths import RESULTS_DIR, ensure_dir
from monitor_localization.utils import load_env, setup_logging, write_json

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def label_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Per-label run counts. Labels are multi-valued, so rows sum to > len(df)."""
    exploded = df.explode("labels")
    counts = (
        exploded.groupby("labels")
        .agg(
            runs=("run_id", "count"),
            reviewed=("manually_reviewed", "sum"),
            prompted=("is_prompted", "sum"),
            with_cot=("has_chain_of_thought", "sum"),
        )
        .sort_values("runs", ascending=False)
    )
    counts["pct_of_runs"] = (100 * counts["runs"] / len(df)).round(2)
    return counts


def summarize(df: pd.DataFrame) -> dict[str, object]:
    reviewed = df[df["manually_reviewed"]]
    return {
        "n_runs": int(len(df)),
        "n_models": int(df["model"].nunique()),
        "n_tasks": int(df["task_id"].nunique()),
        "n_reviewed": int(len(reviewed)),
        "n_reviewed_positive": int(reviewed["is_positive"].sum()),
        "n_reviewed_negative": int((~reviewed["is_positive"]).sum()),
        "n_reviewed_positive_prompted": int(
            (reviewed["is_positive"] & reviewed["is_prompted"]).sum()
        ),
        "n_reviewed_positive_natural": int(
            (reviewed["is_positive"] & ~reviewed["is_prompted"]).sum()
        ),
        "n_with_chain_of_thought": int(df["has_chain_of_thought"].sum()),
        "multi_label_runs": int((df["n_labels"] > 1).sum()),
    }


def length_stats(run_ids: list[int]) -> pd.DataFrame:
    """Message/character distribution for the subset transcripts."""
    rows = [t.metadata_row() for t in load_transcripts(run_ids)]
    frame = pd.DataFrame(rows)
    return frame[["n_messages", "n_characters"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default", help="config name or path")
    parser.add_argument(
        "--with-lengths",
        action="store_true",
        help="also load the evaluation subset's transcripts and report length stats",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    load_env()
    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = ensure_dir(args.output_dir or (RESULTS_DIR / "exploration"))

    try:
        df = metadata_table(cfg.dataset)
    except GatedDatasetError as exc:
        print(f"\n{exc}\n")
        return 1

    summary = summarize(df)
    _section("Corpus summary")
    for key, value in summary.items():
        print(f"  {key:34} {value:>8}")

    positives = summary["n_reviewed_positive"]
    if positives:
        share = 100 * summary["n_reviewed_positive_prompted"] / positives
        print(f"\n  prompted share of reviewed positives: {share:.1f}%")
        print("  -> subset.max_prompted_fraction exists to keep this from")
        print("     dominating the positive stratum.")

    labels = label_frequencies(df)
    _section("Label frequencies")
    print(labels.to_string())

    _section("Reviewed x run_source")
    print(pd.crosstab(df["manually_reviewed"], df["run_source"]).to_string())

    _section("Runs per model (top 15)")
    print(df["model"].value_counts().head(15).to_string())

    artifacts: dict[str, object] = {
        "summary": summary,
        "label_frequencies": labels.reset_index().to_dict(orient="records"),
        "runs_per_model": df["model"].value_counts().to_dict(),
        "runs_per_task": df["task_id"].value_counts().to_dict(),
    }

    if args.with_lengths:
        run_ids = subset_run_ids()
        _section(f"Transcript lengths (evaluation subset, n={len(run_ids)})")
        stats = length_stats(run_ids)
        print(stats.to_string())
        artifacts["subset_length_stats"] = stats.to_dict()

    df.drop(columns=["labels"]).assign(labels=df["labels"].apply(";".join)).to_csv(
        out_dir / "run_metadata.csv", index=False
    )
    write_json(out_dir / "exploration.json", artifacts)
    print(f"\nWrote {out_dir / 'exploration.json'} and {out_dir / 'run_metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
