#!/usr/bin/env python
"""Stage 1: build the reproducible evaluation subset.

Writes `data/evaluation_subset.json` — the run_ids plus the config and
composition that produced them. That file is tracked in git; the transcripts are
not, so the subset is reproducible without redistributing gated data.

    python scripts/build_subset.py
    python scripts/build_subset.py --n-samples 200 --seed 1 --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from monitor_localization.config import ExperimentConfig
from monitor_localization.dataset import (
    GatedDatasetError,
    build_subset,
    metadata_table,
    save_subset,
)
from monitor_localization.paths import EVALUATION_SUBSET
from monitor_localization.utils import load_env, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default", help="config name or path")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the composition without writing the manifest",
    )
    args = parser.parse_args()

    load_env()
    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    subset_cfg = cfg.subset
    overrides = {}
    if args.n_samples is not None:
        overrides["n_samples"] = args.n_samples
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        subset_cfg = subset_cfg.model_copy(update=overrides)

    try:
        metadata = metadata_table(cfg.dataset)
    except GatedDatasetError as exc:
        print(f"\n{exc}\n")
        return 1

    manifest = build_subset(metadata, subset_cfg)

    print(f"\nCandidate pool: {manifest['candidate_pool_size']} runs")
    print(f"Selected:       {manifest['n_selected']} runs\n")
    print("Composition")
    for key, value in manifest["composition"].items():
        if isinstance(value, dict):
            print(f"  {key}")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key:22} {sub_value:>6}")
        else:
            print(f"  {key:24} {value:>6}")
    print("\nLabels")
    for label, count in manifest["label_counts"].items():
        print(f"  {label:28} {count:>6}")

    if manifest["warnings"]:
        print("\nWarnings")
        for warning in manifest["warnings"]:
            print(f"  - {warning}")

    if args.dry_run:
        print("\n[dry run] manifest not written")
        return 0

    path = save_subset(manifest, args.output or EVALUATION_SUBSET)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
