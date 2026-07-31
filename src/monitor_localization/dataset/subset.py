"""Construction of the reproducible evaluation subset.

Selection is by `run_id` only: the manifest records IDs plus the config that
produced them, so the subset can be rebuilt without redistributing transcripts.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitor_localization.config import SubsetConfig
from monitor_localization.paths import EVALUATION_SUBSET
from monitor_localization.utils.io import write_json
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)


def _sample_ids(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> list[int]:
    """Draw up to `n` run_ids from `pool`, deterministically given `rng`.

    Sorting by run_id first makes the draw independent of shard read order, so
    the same seed yields the same subset across machines.
    """
    if n <= 0 or pool.empty:
        return []
    ids = np.sort(pool["run_id"].to_numpy())
    if n >= len(ids):
        return [int(i) for i in ids]
    picked = rng.choice(ids, size=n, replace=False)
    return sorted(int(i) for i in picked)


def build_subset(
    metadata: pd.DataFrame,
    cfg: SubsetConfig | None = None,
) -> dict[str, Any]:
    """Select the evaluation subset and return its manifest.

    Composition is controlled by three knobs:

    - `manually_reviewed_only` restricts to MALT's reviewed pool (~1.8k runs).
      The unreviewed remainder is assumed benign at an estimated 6% error rate,
      which would put noise directly into the false-positive measurement.
    - `positive_fraction` sets the positive/negative balance.
    - `max_prompted_fraction` caps the share of the *positive* stratum drawn from
      `prompted` runs. Without it the positive half is ~92% prompted, mostly one
      model, and the experiment mostly measures detection of prompted behavior
      rather than the natural instances METR flags as the realistic signal.

    Note `max_characters` is not applied here: transcript lengths are not in the
    metadata column, so any length cap is a monitor-stage truncation concern.
    """
    cfg = cfg or SubsetConfig()
    rng = np.random.default_rng(cfg.seed)

    pool = metadata
    if cfg.manually_reviewed_only:
        pool = pool[pool["manually_reviewed"]]
    pool = pool.reset_index(drop=True)

    if pool.empty:
        raise ValueError("No candidate runs after filtering; check manually_reviewed_only.")

    if cfg.family_quotas:
        return _build_by_family(metadata, pool, cfg, rng)

    positives = pool[pool["is_positive"]]
    negatives = pool[~pool["is_positive"]]

    fraction = cfg.positive_fraction
    if fraction is None:
        # No target balance: draw uniformly from the filtered pool.
        selected = _sample_ids(pool, cfg.n_samples, rng)
        return _manifest(metadata, pool, selected, cfg, warnings=[])

    n_positive = int(round(cfg.n_samples * fraction))
    n_negative = cfg.n_samples - n_positive
    warnings: list[str] = []

    # --- positive stratum, with the prompted cap ---
    natural_pos = positives[~positives["is_prompted"]]
    prompted_pos = positives[positives["is_prompted"]]

    if cfg.max_prompted_fraction is None:
        picked_positive = _sample_ids(positives, n_positive, rng)
    else:
        prompted_quota = int(n_positive * cfg.max_prompted_fraction)
        natural_target = n_positive - prompted_quota

        picked_natural = _sample_ids(natural_pos, natural_target, rng)
        shortfall = natural_target - len(picked_natural)
        if shortfall > 0:
            warnings.append(
                f"Only {len(picked_natural)} natural positive runs available "
                f"(wanted {natural_target}); backfilling {shortfall} from prompted runs, "
                f"which pushes the prompted share above max_prompted_fraction="
                f"{cfg.max_prompted_fraction}."
            )
        picked_prompted = _sample_ids(prompted_pos, prompted_quota + shortfall, rng)
        picked_positive = sorted(picked_natural + picked_prompted)

    if len(picked_positive) < n_positive:
        warnings.append(
            f"Positive stratum short: got {len(picked_positive)} of {n_positive} requested."
        )

    # --- negative stratum ---
    picked_negative = _sample_ids(negatives, n_negative, rng)
    if len(picked_negative) < n_negative:
        warnings.append(
            f"Negative stratum short: got {len(picked_negative)} of {n_negative} requested."
        )

    selected = sorted(picked_positive + picked_negative)
    for message in warnings:
        logger.warning(message)
    return _manifest(metadata, pool, selected, cfg, warnings=warnings)


def _families(labels: list[str]) -> set[str]:
    """Behavior families a run belongs to; empty set means benign."""
    from monitor_localization.dataset.schema import BENIGN_LABEL, label_group

    return {label_group(x) for x in labels if x != BENIGN_LABEL}


def _build_by_family(
    metadata: pd.DataFrame,
    pool: pd.DataFrame,
    cfg: SubsetConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Draw a quota per behavior family.

    Families are filled scarcest-first so that a run belonging to two families is
    kept by the one that can least afford to lose it — with 62 reward-hacking
    runs against 1,073 sandbagging ones, the three overlapping runs matter to the
    former and are noise to the latter.
    """
    quotas = dict(cfg.family_quotas or {})
    benign_quota = quotas.pop("benign", 0)
    warnings: list[str] = []

    pool = pool.copy()
    pool["families"] = pool["labels"].apply(_families)

    pools = {
        family: pool[pool["families"].apply(lambda fs, f=family: f in fs)]
        for family in quotas
    }
    # Scarcest first.
    order = sorted(quotas, key=lambda f: len(pools[f]))

    selected: list[int] = []
    per_family: dict[str, int] = {}
    for family in order:
        available = pools[family]
        available = available[~available["run_id"].isin(selected)]
        quota = quotas[family]
        want = len(available) if quota == "all" else int(quota)
        picked = _sample_ids(available, want, rng)
        if want > len(available):
            warnings.append(
                f"Family {family!r}: wanted {want} but only {len(available)} available."
            )
        per_family[family] = len(picked)
        selected.extend(picked)

    benign_pool = pool[pool["families"].apply(len) == 0]
    want_benign = len(benign_pool) if benign_quota == "all" else int(benign_quota)
    picked_benign = _sample_ids(benign_pool, want_benign, rng)
    if want_benign > len(benign_pool):
        warnings.append(
            f"Benign: wanted {want_benign} but only {len(benign_pool)} available."
        )
    per_family["benign"] = len(picked_benign)
    selected.extend(picked_benign)

    for message in warnings:
        logger.warning(message)

    manifest = _manifest(metadata, pool, sorted(selected), cfg, warnings=warnings)
    manifest["composition"]["by_family"] = per_family
    return manifest


def _manifest(
    metadata: pd.DataFrame,
    pool: pd.DataFrame,
    selected: list[int],
    cfg: SubsetConfig,
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble the manifest: the IDs plus everything needed to audit the draw."""
    chosen = metadata[metadata["run_id"].isin(selected)]
    label_counts = Counter(
        label for labels in chosen["labels"] for label in labels
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "config": cfg.model_dump(mode="json"),
        "candidate_pool_size": int(len(pool)),
        "n_selected": len(selected),
        "composition": {
            "positive": int(chosen["is_positive"].sum()),
            "negative": int((~chosen["is_positive"]).sum()),
            "prompted": int(chosen["is_prompted"].sum()),
            "unprompted": int((~chosen["is_prompted"]).sum()),
            "prompted_positive": int((chosen["is_positive"] & chosen["is_prompted"]).sum()),
            "with_chain_of_thought": int(chosen["has_chain_of_thought"].sum()),
            "n_models": int(chosen["model"].nunique()),
            "n_tasks": int(chosen["task_id"].nunique()),
        },
        "label_counts": dict(sorted(label_counts.items(), key=lambda kv: -kv[1])),
        "warnings": warnings,
        "run_ids": selected,
    }


# Chars per token for MALT transcripts, measured with o200k_base (aggregate 4.40).
_CHARS_PER_TOKEN = 4.4


def apply_length_cap(
    manifest: dict[str, Any],
    cap_tokens: int,
    cfg: Any = None,
) -> dict[str, Any]:
    """Drop selected runs whose transcript exceeds `cap_tokens`, recording them.

    Requires loading the selected transcripts, so this is the slow part of subset
    construction. The excluded runs are kept in the manifest with their sizes:
    they are the runs the global monitor cannot process, which is a result rather
    than a nuisance, and the write-up needs the exact count per family.
    """
    from monitor_localization.dataset.loaders import iter_transcripts
    from monitor_localization.dataset.schema import BENIGN_LABEL, label_group

    run_ids = list(manifest["run_ids"])
    kept: list[int] = []
    excluded: list[dict[str, Any]] = []

    for transcript in iter_transcripts(run_ids, cfg=cfg):
        tokens = int(len(transcript.render()) / _CHARS_PER_TOKEN)
        families = sorted(
            {label_group(x) for x in transcript.labels if x != BENIGN_LABEL}
        ) or ["benign"]
        if tokens > cap_tokens:
            excluded.append(
                {
                    "run_id": transcript.run_id,
                    "tokens": tokens,
                    "n_messages": transcript.n_messages,
                    "family": families[0],
                    "labels": list(transcript.labels),
                }
            )
        else:
            kept.append(transcript.run_id)

    by_family: dict[str, int] = {}
    for row in excluded:
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1

    manifest["run_ids"] = sorted(kept)
    manifest["n_selected"] = len(kept)
    manifest["length_cap"] = {
        "max_transcript_tokens": cap_tokens,
        "n_excluded": len(excluded),
        "excluded_by_family": by_family,
        "reason": (
            "Exceeds the per-request input budget, so the global monitor cannot "
            "score the run in a single call. Excluded from the paired comparison; "
            "reportable as a feasibility difference between architectures."
        ),
        "excluded": sorted(excluded, key=lambda r: -r["tokens"]),
    }
    logger.info(
        "Length cap %d tokens: kept %d, excluded %d (%s)",
        cap_tokens,
        len(kept),
        len(excluded),
        by_family,
    )
    return manifest


def save_subset(manifest: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write the manifest to `data/evaluation_subset.json` (tracked in git)."""
    return write_json(path or EVALUATION_SUBSET, manifest)


def load_subset(path: str | Path | None = None) -> dict[str, Any]:
    from monitor_localization.utils.io import read_json

    return read_json(path or EVALUATION_SUBSET)


def subset_run_ids(path: str | Path | None = None) -> list[int]:
    """Just the run_ids, for downstream stages."""
    return list(load_subset(path)["run_ids"])
