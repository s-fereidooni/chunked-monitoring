"""Loading MALT from the Hugging Face Hub.

The `public` split is ~18.6 GB decompressed, almost all of it transcript text.
Label statistics and subset selection only need the `metadata` column, so those
read that column alone via parquet projection (a few MB) instead of pulling the
whole split. Full transcripts are materialised only for explicitly requested
run_ids.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import pandas as pd
import pyarrow.parquet as pq

from monitor_localization.config import DatasetConfig
from monitor_localization.dataset.schema import Message, Transcript
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)

# Which sample within a run to treat as the transcript. MALT stores `samples` as
# the sequence of API calls in the agent loop; each sample's `input` is the whole
# conversation so far, so the widest input plus its completion is the full run.
FlattenStrategy = Literal["longest_input", "concat_all"]

_GATED_HELP = (
    "MALT is a gated dataset. Distinguish the two failures by status code:\n"
    "\n"
    "  401 — no token, or an invalid one. Set HF_TOKEN in .env, or run\n"
    "        `huggingface-cli login`.\n"
    "  403 — the token is fine but this account is not on the authorized list.\n"
    "        Request access at\n"
    "        https://huggingface.co/datasets/metr-evals/malt-public\n"
    "        and wait for it to be granted; a valid token alone is not enough.\n"
)


class GatedDatasetError(RuntimeError):
    """Raised when MALT cannot be read because access has not been granted."""


def hf_token() -> str | None:
    """Token from the environment, if set. `None` falls back to the CLI login."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _data_prefix(cfg: DatasetConfig) -> str:
    """Directory within the repo holding a config's shards."""
    return "data" if cfg.hf_config == "default" else cfg.hf_config


def _filesystem():
    from huggingface_hub import HfFileSystem

    return HfFileSystem(token=hf_token())


def shard_paths(cfg: DatasetConfig) -> list[str]:
    """Resolve the parquet shard paths for a config, sorted for determinism."""
    fs = _filesystem()
    pattern = f"datasets/{cfg.hf_repo}/{_data_prefix(cfg)}/{cfg.split}-*.parquet"
    try:
        paths = sorted(fs.glob(pattern))
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable guidance
        raise GatedDatasetError(f"Could not list {cfg.hf_repo}.\n\n{_GATED_HELP}") from exc
    if not paths:
        raise FileNotFoundError(
            f"No shards matched {pattern!r}. Check hf_config={cfg.hf_config!r} "
            f"and split={cfg.split!r}."
        )
    return paths


def metadata_table(cfg: DatasetConfig | None = None) -> pd.DataFrame:
    """Load every run's metadata (no transcript text) as a DataFrame.

    Adds the derived columns the subset sampler and reports need: `is_positive`,
    `n_labels`, and `is_prompted`.
    """
    from monitor_localization.dataset.schema import BENIGN_LABEL

    cfg = cfg or DatasetConfig()
    fs = _filesystem()
    paths = shard_paths(cfg)
    logger.info("Reading metadata from %d shard(s) of %s", len(paths), cfg.hf_repo)

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            table = pq.read_table(path, columns=["metadata"], filesystem=fs)
        except Exception as exc:  # noqa: BLE001
            raise GatedDatasetError(f"Could not read {path}.\n\n{_GATED_HELP}") from exc
        frames.append(pd.json_normalize(table.column("metadata").to_pylist()))

    df = pd.concat(frames, ignore_index=True)
    df["labels"] = df["labels"].apply(lambda xs: list(xs) if xs is not None else [])
    df["is_positive"] = df["labels"].apply(
        lambda labels: any(label != BENIGN_LABEL for label in labels)
    )
    df["n_labels"] = df["labels"].apply(len)
    df["is_prompted"] = df["run_source"] == "prompted"
    logger.info("Loaded metadata for %d runs", len(df))
    return df


def flatten_samples(
    samples: Sequence[dict[str, Any]],
    strategy: FlattenStrategy = "longest_input",
) -> list[Message]:
    """Collapse a run's `samples` into one linear message list.

    MALT stores `samples` as the successive API calls of an agent loop. Each
    sample's `input` is the full conversation sent to the model at that step, and
    `output` is a list of completion choices, each itself a list of messages.

    `longest_input` (default) takes the sample with the widest input — the last
    step of the loop — and appends its first completion. That reconstructs the
    run without duplicating the prefix that every earlier sample also contains.

    `concat_all` concatenates every sample's completion onto the first input,
    which is only correct for runs whose inputs grow strictly monotonically.
    """
    if not samples:
        return []

    def first_choice(sample: dict[str, Any]) -> list[dict[str, Any]]:
        output = sample.get("output") or []
        if not output:
            return []
        first = output[0]
        # `output` is list[list[message]]; tolerate a flat list[message] too.
        if isinstance(first, dict):
            return [m for m in output if isinstance(m, dict)]
        return list(first or [])

    if strategy == "longest_input":
        widest = max(samples, key=lambda s: len(s.get("input") or []))
        raw = list(widest.get("input") or []) + first_choice(widest)
    elif strategy == "concat_all":
        ordered = sorted(samples, key=lambda s: len(s.get("input") or []))
        raw = list(ordered[0].get("input") or [])
        for sample in ordered:
            raw += first_choice(sample)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown flatten strategy {strategy!r}")

    return [Message.from_raw(m) for m in raw if isinstance(m, dict)]


def row_to_transcript(
    row: dict[str, Any],
    strategy: FlattenStrategy = "longest_input",
) -> Transcript:
    """Build a `Transcript` from one raw parquet row."""
    meta = row.get("metadata") or {}
    labels = meta.get("labels")
    return Transcript(
        run_id=meta.get("run_id", -1),
        task_id=meta.get("task_id") or "",
        model=meta.get("model") or "",
        labels=list(labels) if labels is not None else [],
        manually_reviewed=bool(meta.get("manually_reviewed")),
        run_source=meta.get("run_source") or "",
        has_chain_of_thought=bool(meta.get("has_chain_of_thought")),
        messages=flatten_samples(row.get("samples") or [], strategy=strategy),
    )


def iter_transcripts(
    run_ids: Sequence[int] | None = None,
    cfg: DatasetConfig | None = None,
    strategy: FlattenStrategy = "longest_input",
) -> Iterator[Transcript]:
    """Yield transcripts, optionally restricted to `run_ids`.

    Shards are read one row group at a time so memory stays bounded even though
    the split does not fit in RAM. Iteration stops early once every requested
    run_id has been found.
    """
    cfg = cfg or DatasetConfig()
    fs = _filesystem()
    wanted = set(run_ids) if run_ids is not None else None
    remaining = set(wanted) if wanted is not None else None

    for path in shard_paths(cfg):
        if remaining is not None and not remaining:
            break
        parquet_file = pq.ParquetFile(path, filesystem=fs)
        for group_index in range(parquet_file.num_row_groups):
            if remaining is not None and not remaining:
                break
            # Read metadata first so a row group with no wanted run is skipped
            # without materialising any transcript text.
            if remaining is not None:
                meta_only = parquet_file.read_row_group(group_index, columns=["metadata"])
                ids = {m.get("run_id") for m in meta_only.column("metadata").to_pylist()}
                if not (ids & remaining):
                    continue
            group = parquet_file.read_row_group(group_index)
            for row in group.to_pylist():
                run_id = (row.get("metadata") or {}).get("run_id")
                if remaining is not None:
                    if run_id not in remaining:
                        continue
                    remaining.discard(run_id)
                yield row_to_transcript(row, strategy=strategy)

    if remaining:
        logger.warning(
            "%d requested run_id(s) not found, e.g. %s",
            len(remaining),
            sorted(remaining)[:5],
        )


def load_transcripts(
    run_ids: Sequence[int],
    cfg: DatasetConfig | None = None,
    strategy: FlattenStrategy = "longest_input",
) -> list[Transcript]:
    """Materialise transcripts for `run_ids`, preserving the requested order."""
    by_id = {t.run_id: t for t in iter_transcripts(run_ids, cfg=cfg, strategy=strategy)}
    return [by_id[rid] for rid in run_ids if rid in by_id]
