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

# How to linearise a run's message DAG. See flatten_samples for why `dag` is the
# only strategy correct across both of MALT's row shapes.
FlattenStrategy = Literal["dag", "longest_input", "concat_all"]

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


def _first_choice(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """The sample's first completion, as a list of messages.

    `output` is list[list[message]] — a list of sampled choices, each a message
    list — but tolerate a flat list[message] too.
    """
    output = sample.get("output") or []
    if not output:
        return []
    first = output[0]
    if isinstance(first, dict):
        return [m for m in output if isinstance(m, dict)]
    return list(first or [])


def _node_id(raw: dict[str, Any]) -> int | None:
    return (raw.get("metadata") or {}).get("node_id")


def flatten_samples(
    samples: Sequence[dict[str, Any]],
    strategy: FlattenStrategy = "dag",
) -> list[Message]:
    """Collapse a run's `samples` into one linear message list.

    MALT rows hold a *DAG* of messages, exposed as a list of samples. Two shapes
    occur, and a strategy that handles only one silently discards most of the
    other:

    - **Growing conversation.** Successive agent-loop steps, each sample's
      `input` containing the whole conversation so far. Input messages repeat
      across samples and share `node_id`s.
    - **Independent calls.** Many samples each with a single-message `input`
      (e.g. 107 separate one-shot calls). Nothing repeats; every message is
      distinct.

    `dag` (default) reconstructs both: input messages are deduplicated by
    `node_id` and ordered by it, and each sample's completion is inserted
    directly after the input it answered. Output messages carry no `node_id`, so
    a completion already absorbed into a later sample's input is recognised by
    content and not emitted twice.

    The legacy strategies are kept for comparison and are *both wrong* on at
    least one shape. `longest_input` takes the widest single input plus its
    completion — on independent-call runs that is 2 messages out of 200+, and
    even on growing runs it misses sibling branches of the DAG. `concat_all`
    duplicates the shared prefix on growing runs.
    """
    if not samples:
        return []

    if strategy == "longest_input":
        widest = max(samples, key=lambda s: len(s.get("input") or []))
        raw = list(widest.get("input") or []) + _first_choice(widest)
        return [Message.from_raw(m) for m in raw if isinstance(m, dict)]

    if strategy == "concat_all":
        ordered = sorted(samples, key=lambda s: len(s.get("input") or []))
        raw = list(ordered[0].get("input") or [])
        for sample in ordered:
            raw += _first_choice(sample)
        return [Message.from_raw(m) for m in raw if isinstance(m, dict)]

    if strategy != "dag":  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown flatten strategy {strategy!r}")

    # --- DAG reconstruction ---
    # (sort_key, tier, sequence) keeps completions adjacent to their input.
    items: list[tuple[float, int, int, dict[str, Any]]] = []
    seen_nodes: set[int] = set()
    seen_content: set[tuple[str, str]] = set()
    fallback = 0

    for sample in samples:
        inputs = [m for m in (sample.get("input") or []) if isinstance(m, dict)]
        last_key: float | None = None
        for raw in inputs:
            node = _node_id(raw)
            if node is None:
                # No DAG id: keep positionally, after everything seen so far.
                fallback += 1
                key = float(fallback)
            else:
                key = float(node)
                if node in seen_nodes:
                    last_key = key
                    continue
                seen_nodes.add(node)
            last_key = key
            items.append((key, 0, 0, raw))
            seen_content.add((raw.get("role") or "", raw.get("content") or ""))

        anchor = last_key if last_key is not None else float(fallback)
        for offset, raw in enumerate(_first_choice(sample)):
            if not isinstance(raw, dict):
                continue
            signature = (raw.get("role") or "", raw.get("content") or "")
            # A completion that reappears as a later sample's input would
            # otherwise be emitted twice.
            if signature in seen_content:
                continue
            seen_content.add(signature)
            items.append((anchor, 1, offset, raw))

    items.sort(key=lambda item: (item[0], item[1], item[2]))
    return [Message.from_raw(raw) for _, _, _, raw in items]


def row_to_transcript(
    row: dict[str, Any],
    strategy: FlattenStrategy = "dag",
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
    strategy: FlattenStrategy = "dag",
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
    strategy: FlattenStrategy = "dag",
) -> list[Transcript]:
    """Materialise transcripts for `run_ids`, preserving the requested order."""
    by_id = {t.run_id: t for t in iter_transcripts(run_ids, cfg=cfg, strategy=strategy)}
    return [by_id[rid] for rid in run_ids if rid in by_id]
