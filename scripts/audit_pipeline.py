#!/usr/bin/env python
"""Data-level audit of the whole pipeline, from raw parquet to reported metric.

Written after five bugs reached results undetected while 189 unit tests passed:
transcript flattening dropped 98% of messages; a config flag never reached the
loader so a "successful" pass made 0 API calls; MALT's elicitation prompt was fed
to the monitor; a redaction regex mangled task content; and every assistant
message was emitted twice.

None of those were catchable by the test suite, because the fixtures encode the
same assumptions as the code — when the model of the data is wrong, the fixture
is wrong the same way and the test agrees with the bug. Metrics could not catch
them either: a monitor scoring a corrupted transcript still returns a plausible
number.

So every check here compares the pipeline against *the source data* rather than
against expectations, and runs on the real dataset rather than fixtures. All
checks are read-only and free.

    python scripts/audit_pipeline.py                  # full audit
    python scripts/audit_pipeline.py --limit 40       # quick pass
    python scripts/audit_pipeline.py --config stripped
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from monitor_localization.config import ExperimentConfig
from monitor_localization.dataset import subset_run_ids
from monitor_localization.dataset.loaders import (
    _filesystem,
    flatten_samples,
    shard_paths,
)
from monitor_localization.experiment import ExperimentRun
from monitor_localization.monitors import split_transcript
from monitor_localization.monitors.global_monitor import render_for_monitor
from monitor_localization.utils import load_env, setup_logging

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Audit:
    """Collects check results so one failure does not hide the others."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def record(self, status: str, name: str, detail: str) -> None:
        self.results.append((status, name, detail))
        mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
        print(f"[{mark}] {name}\n         {detail}")

    def check(self, ok: bool, name: str, detail: str, soft: bool = False) -> None:
        self.record(PASS if ok else (WARN if soft else FAIL), name, detail)

    def summary(self) -> int:
        counts = Counter(status for status, _, _ in self.results)
        print(f"\n{'=' * 78}")
        print(f"  {counts[PASS]} passed · {counts[WARN]} warnings · {counts[FAIL]} failed")
        if counts[FAIL]:
            print("\n  FAILURES:")
            for status, name, detail in self.results:
                if status == FAIL:
                    print(f"    - {name}: {detail}")
        print("=" * 78)
        return 1 if counts[FAIL] else 0


def _raw_rows(run_ids: set[int], cfg) -> dict[int, dict[str, Any]]:
    """Fetch raw parquet rows so reconstruction can be compared to its source."""
    import pyarrow.parquet as pq

    fs = _filesystem()
    found: dict[int, dict[str, Any]] = {}
    remaining = set(run_ids)
    for path in shard_paths(cfg):
        if not remaining:
            break
        parquet_file = pq.ParquetFile(path, filesystem=fs)
        for index in range(parquet_file.num_row_groups):
            if not remaining:
                break
            meta = parquet_file.read_row_group(index, columns=["metadata"])
            ids = {m.get("run_id") for m in meta.column("metadata").to_pylist()}
            if not (ids & remaining):
                continue
            for row in parquet_file.read_row_group(index).to_pylist():
                rid = (row.get("metadata") or {}).get("run_id")
                if rid in remaining:
                    found[rid] = row
                    remaining.discard(rid)
    return found


def _msg_key(message: dict[str, Any]) -> tuple[str, str, str, str]:
    call = message.get("function_call") or {}
    return (
        message.get("role") or "",
        message.get("content") or "",
        call.get("name") or "",
        call.get("arguments") or "",
    )


def _raw_expected(row: dict[str, Any]) -> Counter[tuple[str, str, str, str]]:
    """How many times each distinct message *should* appear in the transcript.

    Counted as a multiset, not a set. A set cannot see the difference between a
    message legitimately occurring three times and a deduplicator collapsing it
    to one — which is exactly the failure a content-keyed guard introduces.

    The expected multiplicity of a message is:
      (times it appears as a completion)  +  (times it appears as an input whose
      node_id was never emitted as a completion)
    Because in the growing-conversation shape each completion reappears as a
    later input, counting both would double it. So inputs are counted by
    *distinct node_id*, and completions by occurrence, minus the overlap.
    """
    completions: Counter[tuple[str, str, str, str]] = Counter()
    by_node: dict[int, tuple[str, str, str, str]] = {}
    no_node: Counter[tuple[str, str, str, str]] = Counter()

    for sample in row.get("samples") or []:
        for message in sample.get("input") or []:
            if not isinstance(message, dict):
                continue
            node = (message.get("metadata") or {}).get("node_id")
            if node is None:
                no_node[_msg_key(message)] += 1
            else:
                by_node[node] = _msg_key(message)
        output = sample.get("output") or []
        first = output[0] if output else []
        if isinstance(first, dict):
            first = output
        for message in first or []:
            if isinstance(message, dict):
                completions[_msg_key(message)] += 1

    inputs = Counter(by_node.values()) + no_node
    # An input that echoes a completion is the same node; count it once.
    expected = Counter()
    for key in set(inputs) | set(completions):
        expected[key] = max(inputs[key], completions[key])
    return expected


def audit_reconstruction(audit: Audit, rows: dict[int, dict[str, Any]]) -> None:
    """The check that would have caught both flattening bugs.

    Reconstruction must be a bijection with the raw row's distinct messages:
    nothing dropped, nothing invented, nothing repeated.
    """
    dropped, invented, undercount, overcount = [], [], [], []
    for run_id, row in rows.items():
        expected = _raw_expected(row)
        got = Counter(_msg_key({"role": m.role, "content": m.content,
                                "function_call": m.function_call}) for m in
                      flatten_samples(row.get("samples") or []))
        missing_kinds = set(expected) - set(got)
        extra_kinds = set(got) - set(expected)
        if missing_kinds:
            dropped.append((run_id, len(missing_kinds), len(expected)))
        if extra_kinds:
            invented.append((run_id, len(extra_kinds)))
        # Multiplicity, the part a set-based check cannot see.
        short = sum(max(expected[k] - got[k], 0) for k in expected)
        long = sum(max(got[k] - expected[k], 0) for k in got)
        if short:
            undercount.append((run_id, short, sum(expected.values())))
        if long:
            overcount.append((run_id, long, sum(got.values())))

    audit.check(
        not dropped, "reconstruction drops no kind of message",
        f"{len(dropped)}/{len(rows)} runs lose content entirely",
    )
    audit.check(
        not invented, "reconstruction invents no message",
        f"{len(invented)}/{len(rows)} runs contain content absent from the raw row",
    )
    audit.check(
        not undercount, "repeated messages are kept, not collapsed",
        f"{len(undercount)}/{len(rows)} runs have fewer occurrences than the source"
        + (f", worst {max(undercount, key=lambda x: x[1])[1]} missing of "
           f"{max(undercount, key=lambda x: x[1])[2]}" if undercount else ""),
    )
    audit.check(
        not overcount, "no message emitted more often than the source has it",
        f"{len(overcount)}/{len(rows)} runs over-emit"
        + (f", worst {max(overcount, key=lambda x: x[1])[1]} extra of "
           f"{max(overcount, key=lambda x: x[1])[2]}" if overcount else ""),
    )


def audit_transcripts(audit: Audit, transcripts: list) -> None:
    empty = [t.run_id for t in transcripts if not t.messages]
    audit.check(not empty, "no empty transcript", f"{len(empty)} transcripts have no messages")

    blank = sum(1 for t in transcripts for m in t.messages if not (m.content or "").strip())
    total = sum(len(t.messages) for t in transcripts)
    audit.check(
        blank == 0, "no blank message content",
        f"{blank}/{total} messages are empty or whitespace", soft=True,
    )

    roles = Counter(m.role for t in transcripts for m in t.messages)
    audit.check(
        set(roles) <= {"system", "user", "assistant", "function", "tool"},
        "message roles are recognised", f"roles seen: {dict(roles)}",
    )


def audit_chunking(audit: Audit, transcripts: list, cfg) -> None:
    """Chunks must tile the transcript exactly once when overlap is 0."""
    bad_cover, bad_context = [], []
    for t in transcripts:
        if not t.messages:
            continue
        chunks = split_transcript(t, cfg.chunk)
        covered = [i for c in chunks for i in range(c.start, c.end)]
        if cfg.chunk.overlap == 0 and covered != list(range(len(t.messages))):
            bad_cover.append(t.run_id)
        # Real check: no message may appear twice inside one rendered chunk.
        # (The former version tested `chunk.start < len(chunk.context)`, which
        # chunking.py guarantees false by construction — a tautology, not a
        # check.)
        for chunk in chunks:
            if not chunk.context:
                continue
            context_rendered = {m.render() for m in chunk.context}
            body_rendered = {m.render() for m in chunk.messages}
            if context_rendered & body_rendered:
                bad_context.append(t.run_id)
                break
    audit.check(
        not bad_cover, "chunks tile the transcript exactly once",
        f"{len(bad_cover)} transcripts have gaps or overlaps",
    )
    audit.check(
        not bad_context, "repeated context never duplicates a chunk's own messages",
        f"{len(bad_context)} transcripts double-count their opening messages",
    )


def audit_monitor_input(audit: Audit, transcripts: list, cfg) -> None:
    """What actually reaches the monitor, for the preprocessing the config claims."""
    stripped = cfg.dataset.strip_system_prompt
    leading_system = [t.run_id for t in transcripts
                      if t.messages and t.messages[0].role == "system"]
    if stripped:
        audit.check(
            not leading_system, "no transcript still opens with a system message",
            f"{len(leading_system)} transcripts retain a leading system message",
        )

    # The rendered form is what the monitor sees; check it, not the transcript.
    missing_body = [t.run_id for t in transcripts
                    if t.messages and "<transcript_chunk>" not in render_for_monitor(t)]
    audit.check(
        not missing_body, "rendered input has METR's transcript block",
        f"{len(missing_body)} renders lack <transcript_chunk>",
    )

    # No run metadata may be *added* by the renderer. Presence alone is not a
    # leak: agents inspect their own VM, so tool output legitimately contains
    # TASK_ID=... from an env dump and the model name from settings.json. Only a
    # higher count in the render than in the message bodies means we added it.
    leaked, in_data = [], []
    for t in transcripts[:60]:
        rendered = render_for_monitor(t)
        body = "\n".join(m.content or "" for m in t.messages)
        for kind, value in (("task_id", t.task_id), ("model", t.model)):
            if not value:
                continue
            # The opening messages appear twice in METR's layout (context block
            # plus body), so a value in them is counted twice legitimately.
            head = "\n".join(m.content or "" for m in t.messages[:5])
            allowed = body.count(value) + head.count(value)
            if rendered.count(value) > allowed:
                leaked.append((kind, t.run_id))
            elif value in body:
                in_data.append((kind, t.run_id))
    audit.check(
        not leaked, "renderer adds no run metadata",
        f"{len(leaked)} values appear in the render beyond their occurrences in the "
        f"transcript",
    )
    audit.check(
        not in_data, "no run metadata present in transcript content",
        f"{len(in_data)} runs contain task_id/model in tool output (agents inspecting "
        f"their own VM) — a dataset property, not a rendering bug",
        soft=True,
    )


def audit_results(audit: Audit, cfg, run_ids: list[int]) -> None:
    """Results files: identity, completeness, and no cross-config contamination."""
    for stage in ("global", "chunk"):
        path = ExperimentRun(cfg, stage=stage).results_path
        if not path.is_file():
            audit.record(WARN, f"{stage} results present", f"{path} does not exist")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [r["run_id"] for r in rows]
        audit.check(
            len(ids) == len(set(ids)), f"{stage}: no duplicate run_ids",
            f"{len(ids)} rows, {len(set(ids))} unique",
        )
        audit.check(
            set(ids) <= set(run_ids), f"{stage}: every scored run is in the subset",
            f"{len(set(ids) - set(run_ids))} rows outside the evaluation subset",
        )
        failed = [r for r in rows if r.get("error")]
        audit.check(
            not failed, f"{stage}: no failed rows", f"{len(failed)} rows carry an error",
            soft=True,
        )
        # A pass that served entirely from cache did no work — the bug that made
        # a no-op look like a successful experiment.
        manifest = ExperimentRun(cfg, stage=stage).manifest_path
        if manifest.is_file():
            usage = (json.loads(manifest.read_text()).get("usage") or {})
            calls = usage.get("api_calls")
            audit.check(
                calls is None or calls > 0 or len(rows) == 0,
                f"{stage}: pass made real API calls",
                f"api_calls={calls}, cached_calls={usage.get('cached_calls')}",
                soft=True,
            )


def audit_metrics(audit: Audit, cfg) -> None:
    """Recompute the headline independently and compare to the reported file."""
    path = ExperimentRun(cfg, stage="chunk").results_path.parent / "comparison.json"
    if not path.is_file():
        audit.record(WARN, "comparison.json present", "run scripts/compare.py first")
        return
    reported = json.loads(path.read_text())

    rows = {}
    for stage in ("global", "chunk"):
        p = ExperimentRun(cfg, stage=stage).results_path
        rows[stage] = {json.loads(line)["run_id"]: json.loads(line)
                       for line in p.read_text().splitlines() if line.strip()}
    shared = set(rows["global"]) & set(rows["chunk"])
    positives = [r for r in shared if rows["global"][r].get("is_positive")]
    recovered = sum(1 for r in positives
                    if rows["global"][r]["score"] < 50 <= rows["chunk"][r]["score"])
    lost = sum(1 for r in positives
               if rows["chunk"][r]["score"] < 50 <= rows["global"][r]["score"])
    m = reported.get("mcnemar", {})
    audit.check(
        m.get("recovered") == recovered and m.get("lost") == lost,
        "reported McNemar counts match a fresh recount",
        f"reported {m.get('recovered')}/{m.get('lost')}, recomputed {recovered}/{lost}",
    )
    audit.check(
        reported.get("n_paired") == len(shared),
        "reported pair count matches the results files",
        f"reported {reported.get('n_paired')}, recomputed {len(shared)}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="fixed")
    parser.add_argument("--limit", type=int, default=None,
                        help="audit only the first N runs (round-trip is the slow part)")
    args = parser.parse_args()

    load_env()
    setup_logging()
    cfg = ExperimentConfig.from_yaml(args.config)
    run_ids = subset_run_ids()
    sample = run_ids[: args.limit] if args.limit else run_ids

    audit = Audit()
    print(f"\n{'=' * 78}\nPipeline audit — config={args.config}, {len(sample)} runs\n{'=' * 78}\n")

    print("-- reconstruction vs raw parquet --")
    rows = _raw_rows(set(sample), cfg.dataset)
    audit.check(
        len(rows) == len(sample), "every requested run was found in the dataset",
        f"{len(rows)}/{len(sample)} located",
    )
    audit_reconstruction(audit, rows)

    print("\n-- transcripts as loaded --")
    from monitor_localization.dataset import load_transcripts

    transcripts = load_transcripts(sample, cfg=cfg.dataset)
    audit_transcripts(audit, transcripts)

    print("\n-- chunking --")
    audit_chunking(audit, transcripts, cfg)

    print("\n-- monitor input --")
    audit_monitor_input(audit, transcripts, cfg)

    print("\n-- results files --")
    audit_results(audit, cfg, run_ids)

    print("\n-- reported metrics --")
    audit_metrics(audit, cfg)

    return audit.summary()


if __name__ == "__main__":
    raise SystemExit(main())
