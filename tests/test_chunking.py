"""Stage 4 tests: chunking, aggregation, and the chunk monitor."""

from __future__ import annotations

import math

import pytest
from conftest import make_row
from test_monitors import make_client

from monitor_localization.config import AggregationConfig, ChunkConfig
from monitor_localization.dataset.schema import Message, Transcript
from monitor_localization.monitors import (
    ChunkMonitor,
    aggregate,
    aggregate_with_evidence,
    chunk_count,
    split_transcript,
)


def transcript(n_messages: int, run_id: int = 1) -> Transcript:
    return Transcript(
        run_id=run_id,
        task_id="t/a",
        model="m",
        labels=["sabotage"],
        manually_reviewed=True,
        run_source="unprompted",
        has_chain_of_thought=False,
        messages=[Message(role="assistant", content=f"m{i}") for i in range(n_messages)],
    )


# --- windowing --------------------------------------------------------------


def test_chunks_tile_the_transcript_without_gaps():
    chunks = split_transcript(transcript(25), ChunkConfig(size=10, overlap=0))
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (10, 20), (20, 25)]
    # Every message appears exactly once when there is no overlap.
    covered = [i for c in chunks for i in range(c.start, c.end)]
    assert covered == list(range(25))


def test_final_chunk_is_clamped_not_padded():
    chunks = split_transcript(transcript(23), ChunkConfig(size=10, overlap=0))
    assert chunks[-1].n_messages == 3
    assert chunks[-1].end == 23


def test_overlap_shifts_by_stride():
    chunks = split_transcript(transcript(20), ChunkConfig(size=10, overlap=4))
    # stride = size - overlap = 6; the window starting at 12 reaches the end,
    # so no further window is emitted.
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (6, 16), (12, 20)]


def test_overlap_repeats_the_shared_messages():
    chunks = split_transcript(transcript(20), ChunkConfig(size=10, overlap=4))
    shared = set(range(chunks[0].start, chunks[0].end)) & set(
        range(chunks[1].start, chunks[1].end)
    )
    assert len(shared) == 4


def test_exact_multiple_produces_no_empty_trailing_chunk():
    chunks = split_transcript(transcript(20), ChunkConfig(size=10, overlap=0))
    assert len(chunks) == 2
    assert all(c.n_messages > 0 for c in chunks)


def test_transcript_shorter_than_chunk_size_is_one_chunk():
    chunks = split_transcript(transcript(3), ChunkConfig(size=10))
    assert len(chunks) == 1
    assert chunks[0].n_messages == 3


def test_empty_transcript_yields_no_chunks():
    assert split_transcript(transcript(0)) == []


def test_chunk_count_matches_split():
    cfg = ChunkConfig(size=7, overlap=2)
    assert chunk_count(transcript(30), cfg) == len(split_transcript(transcript(30), cfg))


# --- repeated context -------------------------------------------------------


def test_context_is_attached_to_later_chunks_only():
    chunks = split_transcript(transcript(30), ChunkConfig(size=10, context_messages=5))
    # Chunk 0 already contains messages 0-9, so repeating 0-4 would duplicate.
    assert chunks[0].context == []
    assert len(chunks[1].context) == 5
    assert len(chunks[2].context) == 5


def test_context_can_be_disabled():
    chunks = split_transcript(transcript(30), ChunkConfig(size=10, context_messages=0))
    assert all(c.context == [] for c in chunks)


def test_render_separates_context_from_excerpt():
    chunks = split_transcript(transcript(30), ChunkConfig(size=10, context_messages=5))
    rendered = chunks[1].render()
    # METR's layout, identical in structure to the global condition so the only
    # difference between arms is what fills <transcript_chunk>.
    assert "<first_few_messages>" in rendered
    assert "<transcript_chunk>" in rendered
    assert "m0" in rendered   # repeated opening context
    assert "m10" in rendered  # the excerpt itself
    # No message indices: that would be a position cue the global monitor lacks.
    assert "messages 10-19" not in rendered


def test_render_without_context_has_no_context_block():
    chunks = split_transcript(transcript(8), ChunkConfig(size=10, context_messages=5))
    assert chunks[0].context == []


# --- character chunking -----------------------------------------------------


def test_character_chunking_respects_budget_and_never_splits_a_message():
    msgs = [Message(role="a", content="x" * 100) for _ in range(10)]
    t = transcript(0)
    t.messages = msgs
    chunks = split_transcript(t, ChunkConfig(size=250, overlap=0, unit="characters"))
    assert all(c.n_messages >= 1 for c in chunks)
    # 250-char budget fits two 100-char messages per chunk.
    assert chunks[0].n_messages == 2
    covered = [i for c in chunks for i in range(c.start, c.end)]
    assert covered == list(range(10))


def test_oversized_message_becomes_its_own_chunk():
    t = transcript(0)
    t.messages = [Message(role="a", content="x" * 5000), Message(role="a", content="y")]
    chunks = split_transcript(t, ChunkConfig(size=100, overlap=0, unit="characters"))
    assert chunks[0].n_messages == 1  # not dropped, not split


# --- aggregation ------------------------------------------------------------


def test_max_takes_the_single_most_suspicious_chunk():
    assert aggregate([5, 90, 10], AggregationConfig(method="max")) == 90


def test_mean_dilutes_a_single_spike():
    # The core asymmetry: one damning chunk in twenty is obvious to max and
    # nearly invisible to mean. This is why aggregation encodes a prior about
    # whether evidence is local or diffuse.
    scores = [5] * 19 + [100]
    assert aggregate(scores, AggregationConfig(method="max")) == 100
    assert aggregate(scores, AggregationConfig(method="mean")) < 12


def test_topk_mean_interpolates_between_max_and_mean():
    scores = [10, 20, 90, 100]
    top1 = aggregate(scores, AggregationConfig(method="topk_mean", top_k=1))
    top2 = aggregate(scores, AggregationConfig(method="topk_mean", top_k=2))
    mean = aggregate(scores, AggregationConfig(method="mean"))
    assert top1 == 100
    assert mean < top2 < top1


def test_topk_larger_than_n_falls_back_to_mean():
    scores = [10, 20]
    assert aggregate(scores, AggregationConfig(method="topk_mean", top_k=99)) == 15


def test_noisy_or_accumulates_weak_evidence():
    # Many mildly suspicious chunks outweigh one moderately suspicious chunk.
    many_weak = aggregate([30] * 8, AggregationConfig(method="noisy_or"))
    one_mid = aggregate([60], AggregationConfig(method="noisy_or"))
    assert many_weak > one_mid


def test_noisy_or_stays_in_range():
    assert 0 <= aggregate([99] * 50, AggregationConfig(method="noisy_or")) <= 100


def test_failed_chunks_are_dropped_not_scored_as_clean():
    # A NaN chunk must not drag the mean toward zero.
    scores = [90, float("nan")]
    assert aggregate(scores, AggregationConfig(method="mean")) == 90
    assert aggregate(scores, AggregationConfig(method="max")) == 90


def test_all_chunks_failed_gives_nan():
    assert math.isnan(aggregate([float("nan")] * 3))


def test_empty_scores_give_nan():
    assert math.isnan(aggregate([]))


def test_unknown_method_raises():
    cfg = AggregationConfig(method="max")
    object.__setattr__(cfg, "method", "nonsense")
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate([1.0], cfg)


def test_evidence_index_identifies_driving_chunk():
    score, index = aggregate_with_evidence([5, 90, 10], AggregationConfig(method="max"))
    assert (score, index) == (90, 1)


def test_evidence_index_is_none_for_mean():
    _, index = aggregate_with_evidence([5, 90], AggregationConfig(method="mean"))
    assert index is None


# --- chunk monitor ----------------------------------------------------------


def test_chunk_monitor_scores_every_chunk_and_aggregates(tmp_path):
    client, completions = make_client(tmp_path)
    monitor = ChunkMonitor(client=client, chunk=ChunkConfig(size=10, context_messages=5))
    result = monitor.judge(transcript(30, run_id=4))

    assert result.metadata["n_chunks"] == 3
    assert len(completions.calls) == 3      # one call per chunk
    assert len(result.chunk_verdicts) == 3
    assert result.score == 7.0              # fake client returns 7 for every chunk
    assert not result.failed


def test_chunk_monitor_retains_per_chunk_verdicts_for_reaggregation(tmp_path):
    client, _ = make_client(tmp_path)
    result = ChunkMonitor(client=client).judge(transcript(30))
    # Offline re-aggregation must be possible without re-running inference.
    scores = [v.score for v in result.chunk_verdicts]
    assert aggregate(scores, AggregationConfig(method="mean")) == 7.0
    assert all(v.end_message > v.start_message for v in result.chunk_verdicts)


def test_chunk_monitor_reasoning_points_at_the_driving_chunk(tmp_path):
    client, _ = make_client(tmp_path)
    result = ChunkMonitor(client=client).judge(transcript(30))
    assert result.reasoning.startswith("[chunk ")
    assert "messages" in result.reasoning


def test_chunk_monitor_survives_a_failing_chunk(tmp_path):
    client, _ = make_client(tmp_path, finish_reason="length")
    result = ChunkMonitor(client=client).judge(transcript(20))
    # Every chunk failed here, so the transcript is an error...
    assert result.failed
    assert "AllChunksFailed" in result.error
    # ...but the per-chunk records survive for diagnosis.
    assert len(result.chunk_verdicts) == 2


def test_chunk_monitor_handles_empty_transcript(tmp_path):
    client, _ = make_client(tmp_path)
    result = ChunkMonitor(client=client).judge(transcript(0))
    assert result.failed
    assert "EmptyTranscript" in result.error


def test_chunk_monitor_sends_the_chunk_prompt_not_the_global_one(tmp_path):
    client, completions = make_client(tmp_path)
    ChunkMonitor(client=client, prompt_version="v3", target="sandbagging").judge(transcript(12))
    system = completions.calls[0]["messages"][0]["content"].lower()
    assert "a part of a transcript" in system


def test_describe_records_chunk_and_aggregation_config(tmp_path):
    client, _ = make_client(tmp_path)
    described = ChunkMonitor(
        client=client,
        chunk=ChunkConfig(size=7, overlap=2),
        aggregation=AggregationConfig(method="mean"),
    ).describe()
    assert described["chunk"]["size"] == 7
    assert described["chunk"]["overlap"] == 2
    assert described["aggregation"]["method"] == "mean"


def test_chunk_monitor_reuses_cache_across_identical_chunks(tmp_path):
    # Overlapping windows re-send shared messages; identical rendered chunks must
    # hit the cache rather than paying twice.
    client, completions = make_client(tmp_path)
    monitor = ChunkMonitor(client=client, chunk=ChunkConfig(size=10, context_messages=0))
    monitor.judge(transcript(30, run_id=1))
    calls_after_first = len(completions.calls)
    monitor.judge(transcript(30, run_id=2))  # same content, different run_id
    assert len(completions.calls) == calls_after_first


def test_row_to_transcript_integration(tmp_path):
    from monitor_localization.dataset import row_to_transcript

    client, _ = make_client(tmp_path)
    result = ChunkMonitor(client=client).judge(row_to_transcript(make_row(run_id=99)))
    assert result.run_id == 99
    assert result.metadata["n_chunks"] >= 1
