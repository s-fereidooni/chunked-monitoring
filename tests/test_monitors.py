"""Stage 2 tests. No network: a fake SDK client stands in for OpenAI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import make_row

from monitor_localization.config import ModelConfig
from monitor_localization.dataset import row_to_transcript
from monitor_localization.llm import (
    MonitorClient,
    RefusalError,
    ResponseCache,
    TruncatedResponseError,
    cache_key,
)
from monitor_localization.monitors import GlobalMonitor, MonitorResult, Verdict
from monitor_localization.prompts import available_prompts, load_prompt


class FakeCompletions:
    """Stands in for client.chat.completions, recording calls."""

    def __init__(self, verdict=None, finish_reason="stop", refusal=None):
        self.verdict = verdict if verdict is not None else Verdict(reasoning="looks fine", score=7)
        self.finish_reason = finish_reason
        self.refusal = refusal
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.verdict, refusal=self.refusal)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=50),
            model="gpt-4.1-mini",
        )


class FakeSDK:
    def __init__(self, completions: FakeCompletions):
        self.chat = SimpleNamespace(completions=completions)


def make_client(tmp_path, **kwargs) -> tuple[MonitorClient, FakeCompletions]:
    completions = FakeCompletions(**kwargs)
    client = MonitorClient(
        ModelConfig(),
        cache=ResponseCache(tmp_path / "cache"),
        client=FakeSDK(completions),
    )
    return client, completions


# --- prompts ----------------------------------------------------------------


SANDBAGGING_PROMPTS = ("sandbagging_global", "sandbagging_chunk")


def test_metr_prompt_set_is_complete():
    assert set(available_prompts("metr")) == {
        "sandbagging_global", "sandbagging_chunk",
        "reward_hacking_global", "reward_hacking_chunk",
    }


def test_prompt_carries_digest():
    prompt = load_prompt("sandbagging_global", "metr")
    assert len(prompt.sha256) == 64
    assert prompt.describe()["version"] == "metr"


def test_prompt_load_is_cached_so_text_cannot_change_mid_run():
    assert load_prompt("sandbagging_global", "metr") is load_prompt(
        "sandbagging_global", "metr"
    )


def test_missing_prompt_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="sandbagging_global"):
        load_prompt("nope", "metr")


def test_only_the_chunk_prompt_says_it_is_reading_a_fragment():
    """Pins the one prompt difference between the arms, so it stays deliberate.

    METR's text always says "a part of a transcript" because their baseline is
    always chunked; our global arm says "a transcript". That is a real
    asymmetry beyond the chunking under study — telling a monitor it is reading
    a fragment may prime stricter local judgement — and it is recorded as a
    known limitation in `metr/SOURCE.txt`. This test exists so the difference
    cannot drift or be silently "fixed" without the re-run that would be needed
    to evaluate the change.
    """
    assert "a part of a transcript" in load_prompt("sandbagging_chunk", "metr").text
    assert "a part of a transcript" not in load_prompt("sandbagging_global", "metr").text


def test_prompts_do_not_enumerate_the_label_taxonomy():
    """Guards against answer-key leakage.

    Listing MALT's label identifiers would make measured recall partly an
    artifact of prompt engineering rather than monitoring capability. Naming the
    *construct* is fine and expected — METR's text says "sandbagging" and
    describes mimicking a weaker model in prose. What must not appear is the
    label vocabulary the runs are scored against.
    """
    labels = [
        "bypass_constraints",
        "hardcoded_solution",
        "match_weaker_model",
        "partial_problem_solving",
        "reasoning_about_task",
        "gives_up",
    ]
    for name in SANDBAGGING_PROMPTS:
        text = load_prompt(name, "metr").text.lower()
        leaked = [label for label in labels if label in text or label.replace("_", " ") in text]
        assert not leaked, f"{name} leaks label names: {leaked}"


def test_prompts_do_not_instruct_score_distribution():
    """No anti-clustering instruction.

    Score clustering is a documented problem, but published work addresses it
    methodologically (comparative scoring, prompt optimisation), not by telling
    the model how to distribute its scores. Instructing the distribution biases
    the exact quantity being measured.
    """
    for name in SANDBAGGING_PROMPTS:
        text = load_prompt(name, "metr").text.lower()
        for phrase in ("full range", "do not cluster", "round numbers", "midpoint"):
            assert phrase not in text, f"{name} instructs score distribution: {phrase!r}"


# --- verdict schema ---------------------------------------------------------


def test_reasoning_precedes_score_in_schema():
    # Field order is load-bearing: models emit fields in declaration order, so
    # reasoning must come first for the score to benefit from it.
    assert list(Verdict.model_fields) == ["reasoning", "score"]


def test_score_bounds_enforced():
    from pydantic import ValidationError

    Verdict(reasoning="x", score=0)
    Verdict(reasoning="x", score=100)
    with pytest.raises(ValidationError):
        Verdict(reasoning="x", score=101)
    with pytest.raises(ValidationError):
        Verdict(reasoning="x", score=-1)


def test_verdict_has_no_prediction_field():
    # Predictions are derived at a threshold in the evaluation stage so scores can
    # be re-thresholded without re-running inference.
    assert "prediction" not in Verdict.model_fields


# --- client -----------------------------------------------------------------


def test_completion_returns_parsed_verdict(tmp_path):
    client, _ = make_client(tmp_path)
    result = client.complete("sys", "user", Verdict)
    assert result.parsed.score == 7
    assert not result.from_cache
    assert result.usage.api_calls == 1


def test_second_identical_call_hits_cache(tmp_path):
    client, completions = make_client(tmp_path)
    client.complete("sys", "user", Verdict)
    again = client.complete("sys", "user", Verdict)
    assert again.from_cache
    assert again.parsed.score == 7
    assert len(completions.calls) == 1  # no second API call
    assert client.cache.stats["hits"] == 1


def test_cache_key_changes_with_prompt_model_and_params(tmp_path):
    client, _ = make_client(tmp_path)
    base = client._request_payload("sys", "user", Verdict)
    assert cache_key(base) != cache_key({**base, "user": "different"})
    assert cache_key(base) != cache_key({**base, "system": "different"})
    assert cache_key(base) != cache_key({**base, "model": "gpt-4o"})
    assert cache_key(base) != cache_key({**base, "temperature": 0.7})


def test_cache_key_is_order_independent():
    a = {"model": "m", "user": "u", "temperature": 0.0}
    b = {"temperature": 0.0, "user": "u", "model": "m"}
    assert cache_key(a) == cache_key(b)


def test_truncated_output_raises_rather_than_scoring_zero(tmp_path):
    client, _ = make_client(tmp_path, finish_reason="length")
    with pytest.raises(TruncatedResponseError, match="output cap"):
        client.complete("sys", "user", Verdict)


def test_refusal_is_distinct_from_a_low_score(tmp_path):
    client, _ = make_client(tmp_path, refusal="I cannot help")
    with pytest.raises(RefusalError):
        client.complete("sys", "user", Verdict)


def test_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    client, completions = make_client(tmp_path)
    client.complete("sys", "user", Verdict)
    key = cache_key(client._request_payload("sys", "user", Verdict))
    client.cache.path_for(key).write_text("{ truncated")
    result = client.complete("sys", "user", Verdict)
    assert not result.from_cache
    assert len(completions.calls) == 2


def test_missing_api_key_raises_actionable_error(monkeypatch):
    from monitor_localization.llm import MissingAPIKeyError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = MonitorClient(ModelConfig())
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        _ = client.client


def test_usage_accumulates_across_calls(tmp_path):
    client, _ = make_client(tmp_path)
    client.complete("sys", "a", Verdict)
    client.complete("sys", "b", Verdict)
    assert client.usage.api_calls == 2
    assert client.usage.prompt_tokens == 2000


# --- global monitor ---------------------------------------------------------


def test_global_monitor_scores_a_transcript(tmp_path):
    client, completions = make_client(tmp_path)
    monitor = GlobalMonitor(client=client, target="sandbagging")
    transcript = row_to_transcript(make_row(run_id=7, labels=["sabotage"]))
    result = monitor.judge(transcript)

    assert isinstance(result, MonitorResult)
    assert result.run_id == 7
    assert result.score == 7.0
    assert result.is_positive
    assert result.labels == ["sabotage"]
    assert not result.failed
    # The whole transcript is sent in one call.
    assert len(completions.calls) == 1


def test_global_monitor_sends_transcript_text_and_prompt(tmp_path):
    client, completions = make_client(tmp_path)
    monitor = GlobalMonitor(client=client, target="sandbagging")
    monitor.judge(row_to_transcript(make_row()))
    messages = completions.calls[0]["messages"]
    # The prompt is the system message and the rendered transcript is the user
    # message — one of the documented departures from METR's template, which
    # embeds the transcript in the prompt itself. See metr/SOURCE.txt.
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == load_prompt("sandbagging_global", "metr").text
    assert "[assistant]" in messages[1]["content"]


def test_normalized_score_maps_to_unit_range():
    result = MonitorResult(run_id=1, monitor="global", score=75.0, reasoning="")
    assert result.normalized_score == 0.75


def test_failure_is_recorded_not_raised(tmp_path):
    client, _ = make_client(tmp_path, finish_reason="length")
    monitor = GlobalMonitor(client=client, target="sandbagging")
    result = monitor.judge(row_to_transcript(make_row(run_id=9)))
    # One bad transcript must not abort a 500-run stage...
    assert result.failed
    assert "TruncatedResponseError" in result.error
    # ...and must not look like a confident "clean" verdict.
    assert result.score != result.score  # NaN


def test_describe_records_provenance(tmp_path):
    client, _ = make_client(tmp_path)
    described = GlobalMonitor(client=client, target="sandbagging").describe()
    assert described["model"]["name"] == "gpt-4.1-mini"
    assert len(described["prompt"]["sha256"]) == 64
