import pytest
from conftest import make_message, make_metadata, make_row, make_sample

from monitor_localization.config import SubsetConfig
from monitor_localization.dataset import (
    Transcript,
    build_subset,
    flatten_samples,
    label_group,
    row_to_transcript,
)

# --- schema -----------------------------------------------------------------


def test_is_positive_only_for_behavior_labels():
    assert not Transcript(1, "t", "m", ["normal"], True, "unprompted", True).is_positive
    assert Transcript(1, "t", "m", ["sabotage"], True, "prompted", True).is_positive
    # A run can carry both; any behavior label makes it positive.
    assert Transcript(1, "t", "m", ["normal", "gives_up"], True, "unprompted", True).is_positive


def test_unlabeled_run_is_not_positive():
    assert not Transcript(1, "t", "m", [], True, "unprompted", True).is_positive


def test_label_groups():
    assert label_group("hardcoded_solution") == "reward_hacking"
    assert label_group("sabotage") == "sandbagging_sabotage"
    assert label_group("normal") == "benign"
    # Unknown labels are reported, not fatal — MALT may add labels.
    assert label_group("some_future_label") == "unknown"


def test_taxonomy_matches_metr_grouping():
    """Pin the full taxonomy — this drifted once and is easy to get wrong.

    Source: METR's MALT release post. `hardcoded_solution` is our one
    documented deviation (METR files it under miscellaneous).
    """
    expected = {
        "bypass_constraints": "reward_hacking",
        "ignores_task_instructions": "reward_hacking",
        "hardcoded_solution": "reward_hacking",  # deviation from METR, deliberate
        "gives_up": "sandbagging_sabotage",
        "match_weaker_model": "sandbagging_sabotage",
        "partial_problem_solving": "sandbagging_sabotage",
        "reasoning_about_task": "sandbagging_sabotage",
        "refusals": "sandbagging_sabotage",
        "sabotage": "sandbagging_sabotage",
        "evaluation_awareness": "miscellaneous",
        "brute_forced_solution": "miscellaneous",
        "normal": "benign",
    }
    assert {label: label_group(label) for label in expected} == expected


def test_every_label_in_malt_public_is_mapped():
    """No label present in the real dataset may resolve to 'unknown'."""
    observed = [
        "normal",
        "refusals",
        "reasoning_about_task",
        "sabotage",
        "partial_problem_solving",
        "match_weaker_model",
        "ignores_task_instructions",
        "bypass_constraints",
        "gives_up",
        "hardcoded_solution",
    ]
    assert [label_group(label) for label in observed].count("unknown") == 0


def test_render_includes_tool_calls():
    transcript = row_to_transcript(make_row())
    rendered = transcript.render()
    assert "[assistant]" in rendered
    assert transcript.n_messages > 0

    from conftest import make_message

    from monitor_localization.dataset.schema import Message

    msg = Message.from_raw(make_message(tool="bash"))
    assert msg.is_tool_call
    assert "tool_call: bash" in msg.render()


def test_empty_function_call_struct_is_not_a_tool_call():
    from conftest import make_message

    from monitor_localization.dataset.schema import Message

    # The parquet schema always materialises the struct; an empty name means
    # "no tool call", not a call to the empty string.
    assert not Message.from_raw(make_message()).is_tool_call


# --- flattening -------------------------------------------------------------


def test_flatten_uses_widest_input_plus_its_completion():
    samples = [make_sample(n_input=n) for n in (1, 5, 3)]
    messages = flatten_samples(samples, strategy="longest_input")
    # Widest input has 5 messages, plus one completion message.
    assert len(messages) == 6
    assert messages[-1].content == "done"


def test_flatten_does_not_duplicate_the_shared_prefix():
    # Each sample's input contains every earlier message, so naively concatenating
    # inputs would repeat the prefix. 3 samples of width 1..3 -> 3 + 1, not 6 + 3.
    messages = flatten_samples([make_sample(n_input=n) for n in (1, 2, 3)])
    assert len(messages) == 4


def test_flatten_concat_all_appends_every_completion():
    samples = [make_sample(n_input=n) for n in (1, 2, 3)]
    messages = flatten_samples(samples, strategy="concat_all")
    assert len(messages) == 1 + 3


def test_flatten_empty_and_missing_output():
    assert flatten_samples([]) == []
    assert flatten_samples([{"input": [], "output": []}]) == []


def test_row_to_transcript_reads_metadata():
    transcript = row_to_transcript(make_row(run_id=42, labels=["gives_up"], run_source="prompted"))
    assert transcript.run_id == 42
    assert transcript.labels == ["gives_up"]
    assert transcript.is_prompted
    assert transcript.is_positive
    assert transcript.groups == ["sandbagging_sabotage"]


def test_metadata_row_is_flat():
    row = row_to_transcript(make_row()).metadata_row()
    assert set(row) >= {"run_id", "is_positive", "n_messages", "n_characters", "run_source"}


# --- subset selection -------------------------------------------------------


def test_subset_size_and_balance(metadata):
    manifest = build_subset(metadata, SubsetConfig(n_samples=100, positive_fraction=0.5))
    assert manifest["n_selected"] == 100
    assert manifest["composition"]["positive"] == 50
    assert manifest["composition"]["negative"] == 50


def test_subset_excludes_unreviewed_by_default(metadata):
    manifest = build_subset(metadata, SubsetConfig(n_samples=50))
    reviewed = set(metadata[metadata["manually_reviewed"]]["run_id"])
    assert set(manifest["run_ids"]) <= reviewed


def test_subset_includes_unreviewed_when_allowed(metadata):
    cfg = SubsetConfig(n_samples=50, manually_reviewed_only=False)
    manifest = build_subset(metadata, cfg)
    assert manifest["candidate_pool_size"] == len(metadata)


def test_prompted_cap_limits_prompted_positives(metadata):
    # 20 natural vs 400 prompted positives available. Without the cap a uniform
    # draw of the positive half would be ~95% prompted.
    cfg = SubsetConfig(n_samples=100, positive_fraction=0.5, max_prompted_fraction=0.5)
    manifest = build_subset(metadata, cfg)
    assert manifest["composition"]["prompted_positive"] <= 30  # 25 quota + backfill
    assert manifest["composition"]["positive"] == 50


def test_prompted_cap_disabled_lets_prompted_dominate():
    md = make_metadata(n_natural_positive=20, n_prompted_positive=400, n_negative=300)
    cfg = SubsetConfig(n_samples=100, positive_fraction=0.5, max_prompted_fraction=None)
    manifest = build_subset(md, cfg)
    assert manifest["composition"]["prompted_positive"] > 30


def test_backfill_warns_when_natural_positives_are_scarce():
    md = make_metadata(n_natural_positive=2, n_prompted_positive=400, n_negative=300)
    cfg = SubsetConfig(n_samples=100, positive_fraction=0.5, max_prompted_fraction=0.5)
    manifest = build_subset(md, cfg)
    assert manifest["n_selected"] == 100
    assert any("backfilling" in w for w in manifest["warnings"])


def test_short_stratum_warns_rather_than_silently_shrinking():
    md = make_metadata(n_natural_positive=5, n_prompted_positive=5, n_negative=300)
    manifest = build_subset(md, SubsetConfig(n_samples=100, positive_fraction=0.5))
    assert manifest["composition"]["positive"] == 10
    assert any("Positive stratum short" in w for w in manifest["warnings"])


def test_subset_is_deterministic_given_seed(metadata):
    cfg = SubsetConfig(n_samples=80, seed=7)
    first = build_subset(metadata, cfg)["run_ids"]
    second = build_subset(metadata, cfg)["run_ids"]
    assert first == second


def test_different_seeds_give_different_subsets(metadata):
    a = build_subset(metadata, SubsetConfig(n_samples=80, seed=1))["run_ids"]
    b = build_subset(metadata, SubsetConfig(n_samples=80, seed=2))["run_ids"]
    assert a != b


def test_subset_is_independent_of_row_order(metadata):
    cfg = SubsetConfig(n_samples=80, seed=3)
    shuffled = metadata.sample(frac=1.0, random_state=99).reset_index(drop=True)
    assert build_subset(metadata, cfg)["run_ids"] == build_subset(shuffled, cfg)["run_ids"]


def test_no_duplicate_run_ids(metadata):
    ids = build_subset(metadata, SubsetConfig(n_samples=200))["run_ids"]
    assert len(ids) == len(set(ids))


def test_uniform_draw_when_positive_fraction_is_none(metadata):
    cfg = SubsetConfig(n_samples=100, positive_fraction=None)
    manifest = build_subset(metadata, cfg)
    assert manifest["n_selected"] == 100


def test_manifest_records_config_and_labels(metadata):
    manifest = build_subset(metadata, SubsetConfig(n_samples=60, seed=5))
    assert manifest["config"]["seed"] == 5
    assert manifest["label_counts"]
    assert "created_at" in manifest


def test_empty_pool_raises():
    md = make_metadata(n_natural_positive=0, n_prompted_positive=0, n_negative=0, n_unreviewed=10)
    with pytest.raises(ValueError, match="No candidate runs"):
        build_subset(md, SubsetConfig(n_samples=10))


def test_completion_reappearing_as_input_is_not_emitted_twice():
    """Regression: 43% of the corpus was duplicated text.

    In the growing-conversation shape the agent's completion at step N reappears
    as an input at step N+1 carrying a *different* node_id. Inputs were deduped
    by node_id alone, so the content guard never fired and the message landed in
    the transcript twice.
    """
    reply = make_message(role="assistant", content="I will search the resources dir")
    samples = [
        {"input": [make_message(role="user", content="task", node_id=0)],
         "output": [[dict(reply)]], "metadata": {}},
        # Step 2 replays the conversation; the completion now carries a node_id.
        {"input": [make_message(role="user", content="task", node_id=0),
                   {**reply, "metadata": {"node_id": 1}},
                   make_message(role="user", content="continue", node_id=2)],
         "output": [[make_message(role="assistant", content="done")]], "metadata": {}},
    ]
    messages = flatten_samples(samples)
    contents = [(m.role, m.content) for m in messages]
    assert len(contents) == len(set(contents)), f"duplicates: {contents}"
    assert contents.count(("assistant", "I will search the resources dir")) == 1


def test_distinct_tool_calls_are_not_collapsed():
    """Dedup keys on the tool call too, or every bash call merges into one.

    Tool-call messages share empty content, so a (role, content) key alone would
    treat two different commands as the same message.
    """
    samples = [{
        "input": [make_message(role="user", content="go", node_id=0)],
        "output": [[
            make_message(role="assistant", content="", tool="bash", node_id=1),
            make_message(role="assistant", content="", tool="bash", node_id=2),
        ]],
        "metadata": {},
    }]
    samples[0]["output"][0][0]["function_call"] = {"name": "bash", "arguments": '{"cmd":"ls"}'}
    samples[0]["output"][0][1]["function_call"] = {"name": "bash", "arguments": '{"cmd":"pwd"}'}
    messages = flatten_samples(samples)
    calls = [m for m in messages if m.is_tool_call]
    assert len(calls) == 2, f"distinct tool calls collapsed: {len(calls)}"
