"""Tests for length controls and paired architecture comparison."""

from __future__ import annotations

import pandas as pd

from monitor_localization.evaluation import (
    BOTH_MISSED,
    LOST,
    RECOVERED,
    add_length_columns,
    delta_vs_length,
    fpr_by_length_decile,
    length_by_label,
    mcnemar,
    pair_results,
    per_sample_table,
    score_vs_length,
    transition_counts,
)


def record(run_id, score, labels, chars=44_000, error=None):
    return {
        "run_id": run_id,
        "monitor": "global",
        "score": score,
        "reasoning": "r",
        "labels": labels,
        "is_positive": any(x != "normal" for x in labels),
        "error": error,
        "metadata": {
            "monitored_model": "m",
            "run_source": "prompted",
            "task_id": "t",
            "n_messages": 20,
            "n_characters": chars,
        },
    }


# --- length -----------------------------------------------------------------


def test_tokens_derived_from_characters():
    table = per_sample_table([record(1, 10, ["sabotage"], chars=44_000)])
    out = add_length_columns(table)
    assert abs(out.loc[0, "tokens"] - 10_000) < 1


def test_length_by_label_reports_within_label_spread():
    rows = [record(i, 0, ["sabotage"], chars=c) for i, c in enumerate([4_400, 440_000])]
    out = length_by_label(per_sample_table(rows))
    row = out[out["label"] == "sabotage"].iloc[0]
    # p90/p10 must show the internal variation, not collapse to 1.
    assert row["spread"] > 5


def test_score_vs_length_detects_positive_correlation():
    # Longer transcripts scored higher — the false-positive tendency we need to
    # be able to see on benign runs.
    rows = [record(i, i * 5, ["normal"], chars=10_000 * (i + 1)) for i in range(20)]
    out = score_vs_length(per_sample_table(rows))
    assert out.loc[out["label"] == "normal", "rho"].iloc[0] > 0.9


def test_score_vs_length_flags_uninterpretable_low_spread():
    # All the same length: a correlation here cannot separate length from label.
    rows = [record(i, i, ["sabotage"], chars=50_000) for i in range(20)]
    out = score_vs_length(per_sample_table(rows))
    if not out.empty:
        assert not bool(out.iloc[0]["interpretable"])


def test_score_vs_length_skips_tiny_labels():
    rows = [record(i, i, ["gives_up"], chars=10_000 * (i + 1)) for i in range(5)]
    assert score_vs_length(per_sample_table(rows)).empty


def test_fpr_by_length_decile_isolates_long_runs():
    benign = [record(i, 0, ["normal"], chars=1_000 * (i + 1)) for i in range(18)]
    # Two long benign runs get flagged.
    benign += [record(100 + i, 90, ["normal"], chars=900_000) for i in range(2)]
    out = fpr_by_length_decile(per_sample_table(benign), threshold=50)
    assert out["n_flagged"].sum() == 2
    # The flagged ones must land in the top decile, not be spread around.
    assert out.sort_values("length_decile").iloc[-1]["n_flagged"] == 2


def test_fpr_by_length_decile_ignores_positives():
    rows = [record(i, 90, ["sabotage"], chars=10_000) for i in range(15)]
    assert fpr_by_length_decile(per_sample_table(rows))["n_flagged"].sum() == 0


# --- paired comparison ------------------------------------------------------


def test_pair_results_computes_transitions():
    base = [record(1, 10, ["sabotage"]), record(2, 90, ["sabotage"]),
            record(3, 90, ["sabotage"]), record(4, 10, ["sabotage"])]
    var = [record(1, 90, ["sabotage"]), record(2, 10, ["sabotage"]),
           record(3, 95, ["sabotage"]), record(4, 5, ["sabotage"])]
    out = pair_results(base, var, threshold=50)
    got = dict(zip(out["run_id"], out["transition"], strict=True))
    assert got[1] == RECOVERED
    assert got[2] == LOST
    assert got[4] == BOTH_MISSED


def test_pair_results_drops_runs_that_failed_in_either_arm():
    # A failed call is missing data; scoring it would fabricate a recovery.
    base = [record(1, float("nan"), ["sabotage"], error="boom"), record(2, 10, ["sabotage"])]
    var = [record(1, 90, ["sabotage"]), record(2, 90, ["sabotage"])]
    out = pair_results(base, var, threshold=50)
    assert list(out["run_id"]) == [2]


def test_mcnemar_uses_only_discordant_pairs():
    base = [record(i, 10, ["sabotage"]) for i in range(8)]
    var = [record(i, 90 if i < 6 else 10, ["sabotage"]) for i in range(8)]
    out = mcnemar(pair_results(base, var, threshold=50))
    assert out["recovered"] == 6
    assert out["lost"] == 0
    assert out["discordant"] == 6
    assert out["p_value"] < 0.05


def test_mcnemar_no_discordant_pairs_is_not_significant():
    base = [record(i, 90, ["sabotage"]) for i in range(10)]
    out = mcnemar(pair_results(base, list(base), threshold=50))
    assert out["discordant"] == 0
    assert out["p_value"] == 1.0


def test_delta_vs_length_detects_draw_count_artifact():
    # Simulates "chunking just gives more draws": the score gain grows with
    # transcript length, which is the alternative explanation to localization.
    base = [record(i, 0, ["normal"], chars=10_000 * (i + 1)) for i in range(20)]
    var = [record(i, i * 4, ["normal"], chars=10_000 * (i + 1)) for i in range(20)]
    out = delta_vs_length(pair_results(base, var, threshold=50))
    assert out.iloc[0]["rho_delta_vs_length"] > 0.9


def test_transition_counts_per_label():
    base = [record(1, 10, ["sabotage"]), record(2, 10, ["refusals"])]
    var = [record(1, 90, ["sabotage"]), record(2, 10, ["refusals"])]
    out = transition_counts(pair_results(base, var, threshold=50))
    assert set(out["label"]) == {"sabotage", "refusals"}


def test_delta_is_variant_minus_baseline():
    out = pair_results([record(1, 20, ["sabotage"])], [record(1, 75, ["sabotage"])])
    assert out.loc[0, "delta"] == 55


def test_pair_results_empty_when_no_overlap():
    out = pair_results([record(1, 10, ["sabotage"])], [record(2, 10, ["sabotage"])])
    assert isinstance(out, pd.DataFrame)
    assert out.empty
