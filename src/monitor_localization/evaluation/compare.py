"""Paired comparison of two monitor architectures on the same transcripts.

The comparison is paired — every run is scored by both monitors — so run-level
variance cancels and McNemar's test on the discordant pairs is the right
significance test rather than comparing two independent recall estimates.

Length is carried through every output. Because the global monitor's score
already rises with transcript length, and max aggregation gives long transcripts
many more draws, a raw improvement on long labels is exactly what "chunking is
just more draws" would also produce. Separating those requires the length
regression here plus the shuffled-chunk control.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from monitor_localization.evaluation.failures import (
    FALSE_NEGATIVE,
    TRUE_POSITIVE,
    classify,
)
from monitor_localization.evaluation.length import (
    MIN_SPREAD_FOR_CORRELATION,
    add_length_columns,
)
from monitor_localization.evaluation.metrics import DEFAULT_THRESHOLD

RECOVERED = "recovered"  # global missed it, chunk caught it
LOST = "lost"  # global caught it, chunk missed it
BOTH_CAUGHT = "both_caught"
BOTH_MISSED = "both_missed"


def pair_results(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Join two result sets on run_id, keeping only runs both monitors scored.

    Runs that failed in either arm are dropped rather than counted as misses: a
    failed call is missing data, and scoring it as clean would fabricate a
    recovery or a loss depending on which arm failed.
    """
    def frame(rows: list[dict[str, Any]], suffix: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "run_id": r["run_id"],
                    f"score_{suffix}": (
                        float(r["score"]) if r.get("score") is not None else float("nan")
                    ),
                    f"error_{suffix}": r.get("error"),
                    f"reasoning_{suffix}": r.get("reasoning", ""),
                    "is_positive": bool(r.get("is_positive")),
                    "labels": ";".join(r.get("labels") or []),
                    "n_characters": (r.get("metadata") or {}).get("n_characters"),
                    "n_messages": (r.get("metadata") or {}).get("n_messages"),
                }
                for r in rows
            ]
        )

    left = frame(baseline, "baseline")
    right = frame(variant, "variant").drop(
        columns=["is_positive", "labels", "n_characters", "n_messages"]
    )
    merged = left.merge(right, on="run_id", how="inner")

    usable = merged["score_baseline"].notna() & merged["score_variant"].notna()
    merged = merged[usable].copy()

    merged["outcome_baseline"] = [
        classify(s, p, threshold)
        for s, p in zip(merged["score_baseline"], merged["is_positive"], strict=True)
    ]
    merged["outcome_variant"] = [
        classify(s, p, threshold)
        for s, p in zip(merged["score_variant"], merged["is_positive"], strict=True)
    ]
    merged["delta"] = merged["score_variant"] - merged["score_baseline"]
    merged["transition"] = [
        _transition(b, v)
        for b, v in zip(merged["outcome_baseline"], merged["outcome_variant"], strict=True)
    ]
    return add_length_columns(merged)


def _transition(baseline: str, variant: str) -> str:
    if baseline == FALSE_NEGATIVE and variant == TRUE_POSITIVE:
        return RECOVERED
    if baseline == TRUE_POSITIVE and variant == FALSE_NEGATIVE:
        return LOST
    if baseline == TRUE_POSITIVE and variant == TRUE_POSITIVE:
        return BOTH_CAUGHT
    if baseline == FALSE_NEGATIVE and variant == FALSE_NEGATIVE:
        return BOTH_MISSED
    return f"{baseline}->{variant}"


def mcnemar(paired: pd.DataFrame) -> dict[str, Any]:
    """Exact McNemar on positives: does the variant change detection?

    Only the discordant pairs carry information. Runs both monitors caught, or
    both missed, contribute nothing to the test — which is why raw recall
    differences overstate the evidence.
    """
    from scipy import stats

    positives = paired[paired["is_positive"]]
    recovered = int((positives["transition"] == RECOVERED).sum())
    lost = int((positives["transition"] == LOST).sum())
    discordant = recovered + lost
    p_value = (
        float(stats.binomtest(recovered, discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {
        "n_positive": int(len(positives)),
        "recovered": recovered,
        "lost": lost,
        "discordant": discordant,
        "net": recovered - lost,
        # Not rounded: a strongly significant result rounds to 0.0 at 4dp, which
        # reads as a missing value. Formatting is the caller's job.
        "p_value": p_value,
        "recall_baseline": round(
            float((positives["outcome_baseline"] == TRUE_POSITIVE).mean()), 4
        ),
        "recall_variant": round(
            float((positives["outcome_variant"] == TRUE_POSITIVE).mean()), 4
        ),
    }


def delta_vs_length(paired: pd.DataFrame, min_runs: int = 15) -> pd.DataFrame:
    """Within-label correlation between transcript length and the score delta.

    This is the control for the leading alternative explanation. If chunking
    helps because it localises evidence, the benefit should not be explained by
    length alone. If it helps because max aggregation takes more draws on longer
    transcripts, the delta will rise with length *within* every label, including
    benign — where any increase is a false positive, not a recovery.
    """
    from scipy import stats

    frame = paired.copy()
    frame["label_list"] = frame["labels"].apply(
        lambda s: [x for x in str(s).split(";") if x and x != "normal"] or ["benign"]
    )
    exploded = frame.explode("label_list")

    rows: list[dict[str, Any]] = []
    for label, group in exploded.groupby("label_list"):
        group = group[group["tokens"].notna() & group["delta"].notna()]
        # Spearman is undefined when either input is constant.
        if (
            len(group) < min_runs
            or group["delta"].nunique() < 2
            or group["tokens"].nunique() < 2
        ):
            continue
        p10, p90 = np.percentile(group["tokens"], [10, 90])
        rho, p_value = stats.spearmanr(group["tokens"], group["delta"])
        rows.append(
            {
                "label": label,
                "n": len(group),
                "spread": round(p90 / max(p10, 1), 1),
                "mean_delta": round(float(group["delta"].mean()), 2),
                "rho_delta_vs_length": round(float(rho), 3),
                "p": round(float(p_value), 4),
            }
        )
    columns = ["label", "n", "spread", "mean_delta", "rho_delta_vs_length", "p"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("rho_delta_vs_length", ascending=False)


def _explode_labels(paired: pd.DataFrame) -> pd.DataFrame:
    frame = paired.copy()
    frame["label_list"] = frame["labels"].apply(
        lambda s: [x for x in str(s).split(";") if x and x != "normal"] or ["benign"]
    )
    return frame.explode("label_list")


def length_effect_by_arm(paired: pd.DataFrame, min_runs: int = 15) -> pd.DataFrame:
    """Within-label length effect for each arm separately, plus the delta.

    `delta_vs_length` nets the two arms against each other, which hides whether a
    correlation in the delta reflects chunking introducing a length bias or
    merely inheriting one the global monitor already had. Reporting each arm's
    own slope separates those: if the global slope is already steep and the chunk
    slope matches it, chunking is not the source.

    Two statistics per arm, because they answer different questions:

    - `rho`  Spearman rank correlation — is there a monotone length effect at
             all? Robust to the score distribution being bimodal, which these
             are, but unitless.
    - `slope` OLS of score on log10(tokens), in score points per 10x length —
             how *big* the effect is on the 0-100 scale the threshold lives on.
             A significant rho on a slope of ~1 point/decade is not a threat to
             a decision made at 50.
    """
    from scipy import stats

    exploded = _explode_labels(paired)

    rows: list[dict[str, Any]] = []
    for label, group in exploded.groupby("label_list"):
        group = group[group["tokens"].notna() & (group["tokens"] > 0)]
        if len(group) < min_runs or group["tokens"].nunique() < 2:
            continue
        log_tokens = np.log10(group["tokens"])
        p10, p90 = np.percentile(group["tokens"], [10, 90])
        spread = p90 / max(p10, 1)
        row: dict[str, Any] = {
            "label": label,
            "n": len(group),
            "spread": round(spread, 1),
            # Below this spread the label barely varies in length, so a
            # per-decade slope extrapolates well past the observed range.
            "interpretable": spread >= MIN_SPREAD_FOR_CORRELATION,
        }
        for arm, column in (
            ("global", "score_baseline"),
            ("chunk", "score_variant"),
            ("delta", "delta"),
        ):
            values = group[column]
            # Spearman and OLS are both undefined against a constant input.
            if values.nunique() < 2:
                row |= {f"rho_{arm}": 0.0, f"p_{arm}": 1.0, f"slope_{arm}": 0.0}
                continue
            rho, p_value = stats.spearmanr(log_tokens, values)
            fit = stats.linregress(log_tokens, values)
            row |= {
                f"rho_{arm}": round(float(rho), 3),
                f"p_{arm}": round(float(p_value), 4),
                f"slope_{arm}": round(float(fit.slope), 1),
            }
        rows.append(row)

    columns = ["label", "n", "spread", "interpretable"] + [
        f"{stat}_{arm}"
        for arm in ("global", "chunk", "delta")
        for stat in ("rho", "p", "slope")
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("rho_delta", ascending=False)


def transition_counts(paired: pd.DataFrame) -> pd.DataFrame:
    """Per-label transition table — the recovery analysis the notes call for."""
    frame = paired.copy()
    frame["label_list"] = frame["labels"].apply(
        lambda s: [x for x in str(s).split(";") if x and x != "normal"] or ["benign"]
    )
    exploded = frame.explode("label_list")
    table = (
        exploded.groupby(["label_list", "transition"]).size().unstack(fill_value=0)
    )
    table.index.name = "label"
    return table.reset_index()
