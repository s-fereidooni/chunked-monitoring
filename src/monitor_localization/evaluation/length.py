"""Length-controlled analysis.

Transcript length is confounded with behavior label in MALT — median length
spans 24x across sandbagging labels — and the global monitor's score rises with
length even on benign runs. Any comparison between labels, or between monitor
architectures, therefore has to control for it rather than compare raw means.

Most labels vary 10-37x in length *within* themselves, so length can be tested
independently of label. `partial_problem_solving` is the exception at 2.3x: it
is uniformly long, and within-label length analysis there is uninformative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from monitor_localization.dataset.schema import BENIGN_LABEL
from monitor_localization.evaluation.metrics import DEFAULT_THRESHOLD

# Measured on 36 real MALT transcripts with o200k_base (aggregate 4.40).
CHARS_PER_TOKEN = 4.4

# Below this many runs, a within-label correlation is not worth reporting.
MIN_RUNS_FOR_CORRELATION = 15
# Below this p90/p10 ratio a label has too little internal spread for length to
# be separable from the label itself.
MIN_SPREAD_FOR_CORRELATION = 3.0


def add_length_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Attach token, log-token and decile columns derived from n_characters."""
    out = table.copy()
    chars = pd.to_numeric(out.get("n_characters"), errors="coerce")
    out["tokens"] = chars / CHARS_PER_TOKEN
    out["log_tokens"] = np.log10(out["tokens"].clip(lower=1))
    usable = out["tokens"].notna()
    out["length_decile"] = pd.Series(pd.NA, index=out.index, dtype="object")
    if usable.sum() >= 10:
        out.loc[usable, "length_decile"] = pd.qcut(
            out.loc[usable, "tokens"], 10, labels=False, duplicates="drop"
        )
    return out


def length_by_label(table: pd.DataFrame) -> pd.DataFrame:
    """Per-label length distribution, including the internal spread ratio.

    `spread` is p90/p10 — how much length varies *within* the label. A label with
    low spread cannot be used to separate length effects from label effects.
    """
    frame = add_length_columns(table)
    frame = frame[frame["tokens"].notna()].copy()
    frame["label_list"] = frame["labels"].apply(
        lambda s: [x for x in str(s).split(";") if x and x != BENIGN_LABEL] or [BENIGN_LABEL]
    )
    exploded = frame.explode("label_list")

    rows: list[dict[str, Any]] = []
    for label, group in exploded.groupby("label_list"):
        p10, median, p90 = np.percentile(group["tokens"], [10, 50, 90])
        rows.append(
            {
                "label": label,
                "n": len(group),
                "p10": round(p10),
                "median": round(median),
                "p90": round(p90),
                "max": round(group["tokens"].max()),
                "spread": round(p90 / max(p10, 1), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("median").reset_index(drop=True)


def score_vs_length(table: pd.DataFrame) -> pd.DataFrame:
    """Within-label rank correlation between transcript length and score.

    A positive correlation on the benign rows is the one to watch: it means the
    monitor grows more suspicious of longer transcripts regardless of behavior,
    which inflates false positives on long runs and would be amplified by any
    aggregation that takes a maximum over more chunks.
    """
    from scipy import stats

    frame = add_length_columns(table)
    frame = frame[frame["tokens"].notna()].copy()
    frame["label_list"] = frame["labels"].apply(
        lambda s: [x for x in str(s).split(";") if x and x != BENIGN_LABEL] or [BENIGN_LABEL]
    )
    exploded = frame.explode("label_list")

    rows: list[dict[str, Any]] = []
    for label, group in exploded.groupby("label_list"):
        group = group[group["score"].notna()]
        # Spearman is undefined when either input is constant.
        if (
            len(group) < MIN_RUNS_FOR_CORRELATION
            or group["score"].nunique() < 2
            or group["tokens"].nunique() < 2
        ):
            continue
        p10, p90 = np.percentile(group["tokens"], [10, 90])
        spread = p90 / max(p10, 1)
        rho, p_value = stats.spearmanr(group["tokens"], group["score"])
        rows.append(
            {
                "label": label,
                "n": len(group),
                "spread": round(spread, 1),
                "rho": round(float(rho), 3),
                "p": round(float(p_value), 4),
                "interpretable": spread >= MIN_SPREAD_FOR_CORRELATION,
            }
        )
    columns = ["label", "n", "spread", "rho", "p", "interpretable"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


def fpr_by_length_decile(
    table: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """False-positive rate across length deciles, benign runs only.

    If the monitor's suspicion scales with length, false positives concentrate in
    the upper deciles. Worth checking separately for each architecture: chunked
    monitoring with max aggregation takes the highest of many more draws on long
    transcripts, so it should amplify any such trend.
    """
    frame = add_length_columns(table)
    benign = frame[(~frame["is_positive"]) & frame["tokens"].notna()].copy()
    if benign.empty:
        return pd.DataFrame(columns=["length_decile", "n", "median_tokens", "n_flagged", "fpr"])
    benign["flagged"] = benign["score"] >= threshold
    grouped = (
        benign.groupby("length_decile")
        .agg(
            n=("run_id", "count"),
            median_tokens=("tokens", "median"),
            n_flagged=("flagged", "sum"),
        )
        .reset_index()
    )
    grouped["fpr"] = (grouped["n_flagged"] / grouped["n"]).round(4)
    grouped["median_tokens"] = grouped["median_tokens"].round()
    return grouped
