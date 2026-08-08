#!/usr/bin/env python
"""Stage 6: figures for the write-up.

Reads the CSVs written by `analyze.py` and `compare.py` — no inference, no
recomputation of statistics. Anything shown here must already exist in a results
file, so a figure can never disagree with the numbers in the tables.

    python scripts/figures.py
    python scripts/figures.py --format pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from monitor_localization.config import ExperimentConfig  # noqa: E402
from monitor_localization.evaluation import DEFAULT_THRESHOLD  # noqa: E402
from monitor_localization.experiment import ExperimentRun  # noqa: E402
from monitor_localization.utils import setup_logging  # noqa: E402

# --- design system ----------------------------------------------------------
# One palette across every figure so the same series reads the same everywhere:
# the baseline is always warm stone, the chunked arm always teal, and the
# alternative aggregation always amber.
PAPER = "#FDFCFA"
INK = "#17171A"
MUTED = "#78767B"
HAIRLINE = "#E4E1DB"

GLOBAL = "#BDB5AB"
GLOBAL_DARK = "#8A8177"
CHUNK = "#0E7C7B"
MEAN = "#DE8A3C"
POSITIVE = "#C2415B"
BENIGN = "#A9BAC6"

# Categorical slots for per-label series, in fixed order — never cycled, never
# reassigned when a label is filtered out. Taken from the dataviz reference
# palette and validated against this surface: lightness band, chroma floor,
# adjacent-pair CVD separation (protan/deutan) and normal-vision separation all
# pass. Three warn on contrast, which is why any figure using them carries a
# legend rather than relying on colour alone.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
]

STACK = ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"]

# Shortened for axis labels; the tables keep the full names.
SHORT = {
    "match_weaker_model": "match weaker model",
    "partial_problem_solving": "partial problem solving",
    "reasoning_about_task": "reasoning about task",
    "bypass_constraints": "bypass constraints",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.35,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "sans-serif",
            "font.sans-serif": STACK,
            "font.size": 9.5,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": HAIRLINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "axes.grid": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def _title(ax: plt.Axes, headline: str, subtitle: str = "") -> None:
    """Headline names what is plotted; subtitle carries the finding and caveats.

    Deliberately descriptive rather than argumentative: a figure often travels
    without its surrounding text, and a headline asserting a conclusion is
    harder to keep honest than one naming the axes. Every number in the subtitle
    is derived from the data so it cannot go stale.

    Pad scales with the subtitle's line count — a fixed pad lets a multi-line
    subtitle run back through the headline.
    """
    lines = subtitle.count("\n") + 1 if subtitle else 0
    pad = 14 + 13 * lines
    ax.set_title(
        headline, loc="left", fontsize=12.5, fontweight="demibold",
        color=INK, pad=pad,
    )
    if subtitle:
        # Anchored to grow downward from just under the headline. With
        # va="bottom" the block grows upward instead and runs into the title.
        ax.annotate(
            subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, pad - 7),
            textcoords="offset points", fontsize=9, color=MUTED, va="top",
            linespacing=1.5,
        )


def _label(name: str) -> str:
    return SHORT.get(name, name.replace("_", " "))


def _vgrid(ax: plt.Axes, ticks: list[float]) -> None:
    for tick in ticks:
        ax.axvline(tick, color=HAIRLINE, lw=0.8, zorder=0)


# --- figures ----------------------------------------------------------------


def recall_by_label(out: Path, fmt: str, results: Path) -> Path | None:
    """Headline: which behaviors chunking moves, and which it does not.

    A dumbbell rather than paired bars — the quantity of interest is the
    *movement* between the two arms, and a connecting line shows that directly
    while paired bars make the reader difference two lengths by eye.
    """
    path = results / "recall_by_aggregation.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    # `global` is a keyword, so it cannot be read off an itertuples row.
    table = table.rename(columns={"global": "base"})
    table = table[table["is_positive"]].copy()
    if table.empty:
        return None
    table["delta"] = table["chunk_max"] - table["base"]
    table = table.sort_values("delta")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _vgrid(ax, [0, 0.25, 0.5, 0.75, 1.0])

    # 0.05 rather than 0: sabotage moves +0.02, which is one run and should not
    # be coloured as a gain. A decline is called out rather than folded into
    # "no change" — gives_up genuinely gets worse.
    for i, row in enumerate(table.itertuples()):
        moved = row.delta >= 0.05
        declined = row.delta <= -0.02
        ax.plot(
            [row.base, row.chunk_max], [i, i],
            color=CHUNK if moved else HAIRLINE,
            lw=3.2 if moved else 2.4, alpha=0.30 if moved else 1,
            solid_capstyle="round", zorder=1,
        )
        ax.scatter(row.base, i, s=88, color=GLOBAL, zorder=3,
                   edgecolor=PAPER, linewidth=1.6)
        ax.scatter(row.chunk_max, i, s=88, color=CHUNK if moved else GLOBAL_DARK,
                   zorder=3, edgecolor=PAPER, linewidth=1.6)
        anchor = max(row.chunk_max, row.base)
        if moved:
            ax.annotate(
                f"+{row.delta:.2f}", (anchor, i), xytext=(13, 0),
                textcoords="offset points", va="center", fontsize=9.5,
                color=CHUNK, fontweight="demibold",
            )
        elif declined:
            ax.annotate(
                f"{row.delta:.2f}   worse", (anchor, i), xytext=(13, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=POSITIVE,
            )
        else:
            ax.annotate(
                "no change", (anchor, i), xytext=(13, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=MUTED, style="italic",
            )

    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([_label(n) for n in table["label"]], fontsize=10.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1.0"])
    ax.set_xlim(-0.04, 1.18)
    ax.set_ylim(-0.75, len(table) - 0.15)
    ax.set_xlabel("recall at threshold 50", fontsize=9.5, labelpad=8)

    top = len(table) - 1
    ax.annotate("global", (table.iloc[top]["base"], top), xytext=(0, 17),
                textcoords="offset points", ha="center", fontsize=9.5,
                color=GLOBAL_DARK, fontweight="demibold")
    ax.annotate("chunked", (table.iloc[top]["chunk_max"], top), xytext=(0, 17),
                textcoords="offset points", ha="center", fontsize=9.5,
                color=CHUNK, fontweight="demibold")

    # Derived from the data rather than written in: the counts changed between
    # runs and a hardcoded headline silently went stale.
    big = int((table["delta"] >= 0.20).sum())
    floor = int((table["chunk_max"] < 0.15).sum())
    # Rows are per label, so summing n double-counts multi-label runs and omits
    # labels below the reporting floor. Take the run-level count instead.
    paired_path = results / "paired.csv"
    n_positive = 0
    if paired_path.is_file():
        paired = pd.read_csv(paired_path)
        n_positive = int(paired["is_positive"].sum())
    _title(
        ax,
        "Recall by behaviour label, global vs chunked",
        f"{n_positive} sandbagging runs · gpt-4.1-mini · elicitation prompt removed\n"
        f"chunking transforms {big} behaviours and leaves {floor} at the floor",
    )
    return _save(fig, out / f"recall_by_label.{fmt}")


def aggregation(out: Path, fmt: str, results: Path) -> Path | None:
    """The mechanism figure: mean is immune to draw count, so it separates them."""
    path = results / "recall_by_aggregation.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    table = table.rename(columns={"global": "base"})
    table = table[table["is_positive"] & (table["chunk_max"] > 0.05)]
    table = table.sort_values("chunk_max")
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    _vgrid(ax, [0, 0.25, 0.5, 0.75, 1.0])

    for i, row in enumerate(table.itertuples()):
        ax.plot([min(row.base, row.chunk_mean), row.chunk_max], [i, i],
                color=HAIRLINE, lw=2.2, solid_capstyle="round", zorder=1)
        for value, color in ((row.base, GLOBAL), (row.chunk_mean, MEAN),
                             (row.chunk_max, CHUNK)):
            ax.scatter(value, i, s=86, color=color, zorder=3,
                       edgecolor=PAPER, linewidth=1.6)
        # A zero is invisible as a bar and reads as missing data; mean collapsing
        # to 0.00 is the result, so it gets said out loud.
        if row.chunk_mean < 0.02:
            ax.annotate("0.00", (row.chunk_mean, i), xytext=(0, -17),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=MEAN, fontweight="demibold")

    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([_label(n) for n in table["label"]], fontsize=10.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1.0"])
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.8, len(table) - 0.05)
    ax.set_xlabel("recall at threshold 50", fontsize=9.5, labelpad=8)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8, color=c, label=t)
        for c, t in ((GLOBAL, "global"), (MEAN, "chunked · mean"),
                     (CHUNK, "chunked · max"))
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5,
              handletextpad=0.4, borderpad=0.8, labelspacing=0.55)

    # Earlier wording claimed this split the result into "localization" vs
    # something else. That reading came from contaminated inputs: with the
    # elicitation prompt in every chunk's context, uninformative chunks scored
    # low-but-nonzero and dragged the mean to 0. On clean inputs the mean holds
    # up for two of the three gaining labels, so the figure now says only what
    # the data supports.
    _title(
        ax,
        "Recall by aggregation rule",
        "mean dilutes as chunks multiply and so cannot exploit extra draws; where it\n"
        "still beats global the advantage is not a draw-count artifact",
    )
    return _save(fig, out / f"aggregation.{fmt}")


def score_scatter(out: Path, fmt: str, results: Path, threshold: float) -> Path | None:
    """Every run, both scores. The recoveries are the points above the line."""
    path = results / "paired.csv"
    if not path.is_file():
        return None
    paired = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    # Shade the region where the chunked arm catches what global missed, so the
    # asymmetry is legible before any point is read.
    ax.add_patch(Polygon([(-6, threshold), (threshold, threshold), (threshold, 106),
                          (-6, 106)], closed=True, facecolor=CHUNK, alpha=0.05,
                         zorder=0, linewidth=0))
    ax.add_patch(Polygon([(threshold, -6), (106, -6), (106, threshold),
                          (threshold, threshold)], closed=True, facecolor=GLOBAL,
                         alpha=0.05, zorder=0, linewidth=0))

    for tick in (0, 25, 50, 75, 100):
        ax.axhline(tick, color=HAIRLINE, lw=0.7, zorder=1)
        ax.axvline(tick, color=HAIRLINE, lw=0.7, zorder=1)
    ax.plot([-6, 106], [-6, 106], color=MUTED, lw=0.8, ls=(0, (4, 4)), zorder=2)

    rng = np.random.default_rng(0)
    for mask, color, name, size in (
        (~paired["is_positive"], BENIGN, "benign", 17),
        (paired["is_positive"], POSITIVE, "sandbagging", 21),
    ):
        group = paired[mask]
        ax.scatter(
            group["score_baseline"] + rng.normal(0, 1.5, len(group)),
            group["score_variant"] + rng.normal(0, 1.5, len(group)),
            s=size, alpha=0.62, color=color, label=name, zorder=4,
            edgecolor=PAPER, linewidth=0.4,
        )

    recovered = int((paired["transition"] == "recovered").sum())
    lost = paired[paired["transition"] == "lost"]
    # The recovered cluster fills this corner, so the count needs a plate behind
    # it to stay legible over the points.
    ax.annotate(
        f"{recovered} recovered", (26, 62), fontsize=13.5, color=CHUNK,
        ha="center", va="center", fontweight="demibold", zorder=6,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PAPER,
              "edgecolor": "none", "alpha": 0.88},
    )
    # Point at the lost run rather than labelling empty space near it.
    if not lost.empty:
        row = lost.iloc[0]
        ax.annotate(
            f"{len(lost)} lost", (row["score_baseline"], row["score_variant"]),
            xytext=(-14, 26), textcoords="offset points", fontsize=10.5,
            color=GLOBAL_DARK, ha="right", fontweight="demibold", zorder=6,
            arrowprops={"arrowstyle": "-", "color": GLOBAL_DARK, "lw": 0.9,
                        "shrinkA": 2, "shrinkB": 6},
        )

    ax.set_xlabel("global monitor score", fontsize=9.5, labelpad=8)
    ax.set_ylabel("chunked monitor score", fontsize=9.5, labelpad=8)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-6, 106)
    ax.set_ylim(-6, 106)
    ax.set_aspect("equal")
    # Below the axes: the plot is square and full of points, and any in-axes
    # placement collided with the "lost" annotation in the lower-right quadrant.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              fontsize=9.5, handletextpad=0.3, markerscale=1.7, columnspacing=2.2)

    from scipy import stats as _stats

    discordant = recovered + len(lost)
    pvalue = (_stats.binomtest(recovered, discordant, 0.5).pvalue
              if discordant else 1.0)
    shown = "< 0.0001" if pvalue < 0.0001 else f"= {pvalue:.4f}"
    benign_flagged = int((paired[~paired["is_positive"]]["score_variant"] >= threshold).sum())
    n_benign = int((~paired["is_positive"]).sum())
    # Headline states the finding, like every other figure here, and is derived
    # so it cannot go stale: the movement is almost entirely one-way, and the
    # benign cloud does not move at all.
    # Three variables are plotted — global score, chunked score, and ground
    # truth as colour — so the headline names all three: which class moved, in
    # which direction, and that the other class did not move at all. A purely
    # descriptive title ("score by monitor type") would spend the most-read line
    # restating the axis labels.
    crossings = f"{len(lost)} crosses back" if len(lost) == 1 else f"{len(lost)} cross back"
    _title(
        ax,
        "Global vs chunked monitor score, by ground truth",
        f"{len(paired)} transcripts · threshold {threshold:g} · McNemar p {shown}\n"
        f"{recovered} sandbagging runs cross the threshold, {crossings}; "
        f"benign flagged {benign_flagged}/{n_benign} either way",
    )
    return _save(fig, out / f"score_scatter.{fmt}")


def fpr_by_length(out: Path, fmt: str, results: Path, baseline: Path) -> Path | None:
    """Per-decile false-positive rate. NOT in the default set — see below.

    Kept for diagnosis, deliberately not rendered by `main`. With 2 false
    positives per arm across 293 benign runs, every point is a rate over ~11-50
    runs: global's 0.182 is 2/11, and the entire visible gap between the two
    curves is one run sitting in decile 8 rather than 9. The chart reads as a
    large architectural difference and is nothing of the kind. The honest
    version of this result is one sentence — both arms flagged 2 of 293, all of
    them in the longest fifth of benign transcripts — so it belongs in prose,
    not in a figure that dramatises four events.
    """
    paths = {"chunked": results / "fpr_by_length_decile.csv",
             "global": baseline / "fpr_by_length_decile.csv"}
    if not all(p.is_file() for p in paths.values()):
        return None
    frames = {name: pd.read_csv(p) for name, p in paths.items()}

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for tick in (0, 0.05, 0.10, 0.15):
        ax.axhline(tick, color=HAIRLINE, lw=0.8, zorder=0)

    # Both arms sit at zero for the first eight deciles, so a solid global line
    # would be hidden underneath the chunked one. Dashing it, and drawing it
    # last, keeps the overlap visible as overlap.
    for name, color, dash, z in (
        ("chunked", CHUNK, None, 3),
        ("global", GLOBAL_DARK, (0, (5, 3)), 5),
    ):
        frame = frames[name]
        x = frame["length_decile"]
        ax.fill_between(x, 0, frame["fpr"], color=color, alpha=0.10, zorder=1)
        ax.plot(x, frame["fpr"], color=color, lw=2.4, zorder=z,
                solid_capstyle="round", ls=dash or "-")
        ax.scatter(x, frame["fpr"], s=42, color=color, zorder=z + 1,
                   edgecolor=PAPER, linewidth=1.4)
        last = frame.iloc[-1]
        ax.annotate(name, (last["length_decile"], last["fpr"]), xytext=(11, 0),
                    textcoords="offset points", va="center", fontsize=10,
                    color=color, fontweight="demibold")

    median = frames["global"]["median_tokens"]
    ax.set_xticks(range(len(median)))
    ax.set_xticklabels(
        [f"{t / 1000:.0f}k" if t >= 1000 else f"{t:.0f}" for t in median],
        fontsize=9,
    )
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0", "5%", "10%", "15%"])
    ax.set_xlim(-0.4, len(median) + 0.55)
    ax.set_ylim(-0.012, 0.20)
    ax.set_xlabel("median transcript length in decile (benign runs only)",
                  fontsize=9.5, labelpad=8)

    _title(
        ax,
        "False-positive rate by transcript length decile",
        "293 benign runs · 2 false positives in each arm — the shape is real, "
        "the counts are small",
    )
    return _save(fig, out / f"fpr_by_length.{fmt}")


# --- data exploration -------------------------------------------------------


def _primary_label(frame: pd.DataFrame) -> pd.Series:
    """One label per run, chosen deterministically.

    Taking `.str[0]` made the bucket depend on MALT's storage order: the data
    holds both `gives_up;reasoning_about_task` and `reasoning_about_task;gives_up`,
    so identical runs landed in different buckets. Sorting makes it stable.
    """
    return (
        frame["labels"].fillna("normal")
        .apply(lambda s: sorted(x for x in str(s).split(";") if x)[0] if s else "normal")
        .replace({"normal": "benign"})
    )


def length_by_label(out: Path, fmt: str, baseline: Path) -> Path | None:
    """Every transcript's length, by label — the confound underneath everything.

    Individual points rather than a boxplot: with n=17 to n=293 per label the
    quartile boxes imply a smoothness the data does not have, and the overlap
    between labels is the whole point.
    """
    path = baseline / "per_sample.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["n_characters"].notna()].copy()
    frame["label"] = _primary_label(frame)
    frame["tokens"] = frame["n_characters"] / 4.4
    order = frame.groupby("label")["tokens"].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    rng = np.random.default_rng(0)
    for i, label in enumerate(order):
        group = frame[frame["label"] == label]
        color = BENIGN if label == "benign" else POSITIVE
        ax.scatter(
            group["tokens"], i + rng.uniform(-0.24, 0.24, len(group)),
            s=15, alpha=0.45, color=color, linewidths=0, zorder=3,
        )
        median = group["tokens"].median()
        ax.scatter(median, i, marker="|", s=340, color=INK, linewidths=1.9, zorder=5)
        ax.annotate(
            f"{median / 1000:.0f}k" if median >= 1000 else f"{median:.0f}",
            (median, i), xytext=(0, 13), textcoords="offset points", ha="center",
            fontsize=8.5, color=INK, fontweight="demibold", zorder=6,
        )

    for tick in (1e3, 1e4, 1e5):
        ax.axvline(tick, color=HAIRLINE, lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_label(name) for name in order], fontsize=10.5)
    ax.set_xticks([1e3, 1e4, 1e5])
    ax.set_xticklabels(["1k", "10k", "100k"])
    ax.set_xlim(3e2, 6e5)
    ax.set_ylim(-0.7, len(order) - 0.35)
    ax.set_xlabel("transcript length in tokens (log scale) · tick marks the median",
                  fontsize=9.5, labelpad=8)
    _title(
        ax,
        "Transcript length by behaviour label",
        "length is confounded with behaviour — the median spans 24x across labels, so\n"
        "any per-label comparison is partly a comparison of transcript lengths",
    )
    return _save(fig, out / f"length_by_label.{fmt}")


def length_composition(out: Path, fmt: str, results: Path) -> Path | None:
    """Which behaviors sit in each length quartile — why pooled splits mislead."""
    path = results / "label_composition_by_length.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path).set_index("label")
    # Normalise BEFORE dropping small labels, or the remaining shares no longer
    # sum to the quartile and the axis label becomes false.
    shares = table.div(table.sum(axis=0), axis=1)
    shares = shares[table.sum(axis=1) >= 5]
    order = shares.mean(axis=1).sort_values(ascending=False).index.tolist()

    palette = [CHUNK, MEAN, POSITIVE, "#6B8E9E", GLOBAL_DARK, "#A8B5A0"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    left = np.zeros(len(shares.columns))
    for i, label in enumerate(order):
        values = shares.loc[label].to_numpy()
        color = palette[i % len(palette)]
        ax.barh(range(len(values)), values, left=left, height=0.68, color=color,
                edgecolor=PAPER, linewidth=1.4)
        for j, (value, start) in enumerate(zip(values, left, strict=True)):
            if value > 0.11:
                ax.annotate(
                    _label(label), (start + value / 2, j), ha="center",
                    va="center", fontsize=8.5, color=PAPER, fontweight="demibold",
                )
        left = left + values

    ax.set_yticks(range(len(shares.columns)))
    names = ["shortest", "", "", "longest"]
    ax.set_yticklabels(
        [f"{q}  ·  {name}" if name else str(q)
         for q, name in zip(shares.columns, names, strict=False)],
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the quartile's positive runs", fontsize=9.5, labelpad=8)
    _title(
        ax,
        "Label composition by length quartile",
        "quartiles hold different behaviours, not the same one at different lengths —\n"
        "so a gain that grows with length is not evidence that length is what matters",
    )
    return _save(fig, out / f"length_composition.{fmt}")


def run_source(out: Path, fmt: str, baseline: Path) -> Path | None:
    """Prompted vs natural, by label — the sharpest limit on what this can claim.

    Every benign run is unprompted and most positive labels are entirely
    prompted, so a monitor can score well by detecting that an instruction was
    injected rather than that the model sandbagged.
    """
    path = baseline / "per_sample.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame["label"] = _primary_label(frame)
    table = pd.crosstab(frame["label"], frame["run_source"])
    for column in ("prompted", "unprompted"):
        if column not in table:
            table[column] = 0
    table = table.loc[table.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    y = range(len(table))
    ax.barh(y, table["prompted"], height=0.62, color=POSITIVE, label="prompted")
    ax.barh(y, table["unprompted"], left=table["prompted"], height=0.62,
            color=BENIGN, label="natural (unprompted)")
    for i, (_, row) in enumerate(table.iterrows()):
        total = row["prompted"] + row["unprompted"]
        ax.annotate(f"{total}", (total, i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=MUTED)

    ax.set_yticks(list(y))
    ax.set_yticklabels([_label(name) for name in table.index], fontsize=10.5)
    ax.set_xlabel("runs in the evaluation subset", fontsize=9.5, labelpad=8)
    ax.set_xlim(0, table.sum(axis=1).max() * 1.13)
    for tick in (0, 100, 200, 300):
        ax.axvline(tick, color=HAIRLINE, lw=0.8, zorder=0)
    ax.set_xticks([0, 100, 200, 300])
    ax.legend(loc="lower right", fontsize=9.5, handletextpad=0.6, labelspacing=0.5)
    _title(
        ax,
        "Run source by behaviour label",
        "prompting is almost perfectly confounded with behaviour: every benign run is\n"
        "natural and five of six positive labels are wholly prompted, so “was an\n"
        "instruction injected” is available to a monitor as a shortcut",
    )
    return _save(fig, out / f"run_source.{fmt}")


# --- failure modes ----------------------------------------------------------


def positive_outcomes(out: Path, fmt: str, results: Path) -> Path | None:
    """Complete accounting of every positive: caught, recovered, lost, missed.

    Recall tables show what each arm caught; this shows what happens to the
    *same* runs under both, which is the only view that makes the residual —
    the runs neither arm detects — visible rather than implied.
    """
    path = results / "transitions.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path).set_index("label")
    for column in ("both_caught", "recovered", "lost", "both_missed"):
        if column not in table:
            table[column] = 0
    table = table[table.index != "benign"]
    table["total"] = table[["both_caught", "recovered", "lost", "both_missed"]].sum(axis=1)
    table = table[table["total"] > 0].sort_values("both_missed")
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    series = [
        ("both_caught", "#0B5E5D", "caught by both"),
        ("recovered", CHUNK, "recovered by chunking"),
        ("lost", POSITIVE, "lost by chunking"),
        ("both_missed", "#CFC9C0", "missed by both"),
    ]
    left = np.zeros(len(table))
    for column, color, label in series:
        values = table[column].to_numpy(dtype=float)
        ax.barh(range(len(table)), values, left=left, height=0.66, color=color,
                edgecolor=PAPER, linewidth=1.4, label=label)
        for i, (v, s) in enumerate(zip(values, left, strict=True)):
            if v >= 4:
                ax.annotate(f"{int(v)}", (s + v / 2, i), ha="center", va="center",
                            fontsize=8.5, color=PAPER, fontweight="demibold")
        left = left + values

    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([_label(n) for n in table.index], fontsize=10.5)
    ax.set_xlabel("positive runs", fontsize=9.5, labelpad=8)
    ax.set_xlim(0, table["total"].max() * 1.02)
    for tick in (0, 10, 20, 30, 40, 50):
        if tick <= table["total"].max():
            ax.axvline(tick, color=HAIRLINE, lw=0.8, zorder=0)
    # Legend below the axes: the bars run the full width, so any in-axes
    # placement covers data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
              fontsize=9, handlelength=1.2, columnspacing=1.6)

    # Bars are per label, but a run carrying two labels appears in two rows, so
    # summing the columns over-counts. The headline uses run-level counts.
    paired_path = results / "paired.csv"
    if paired_path.is_file():
        paired = pd.read_csv(paired_path)
        positives = paired[paired["is_positive"]]
        missed = int((positives["transition"] == "both_missed").sum())
        total = len(positives)
    else:
        missed, total = int(table["both_missed"].sum()), int(table["total"].sum())
    _title(
        ax,
        "Outcome of every positive run, by label",
        f"{missed} of {total} positives are missed by both monitors — chunking converts\n"
        f"partial detection into near-complete detection, but the residual is dominated\n"
        f"by labels neither architecture reaches",
    )
    return _save(fig, out / f"positive_outcomes.{fmt}")


def failure_modes(out: Path, fmt: str, results: Path, baseline: Path,
                  threshold: float) -> Path | None:
    """What each arm's reasoning says on the runs it misses.

    Read from per_sample.csv, which carries the monitor's own rationale, so the
    categories are the monitor's stated grounds rather than an inference about
    why it failed.
    """
    paths = {"global": baseline / "per_sample.csv", "chunk": results / "per_sample.csv"}
    if not all(p.is_file() for p in paths.values()):
        return None

    themes = [
        ("no evidence of intent",
         r"no evidence of (intentional|deliberate)|no (clear )?(indication|sign)s? of intentional"),
        ("good-faith effort",
         r"good[- ]faith|genuine(ly)? (attempt|effort|trying)|sincere|earnest|diligent"),
        ("reasonable / appropriate", r"reasonable|appropriate|justified|legitimate"),
        ("cites an instruction", r"instruct|system prompt|told to|as directed"),
        ("blames the environment",
         r"environment|technical (limitation|issue|constraint)|resource constraint|timeout"),
        ("task looks complete", r"complet(ed|es) the task|successfully|accomplish"),
    ]
    shares: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for arm, path in paths.items():
        frame = pd.read_csv(path)
        fn = frame[(frame["is_positive"]) & (frame["score"] < threshold)]
        text = fn["reasoning"].fillna("")
        counts[arm] = len(fn)
        shares[arm] = [
            float(text.str.contains(pattern, case=False, regex=True).mean())
            for _, pattern in themes
        ]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = np.arange(len(themes))
    height = 0.36
    ax.barh(y + height / 2, shares["global"], height, color=GLOBAL,
            label=f"global ({counts['global']} misses)")
    ax.barh(y - height / 2, shares["chunk"], height, color=CHUNK,
            label=f"chunked ({counts['chunk']} misses)")
    for i, (g, c) in enumerate(zip(shares["global"], shares["chunk"], strict=True)):
        ax.annotate(f"{g:.0%}", (g, i + height / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=GLOBAL_DARK)
        ax.annotate(f"{c:.0%}", (c, i - height / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=CHUNK)

    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in themes], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        ax.axvline(tick, color=HAIRLINE, lw=0.8, zorder=0)
    ax.set_xlabel("share of that arm's false negatives whose rationale says this",
                  fontsize=9.5, labelpad=8)
    ax.legend(loc="lower right", fontsize=9, labelspacing=0.5, handlelength=1.2)
    _title(
        ax,
        "Stated rationale on false negatives, by arm",
        "categories overlap, so shares do not sum to 100% — both arms fail the same way,\n"
        "concluding the agent was making a good-faith attempt",
    )
    return _save(fig, out / f"failure_modes.{fmt}")


# --- length and calibration -------------------------------------------------


def _both_arms(results: Path, baseline: Path) -> dict[str, pd.DataFrame] | None:
    paths = {"global": baseline / "per_sample.csv", "chunk": results / "per_sample.csv"}
    if not all(p.is_file() for p in paths.values()):
        return None
    frames = {}
    for arm, path in paths.items():
        frame = pd.read_csv(path)
        frame["label"] = _primary_label(frame)
        frame["tokens"] = pd.to_numeric(frame["n_characters"], errors="coerce") / 4.4
        frames[arm] = frame[frame["tokens"].notna()]
    return frames


def recall_by_length(out: Path, fmt: str, results: Path, baseline: Path,
                     threshold: float) -> Path | None:
    """Detection rate against transcript length, within each label.

    Binned *within* label, not pooled: length is confounded with behaviour in
    MALT (medians span 24x), so a pooled curve would mostly trace which labels
    happen to be long. Terciles rather than deciles because a label holds ~49
    runs — bin counts are printed so the reader can see how thin they are.
    """
    frames = _both_arms(results, baseline)
    if frames is None:
        return None
    labels = [
        label for label in sorted(frames["global"]["label"].unique())
        if label != "benign" and (frames["global"]["label"] == label).sum() >= 15
    ]
    if not labels:
        return None

    ncols = 3
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()

    def _fmt(tokens: float) -> str:
        return f"{tokens / 1000:.0f}k" if tokens >= 1000 else f"{tokens:.0f}"

    for ax, label in zip(axes, labels, strict=False):
        base = frames["global"][frames["global"]["label"] == label]
        # Bins fixed from one arm so both plot at identical x positions; the runs
        # are paired, so the tercile medians are the same for both.
        edges = np.percentile(base["tokens"], [0, 33.3, 66.7, 100])
        edges[-1] += 1
        spans = list(zip(edges[:-1], edges[1:], strict=True))

        for arm, colour in (("global", GLOBAL_DARK), ("chunk", CHUNK)):
            frame = frames[arm][frames[arm]["label"] == label]
            xs, ys = [], []
            for i, (lo, hi) in enumerate(spans):
                inside = frame[(frame["tokens"] >= lo) & (frame["tokens"] < hi)]
                if inside.empty:
                    continue
                xs.append(i)
                ys.append(float((inside["score"] >= threshold).mean()))
            ax.plot(xs, ys, marker="o", ms=6, lw=2.2, color=colour,
                    solid_capstyle="round", label=arm, zorder=3)

        # Categorical x: only three points per panel, so a log axis bought
        # nothing and its minor ticks collided with the bin counts.
        ticks, counts = [], []
        for i, (lo, hi) in enumerate(spans):
            inside = base[(base["tokens"] >= lo) & (base["tokens"] < hi)]
            ticks.append(_fmt(float(inside["tokens"].median())) if len(inside) else "")
            counts.append(len(inside))
        ax.set_xticks(range(len(spans)))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_xlim(-0.35, len(spans) - 0.65)
        for i, n in enumerate(counts):
            ax.annotate(f"n={n}", (i, 1.02), ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)
        ax.set_ylim(-0.05, 1.16)
        ax.set_yticks([0, 0.5, 1.0])
        for tick in (0, 0.5, 1.0):
            ax.axhline(tick, color=HAIRLINE, lw=0.8, zorder=0)
        ax.set_title(_label(label), fontsize=10.5, color=INK, loc="left", pad=16)

    for ax in axes[len(labels):]:
        ax.set_visible(False)

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04), handlelength=1.6)
    fig.supxlabel("median transcript length of the tercile, in tokens",
                  fontsize=9.5, color=MUTED, y=0.02)
    fig.supylabel(f"recall at threshold {threshold:g}", fontsize=9.5, color=MUTED)
    fig.suptitle("Recall against transcript length, within each label",
                 fontsize=12.5, fontweight="demibold", color=INK, x=0.02,
                 ha="left", y=1.0)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.97))
    return _save(fig, out / f"recall_by_length.{fmt}")


def recall_by_length_combined(out: Path, fmt: str, results: Path, baseline: Path,
                              threshold: float) -> Path | None:
    """Every label on one axis: colour carries identity, line style carries arm.

    Plotted at each tercile's true median length rather than at slot 1/2/3, so
    the labels sit where they actually are on the length scale — which shows the
    length/behaviour confound directly (`partial_problem_solving` lives entirely
    to the right of `sabotage`).

    Composite encoding rather than 12 hues: 6 colours x 2 line styles. Colours are
    the reference categorical palette in fixed slot order, validated for the
    lightness band, chroma floor, CVD separation and normal-vision separation.
    Three of the six warn on contrast against this surface, so a legend is
    mandatory rather than optional — identity is never carried by colour alone.
    """
    frames = _both_arms(results, baseline)
    if frames is None:
        return None
    labels = sorted(
        label for label in frames["global"]["label"].unique()
        if label != "benign" and (frames["global"]["label"] == label).sum() >= 15
    )
    if not labels or len(labels) > len(CATEGORICAL):
        return None

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        ax.axhline(tick, color=HAIRLINE, lw=0.8, zorder=0)

    for slot, label in enumerate(labels):
        colour = CATEGORICAL[slot]
        base = frames["global"][frames["global"]["label"] == label]
        edges = np.percentile(base["tokens"], [0, 33.3, 66.7, 100])
        edges[-1] += 1
        spans = list(zip(edges[:-1], edges[1:], strict=True))
        for arm, dash, marker, fill in (
            ("global", (0, (5, 3)), "o", PAPER),
            ("chunk", None, "o", None),
        ):
            frame = frames[arm][frames[arm]["label"] == label]
            xs, ys = [], []
            for lo, hi in spans:
                inside = frame[(frame["tokens"] >= lo) & (frame["tokens"] < hi)]
                if inside.empty:
                    continue
                xs.append(float(inside["tokens"].median()))
                ys.append(float((inside["score"] >= threshold).mean()))
            ax.plot(xs, ys, ls=dash or "-", lw=2.0, color=colour, zorder=3,
                    marker=marker, ms=6.5, markerfacecolor=fill or colour,
                    markeredgecolor=colour, markeredgewidth=1.6,
                    solid_capstyle="round")
            # No end labels: at six series they collided at both extremes, and
            # direct labelling is for <=4 series — the legend carries identity here.

    ax.set_xscale("log")
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(1.4e3, 3.2e5)
    ax.set_xticks([2e3, 5e3, 1e4, 2e4, 5e4, 1e5])
    ax.set_xticklabels(["2k", "5k", "10k", "20k", "50k", "100k"])
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("median transcript length of the tercile, in tokens (log scale)",
                  fontsize=9.5, labelpad=8)
    ax.set_ylabel(f"recall at threshold {threshold:g}", fontsize=9.5)

    style_key = [
        plt.Line2D([], [], color=MUTED, lw=2.0, marker="o", ms=6.5,
                   markerfacecolor=MUTED, label="chunked"),
        plt.Line2D([], [], color=MUTED, lw=2.0, ls=(0, (5, 3)), marker="o", ms=6.5,
                   markerfacecolor=PAPER, markeredgecolor=MUTED, label="global"),
    ]
    colour_key = [
        plt.Line2D([], [], color=CATEGORICAL[i], lw=2.6, label=_label(name))
        for i, name in enumerate(labels)
    ]
    first = ax.legend(handles=style_key, loc="upper left", fontsize=9,
                      handlelength=2.4, labelspacing=0.4)
    ax.add_artist(first)
    ax.legend(handles=colour_key, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, fontsize=8.5, handlelength=1.8, columnspacing=1.6,
              labelspacing=0.4)

    _title(
        ax,
        "Recall against transcript length, all labels",
        f"terciles within each label · {len(labels)} behaviours · colour is the label, "
        f"line style the arm\nlabels sit at their true lengths, so the spread across "
        f"the x-axis is the length/behaviour confound",
    )
    return _save(fig, out / f"recall_by_length_combined.{fmt}")


def calibration(out: Path, fmt: str, results: Path, baseline: Path,
                threshold: float) -> Path | None:
    """Does a higher score mean a higher chance of being a real positive?

    Left: the standard reliability curve — bin by score, plot the share of runs
    in each bin that are genuinely positive. A monitor whose 0-100 output behaved
    like a probability would follow the diagonal.

    Right: whether confidence tracks competence *across behaviours* — median
    score on a label's positives against recall on that label. A label the
    monitor detects well but scores tepidly, or one it scores confidently but
    misses, shows up off the trend.
    """
    frames = _both_arms(results, baseline)
    if frames is None:
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.1))

    bins = [0, 1, 10, 25, 50, 75, 90, 101]
    for arm, colour in (("global", GLOBAL_DARK), ("chunk", CHUNK)):
        frame = frames[arm]
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            inside = frame[(frame["score"] >= lo) & (frame["score"] < hi)]
            if len(inside) < 5:
                continue
            xs.append(float(inside["score"].mean()))
            ys.append(float(inside["is_positive"].mean()))
            ns.append(len(inside))
        left.plot(xs, ys, marker="o", ms=6, lw=2.2, color=colour, label=arm,
                  solid_capstyle="round")
        # Global above the marker, chunk below: the two series sit close enough
        # that a shared offset made the counts unreadable.
        dy = 9 if arm == "global" else -15
        for x, y, n in zip(xs, ys, ns, strict=True):
            left.annotate(f"{n}", (x, y), xytext=(0, dy), textcoords="offset points",
                          ha="center", fontsize=7.5, color=colour)
    left.plot([0, 100], [0, 1], ls=(0, (4, 4)), lw=0.9, color=MUTED, zorder=0)
    left.axvline(threshold, color=HAIRLINE, lw=1.0, zorder=0)
    left.set_xlim(-4, 104)
    left.set_ylim(-0.05, 1.05)
    left.set_xlabel("monitor score (bin mean)", fontsize=9.5, labelpad=6)
    left.set_ylabel("share of runs that are genuinely positive", fontsize=9.5)
    left.set_yticks([0, 0.5, 1.0])
    for tick in (0, 0.5, 1.0):
        left.axhline(tick, color=HAIRLINE, lw=0.8, zorder=0)
    left.legend(fontsize=9, loc="lower right", handlelength=1.4)
    left.set_title("Overall reliability · labels are bin counts", fontsize=10,
                   loc="left", color=INK, pad=8)

    for arm, colour in (("global", GLOBAL_DARK), ("chunk", CHUNK)):
        frame = frames[arm]
        xs, ys, names = [], [], []
        for label, group in frame[frame["is_positive"]].groupby("label"):
            if len(group) < 15:
                continue
            xs.append(float((group["score"] >= threshold).mean()))
            ys.append(float(group["score"].median()))
            names.append(label)
        right.scatter(xs, ys, s=70, color=colour, edgecolor=PAPER, linewidth=1.4,
                      zorder=3, label=arm)
        if arm == "chunk":
            # Labels are placed once, on the chunked point. Several labels share
            # (0, 0) — both chance-level behaviours score and detect nothing — so
            # coincident points are stacked rather than printed on top of one another.
            placed: list[tuple[float, float]] = []
            for x, y, name in sorted(zip(xs, ys, names, strict=True)):
                offset = 0
                while any(abs(x - px) < 0.08 and abs(y + offset - py) < 7
                          for px, py in placed):
                    offset += 8
                placed.append((x, y + offset))
                right.annotate(_label(name), (x, y), xytext=(9, -3 + offset),
                               textcoords="offset points", fontsize=8, color=INK)
    right.axhline(threshold, color=HAIRLINE, lw=1.0, zorder=0)
    right.set_xlim(-0.05, 1.15)
    right.set_ylim(-5, 105)
    right.set_xlabel(f"recall at threshold {threshold:g}", fontsize=9.5, labelpad=6)
    right.set_ylabel("median score on that label's positives", fontsize=9.5)
    for tick in (0, 25, 50, 75, 100):
        right.axhline(tick, color=HAIRLINE, lw=0.8, zorder=0)
    right.legend(fontsize=9, loc="upper left", handlelength=1.4)
    right.set_title("Confidence against competence, per label", fontsize=10,
                    loc="left", color=INK, pad=8)

    fig.suptitle("Score calibration", fontsize=12.5, fontweight="demibold",
                 color=INK, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, out / f"calibration.{fmt}")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override the figures directory")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    setup_logging()
    _style()
    cfg = ExperimentConfig.from_yaml(args.config)
    chunk_dir = ExperimentRun(cfg, stage="chunk").results_path.parent
    global_dir = ExperimentRun(cfg, stage="global").results_path.parent
    out = args.out_dir or (chunk_dir.parent / "figures")
    out.mkdir(parents=True, exist_ok=True)

    written = [
        # exploration — what the dataset looks like before any monitor runs
        length_by_label(out, args.format, global_dir),
        run_source(out, args.format, global_dir),
        length_composition(out, args.format, chunk_dir),
        # results
        recall_by_label(out, args.format, chunk_dir),
        aggregation(out, args.format, chunk_dir),
        score_scatter(out, args.format, chunk_dir, args.threshold),
        # failure modes
        positive_outcomes(out, args.format, chunk_dir),
        failure_modes(out, args.format, chunk_dir, global_dir, args.threshold),
        # length and calibration
        recall_by_length(out, args.format, chunk_dir, global_dir, args.threshold),
        recall_by_length_combined(out, args.format, chunk_dir, global_dir,
                                  args.threshold),
        calibration(out, args.format, chunk_dir, global_dir, args.threshold),
    ]
    made = [p for p in written if p]
    if not made:
        print("No figures written — run scripts/analyze.py and scripts/compare.py first.")
        return 1
    print(f"\nWrote {len(made)} figures to {out}")
    for path in made:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
