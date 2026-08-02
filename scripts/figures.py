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
    """Headline states the finding; subtitle carries the caveat or the n.

    Pad scales with the subtitle's line count — a fixed pad lets a two-line
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

    _title(ax, "Chunking moves three behaviours and leaves two at the floor",
           "recall on 261 sandbagging runs · gpt-4.1-mini monitor · paired design")
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

    _title(
        ax,
        "Two failure modes hide behind one number",
        "mean cannot exploit extra draws — where it collapses, evidence was local;\n"
        "where it survives, the gain is something else entirely",
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
    ax.legend(loc="upper left", bbox_to_anchor=(0.50, 0.42), fontsize=9.5,
              handletextpad=0.3, markerscale=1.7, labelspacing=0.5)

    _title(ax, "Every run, scored twice",
           f"{len(paired)} transcripts · threshold {threshold:g} · McNemar p < 0.0001")
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
        "Chunking redistributes false positives rather than removing them",
        "293 benign runs · 2 false positives in each arm — the shape is real, "
        "the counts are small",
    )
    return _save(fig, out / f"fpr_by_length.{fmt}")


# --- data exploration -------------------------------------------------------


def _primary_label(frame: pd.DataFrame) -> pd.Series:
    return frame["labels"].fillna("normal").str.split(";").str[0].replace(
        {"normal": "benign"}
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
        "Length is confounded with behaviour",
        "median spans 24x across labels — so any per-label comparison is partly\n"
        "a comparison of transcript lengths",
    )
    return _save(fig, out / f"length_by_label.{fmt}")


def length_composition(out: Path, fmt: str, results: Path) -> Path | None:
    """Which behaviors sit in each length quartile — why pooled splits mislead."""
    path = results / "label_composition_by_length.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path).set_index("label")
    table = table[table.sum(axis=1) >= 5]
    shares = table.div(table.sum(axis=0), axis=1)
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
        "Length quartiles hold different behaviours, not the same one at "
        "different lengths",
        "which is why a gain that grows with length is not evidence that length "
        "is what matters",
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
        "Prompting is almost perfectly confounded with behaviour",
        "every benign run is natural and five of six positive labels are wholly\n"
        "prompted — so “was an instruction injected” is available as a shortcut",
    )
    return _save(fig, out / f"run_source.{fmt}")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    setup_logging()
    _style()
    cfg = ExperimentConfig.from_yaml(args.config)
    chunk_dir = ExperimentRun(cfg, stage="chunk").results_path.parent
    global_dir = ExperimentRun(cfg, stage="global").results_path.parent
    out = chunk_dir.parent / "figures"
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
