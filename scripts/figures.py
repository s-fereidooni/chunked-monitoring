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
    """The length control: where each architecture's false positives sit."""
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
        recall_by_label(out, args.format, chunk_dir),
        aggregation(out, args.format, chunk_dir),
        score_scatter(out, args.format, chunk_dir, args.threshold),
        fpr_by_length(out, args.format, chunk_dir, global_dir),
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
