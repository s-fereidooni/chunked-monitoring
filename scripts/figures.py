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
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from monitor_localization.config import ExperimentConfig  # noqa: E402
from monitor_localization.evaluation import DEFAULT_THRESHOLD  # noqa: E402
from monitor_localization.experiment import ExperimentRun  # noqa: E402
from monitor_localization.utils import setup_logging  # noqa: E402

GLOBAL_COLOR = "#8c8c8c"
CHUNK_COLOR = "#1f77b4"
MEAN_COLOR = "#c44e52"
POSITIVE_COLOR = "#c44e52"
BENIGN_COLOR = "#9fb8c8"

# Shortened for axis labels; the tables keep the full names.
SHORT = {
    "match_weaker_model": "match weaker\nmodel",
    "partial_problem_solving": "partial problem\nsolving",
    "reasoning_about_task": "reasoning about\ntask",
    "bypass_constraints": "bypass\nconstraints",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
        }
    )


def _label(name: str) -> str:
    return SHORT.get(name, name.replace("_", " "))


def recall_by_label(out: Path, fmt: str, results: Path) -> Path | None:
    """Headline: which behaviors chunking moves, and which it does not."""
    path = results / "recall_by_aggregation.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    table = table[table["is_positive"]].sort_values("chunk_max")
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = range(len(table))
    height = 0.38
    ax.barh([i + height / 2 for i in y], table["global"], height,
            label="global", color=GLOBAL_COLOR)
    ax.barh([i - height / 2 for i in y], table["chunk_max"], height,
            label="chunked (max)", color=CHUNK_COLOR)
    ax.set_yticks(list(y))
    ax.set_yticklabels([_label(name) for name in table["label"]])
    ax.set_xlabel("recall at threshold 50")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Chunking moves three labels and leaves two untouched", loc="left")
    for i, (g, c) in enumerate(zip(table["global"], table["chunk_max"], strict=True)):
        if c - g > 0.05:
            ax.annotate(f"+{c - g:.2f}", (c, i - height / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=CHUNK_COLOR)
    return _save(fig, out / f"recall_by_label.{fmt}")


def aggregation(out: Path, fmt: str, results: Path) -> Path | None:
    """The mechanism figure: mean is immune to draw count, so it separates them."""
    path = results / "recall_by_aggregation.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    table = table[table["is_positive"] & (table["chunk_max"] > 0.05)]
    table = table.sort_values("chunk_max", ascending=False)
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = range(len(table))
    width = 0.27
    ax.bar([i - width for i in x], table["global"], width,
           label="global", color=GLOBAL_COLOR)
    ax.bar(list(x), table["chunk_max"], width,
           label="chunked, max", color=CHUNK_COLOR)
    ax.bar([i + width for i in x], table["chunk_mean"], width,
           label="chunked, mean", color=MEAN_COLOR)
    # A zero bar is invisible and reads as missing data, so label it explicitly —
    # mean collapsing to 0.00 is a result, not an absent measurement.
    for i, value in enumerate(table["chunk_mean"]):
        if value < 0.02:
            ax.annotate("0.00", (i + width, 0.01), ha="center", va="bottom",
                        fontsize=8, color=MEAN_COLOR, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_label(name) for name in table["label"]])
    ax.set_ylabel("recall at threshold 50")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        "Mean aggregation cannot exploit extra draws — so where it still wins,\n"
        "the gain is not a draw-count artifact",
        loc="left",
    )
    return _save(fig, out / f"aggregation.{fmt}")


def score_scatter(out: Path, fmt: str, results: Path, threshold: float) -> Path | None:
    """Every run, both scores. The recoveries are the points above the line."""
    path = results / "paired.csv"
    if not path.is_file():
        return None
    paired = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    jitter = 1.6
    rng = __import__("numpy").random.default_rng(0)
    for mask, color, name in (
        (~paired["is_positive"], BENIGN_COLOR, "benign"),
        (paired["is_positive"], POSITIVE_COLOR, "sandbagging"),
    ):
        group = paired[mask]
        ax.scatter(
            group["score_baseline"] + rng.normal(0, jitter, len(group)),
            group["score_variant"] + rng.normal(0, jitter, len(group)),
            s=13, alpha=0.55, color=color, label=name, linewidths=0,
        )
    ax.plot([0, 100], [0, 100], color="#333", lw=0.7, ls="--", zorder=0)
    ax.axhline(threshold, color="#333", lw=0.7, alpha=0.5)
    ax.axvline(threshold, color="#333", lw=0.7, alpha=0.5)
    ax.set_xlabel("global monitor score")
    ax.set_ylabel("chunked monitor score")
    ax.set_xlim(-6, 106)
    ax.set_ylim(-6, 106)
    ax.set_aspect("equal")

    recovered = int((paired["transition"] == "recovered").sum())
    lost = int((paired["transition"] == "lost").sum())
    ax.annotate(f"{recovered} recovered", (4, 96), fontsize=9, color=CHUNK_COLOR,
                va="top", fontweight="bold")
    ax.annotate(f"{lost} lost", (96, 4), fontsize=9, color="#555",
                ha="right", fontweight="bold")
    ax.legend(frameon=False, loc="center right", fontsize=8)
    ax.set_title("Paired scores, one point per run", loc="left")
    return _save(fig, out / f"score_scatter.{fmt}")


def fpr_by_length(out: Path, fmt: str, results: Path, baseline: Path) -> Path | None:
    """The length control: where each architecture's false positives sit."""
    paths = {"chunked": results / "fpr_by_length_decile.csv",
             "global": baseline / "fpr_by_length_decile.csv"}
    if not all(p.is_file() for p in paths.values()):
        return None
    frames = {name: pd.read_csv(p) for name, p in paths.items()}

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for name, color in (("global", GLOBAL_COLOR), ("chunked", CHUNK_COLOR)):
        frame = frames[name]
        ax.plot(frame["length_decile"], frame["fpr"], marker="o", ms=4,
                color=color, label=name)
    ax.set_xlabel("transcript length decile (benign runs only)")
    ax.set_ylabel("false-positive rate")
    ax.set_xticks(range(10))
    ax.legend(frameon=False)
    ax.set_title(
        "False positives concentrate in the longest transcripts for both arms",
        loc="left",
    )
    median = frames["global"]["median_tokens"]
    ax.set_xticklabels(
        [f"{int(t / 1000)}k" if t >= 1000 else str(int(t)) for t in median],
        fontsize=7,
    )
    ax.set_xlabel("median tokens in decile (benign runs only)")
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
