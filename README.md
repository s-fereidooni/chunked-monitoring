# Chunk-wise monitoring

Does monitoring an agent transcript in **chunks** detect sandbagging better than monitoring
the **whole transcript** at once — and if so, why?

METR's MALT baseline already chunks ("the transcript is chunked, with the first five
messages repeated across all chunks for context about the task") but publishes no ablation.
SHADE-Arena, Redwood's control work and most of the surrounding literature monitor whole
transcripts. Nobody has published what chunking buys, on which behaviours, or under which
aggregation rule.

This is a research prototype, not a production monitoring system. The deliverable is a
LessWrong post; scope was kept deliberately small.

## Result

554 MALT runs — 261 sandbagging across six behaviour labels, 293 benign — each scored twice
by `gpt-4.1-mini`, once whole and once in 20-message chunks aggregated by max.

| | global | chunk |
|---|---|---|
| ROC AUC | 0.761 | **0.821** |
| recall @ score ≥ 50 | 0.303 | **0.621** |
| FPR @ score ≥ 50 | 0.007 | 0.007 |
| runs caught at ≤5% FPR | 0.556 | **0.651** |

McNemar's exact test on the discordant pairs: **84 recovered, 1 lost**, p < 0.0001.

ROC AUC is the number to quote. "TPR at 5% FPR" is *not* a matched comparison here — the
score grid is coarse enough that the arms land at 4.8% and 1.4% actual FPR, so chunking is
held to under a third of the budget global gets.

The gain is concentrated: `partial_problem_solving` 0.149 → 0.936 and `match_weaker_model`
0.306 → 0.898, while `reasoning_about_task` and `gives_up` stay at exactly 0.000 in both
arms. See [`docs/experiments.md`](docs/experiments.md) for the full log, including what does
*not* replicate and which mechanism questions remain open.

## Read this before trusting any of it

Ten bugs reached results or tooling undetected over the course of this work, and **not one
was caught by the test suite or by the metrics.** Five were found by reading raw data, five
by independent review — the last of them in a codebase whose own audit was reporting green.
Among them: 32% of the corpus silently duplicated, a config flag that never reached the
loader so a "successful" pass made zero API calls, and a length statistic that omitted tool
arguments and so biased the length control along the axis it existed to control for.

The cause is structural. Test fixtures encode the *author's* model of the data, so when
that model is wrong the fixture is wrong the same way and the test agrees with the bug.
Metrics cannot help either: a monitor scoring a corrupted transcript still returns a
plausible number.

`scripts/audit_pipeline.py` exists because of that — it compares pipeline output against
the source parquet rather than against expectations. Note that its own first version was
blind to the bug it was written for, comparing sets where multiplicity was the thing that
mattered. Treat it as code to review, not as an oracle.

[`docs/experiments.md`](docs/experiments.md) logs every run in order, including the claims
that were **retracted** when better inputs contradicted them.

## Setup

```bash
uv sync --all-extras
cp .env.example .env    # then fill in
```

`OPENAI_API_KEY` is needed for the monitor. Responses are content-addressed and cached under
`.cache/responses`, so re-running a scored subset costs nothing.

### Dataset access

[MALT](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/)
(Manually-reviewed Agentic Labeled Transcripts) by METR — we use the public variant,
[`metr-evals/malt-public`](https://huggingface.co/datasets/metr-evals/malt-public): 7,179
runs over 30 task families.

It is **gated**. Two steps:

1. Accept the terms on the dataset page while signed in to Hugging Face.
2. Supply a token via `HF_TOKEN` in `.env`, or `huggingface-cli login`.

~3.6 GB compressed / ~18.6 GB on disk. Transcripts are never committed here — `data/evaluation_subset.json`
holds the 554 run IDs and the config that selected them, which makes the subset reproducible
without redistributing gated data.

## Reproducing

Every script takes `--config`. **`fixed` is the current one** and is the default;
`stripped` and `default` are earlier runs kept only for comparison and are stale — `default`
still feeds MALT's elicitation prompt to the monitor, which inflates its numbers.

```bash
uv run python scripts/build_subset.py                      # -> data/evaluation_subset.json
uv run python scripts/run_monitor.py --monitor global      # -> results/<exp>/global/
uv run python scripts/run_monitor.py --monitor chunk       # -> results/<exp>/chunk/
uv run python scripts/analyze.py --stage global            # metrics, per-sample, failures
uv run python scripts/analyze.py --stage chunk
uv run python scripts/compare.py                           # paired McNemar + length controls
uv run python scripts/figures.py                           # 18 figures
```

Checks, both worth running before believing a results directory:

```bash
uv run python scripts/audit_pipeline.py    # 22 data-level checks against the raw parquet
uv run python scripts/audit_leakage.py     # features that predict the label by themselves
uv run python scripts/show_run.py 329102   # one transcript + both arms' verdicts
```

`results/sandbagging-fixed/figures/` is tracked, so the figures are readable without
dataset access or an API key. The CSVs and `results.jsonl` are not: they quote monitor
reasoning derived from gated MALT transcripts. Everything else under `results/` is
reproducible from the tracked subset plus the configs.

## Layout

```
configs/                      experiment configs — fixed.yaml is current
data/evaluation_subset.json   tracked: the 554 run IDs and how they were chosen
docs/experiments.md           experiment log, newest last, with retractions
results/sandbagging-fixed/figures/   tracked: 18 figures for the current run
src/monitor_localization/
  dataset/                    parquet -> linear transcript, schema, subset construction
  monitors/                   global + chunk monitors, chunking, aggregation
  prompts/metr/               METR's monitor prompts, verbatim + SOURCE.txt
  evaluation/                 metrics, paired comparison, length controls
  llm/                        client + content-addressed response cache
scripts/                      entry points, one per pipeline stage
tests/                        178 tests
```

## Deliberately out of scope

Localization annotations, hierarchical/hybrid/adaptive monitoring, synthetic benchmarks,
multiple monitor models, multiple datasets. The codebase is structured so these drop in
without rework.

Two open questions are worth naming. A **shuffled-chunk control** (same length, same chunk
count, order destroyed) has never been run, and it is the only thing that would separate
"shorter context helps" from "more draws help". And METR's prompt always says "a part of a
transcript" because their baseline is always chunked, so our global arm's "a transcript" is
one phrase of prompt difference on top of the chunking — see
[`prompts/metr/SOURCE.txt`](src/monitor_localization/prompts/metr/SOURCE.txt).

## Acknowledgements

Both the dataset and the monitor prompts are METR's.

**Dataset:** [MALT](https://huggingface.co/datasets/metr-evals/malt-public) — see the
dataset page for its terms.

**Prompts:** the four files in `src/monitor_localization/prompts/metr/` are reproduced
verbatim from METR's
[MALT release](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/),
deliberately rather than rewritten — the experiment measures what chunking does to METR's
baseline monitor, so a prompt of our own would confound it with prompt engineering.
[`prompts/metr/SOURCE.txt`](src/monitor_localization/prompts/metr/SOURCE.txt) records the five
places our copy deviates, including one known asymmetry between the arms that is not yet
tested.

Portions of this codebase were written with Claude Opus 5 via Claude Code.
