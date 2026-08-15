# Chunk-wise monitoring

Monitoring long agent transcripts **in chunks**, rather than all at once, catches
sandbagging that whole-transcript monitoring misses.

Each of 554 MALT transcripts (261 sandbagging, 293 benign) is scored twice by
`gpt-4.1-mini`: once whole, and once in 20-message chunks with the run's score taken as
the max over chunks. Chunking recovers **84** runs the global monitor missed and loses
**1** (McNemar p < 0.0001); ROC AUC goes 0.761 → 0.821.

The gain is concentrated by behaviour — of the runs the global monitor missed, chunking
recovers 92% of `partial_problem_solving`, 88% of `match_weaker_model` and 45% of
`refusals`, while `reasoning_about_task` and `gives_up` stay at zero recall in **both**
arms.

📝 **Blog post:** _(link to follow)_

Figures are in [`results/sandbagging-fixed/figures/`](results/sandbagging-fixed/figures);
[`docs/experiments.md`](docs/experiments.md) is the full run log, including claims that
were retracted when better inputs contradicted them.

## Setup

```bash
uv sync --all-extras
cp .env.example .env    # OPENAI_API_KEY, HF_TOKEN
```

MALT is **gated**: accept the terms on the
[dataset page](https://huggingface.co/datasets/metr-evals/malt-public) while signed in to
Hugging Face, then supply a token. Transcripts are never committed here —
`data/evaluation_subset.json` holds the 554 run IDs and the config that selected them, so
the subset is reproducible without redistributing gated data.

## Running

All scripts default to `--config fixed`, the corrected run. `stripped` and `default` are
earlier runs kept only so the log stays reproducible; both are stale.

```bash
uv run python scripts/build_subset.py                   # -> data/evaluation_subset.json
uv run python scripts/run_monitor.py --monitor global
uv run python scripts/run_monitor.py --monitor chunk
uv run python scripts/analyze.py --stage global         # metrics, per-sample, failures
uv run python scripts/analyze.py --stage chunk
uv run python scripts/compare.py                        # paired McNemar + length controls
uv run python scripts/figures.py                        # 18 figures
```

Worth running before believing any results directory:

```bash
uv run python scripts/audit_pipeline.py   # 22 checks of output against the raw parquet
uv run python scripts/audit_leakage.py    # features that predict the label by themselves
uv run python scripts/show_run.py 329102  # one transcript + both arms' verdicts
```

Ten bugs reached results or tooling undetected during this work and **none was caught by
the test suite or by the metrics** — test fixtures encode the author's model of the data,
so when that model is wrong the fixture is wrong the same way. `audit_pipeline.py` exists
for that reason, and compares output against the source parquet rather than against
expectations.

## Credits

**Dataset and prompts are METR's.**
[MALT](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/)
(Manually-reviewed Agentic Labeled Transcripts) supplies the transcripts and labels; the
four files in [`src/monitor_localization/prompts/metr/`](src/monitor_localization/prompts/metr)
are METR's monitor prompts reproduced verbatim, deliberately rather than rewritten, so the
comparison measures chunking and not prompt engineering.
[`SOURCE.txt`](src/monitor_localization/prompts/metr/SOURCE.txt) records the five places
our copy deviates, including one known asymmetry between the arms.

Produced during a 30-hour AI Safety Technical Project Sprint facilitated by
**BlueDot Impact**. Thanks to **Adrians Skapars** for mentorship, feedback and helpful
discussions.

Portions of this codebase were written with Claude Opus 5 via Claude Code.
