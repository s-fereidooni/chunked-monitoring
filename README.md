# Monitor Localization

Research prototype investigating whether **localized (chunk-wise) monitoring** can recover
blind spots of **global LLM monitors**.

> Which monitor blind spots can chunk-wise monitoring recover?

The goal is not a production monitoring system but a clean experimental framework for
comparing monitoring architectures. First milestone: a LessWrong post.

See [`notes.md`](notes.md) for the full research plan and
[`docs/experiments.md`](docs/experiments.md) for the experiment log.

## Setup

```bash
uv venv --python 3.13
uv pip install -e ".[analysis,dev]"
cp .env.example .env    # then fill in
```

### Dataset access

The dataset is [MALT](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/)
(Manually-reviewed Agentic Labeled Transcripts) from METR — 10,919 agent transcripts labeled
with behaviors that threaten evaluation integrity. We use the public variant,
[`metr-evals/malt-public`](https://huggingface.co/datasets/metr-evals/malt-public):
7,179 runs over 30 task families and 169 tasks.

It is **gated**. Two steps are required:

1. Accept the terms on the dataset page while signed in to Hugging Face.
2. Provide a token, either via `HF_TOKEN` in `.env` or `huggingface-cli login`.

The full split is ~3.6 GB compressed / ~18.6 GB on disk.

## Layout

```
notes.md          research plan
configs/          experiment configs (YAML)
analysis/         notebooks
data/
  raw/            HF cache (gitignored)
  processed/      derived artifacts (gitignored)
  evaluation_subset.json   tracked — defines the reproducible eval set
docs/             experiment log
results/          per-experiment outputs (gitignored)
src/monitor_localization/
  dataset/        loading, schema, subset construction
  monitors/       global + chunk-wise monitors, chunking, aggregation
  prompts/        versioned monitor prompts
  evaluation/     metrics, recovery analysis
  utils/          logging, I/O
scripts/          entry points
tests/
```

## Pipeline

| Stage | What | Output |
|---|---|---|
| 1 | Dataset exploration, build eval subset | `analysis/dataset_report.ipynb`, `data/evaluation_subset.json` |
| 2 | Global monitor (whole transcript → one judgment) | `results/<exp>/global/` |
| 3 | Global failure analysis (confusion matrix, P/R/F1, per-sample CSV) | `results/<exp>/global/metrics.json` |
| 4 | Chunk-wise monitor (split → judge each → aggregate) | `results/<exp>/chunk/` |
| 5 | Recovery analysis (global FN → chunk TP) | `results/<exp>/recovery/` |

Deliberately deferred: localization annotations, hierarchical/hybrid/adaptive monitoring,
synthetic benchmarks, multiple monitor models, multiple datasets. The codebase is structured
so these drop in without rework.

## Acknowledgements

Dataset: [MALT](https://huggingface.co/datasets/metr-evals/malt-public) by METR, MIT licensed.

Portions of this codebase were written with Claude Opus 5 via Claude Code.
