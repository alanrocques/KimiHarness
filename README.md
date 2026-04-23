# mini-meta-harness

A budget reproduction of **Meta-Harness**
([arXiv:2603.28052](https://arxiv.org/abs/2603.28052)) that swaps the paper's
Claude Opus 4.6 proposer for **Kimi K2.6** via Moonshot's
Anthropic-compatible API. 12 iterations on Symptom2Disease for under $15.

The full build spec lives in [`docs/SPEC.md`](docs/SPEC.md). Open items are in
[`QUESTIONS.md`](QUESTIONS.md).

## TL;DR

Meta-Harness is an outer loop that searches over **harness code** — the
Python wrapper around an LLM that decides what to prompt, retrieve, and
parse. A coding-agent proposer has filesystem access to every prior harness,
its score, and its raw per-example traces; it reads them and writes a better
one. This repo reimplements that loop small enough to fit a portfolio budget,
with clean on-disk traces a downstream visualizer can consume.

## Quickstart

```bash
uv sync --extra dev --extra notebook          # install
cp .env.example .env && $EDITOR .env          # MOONSHOT_API_KEY=...
uv run mmh run --iterations 3 --mock          # end-to-end, no API calls
uv run mmh run --iterations 12                # the real thing (~$10–15)
```

Then open `notebooks/explore_run.ipynb` to see the accuracy chart, cost
breakdown, best harness, and baseline-vs-best confusion matrix.

## CLI

```bash
mmh run --iterations 12                         # full run
mmh run --iterations 3 --mock                   # canned responses, no API
mmh run --iterations 12 --target-model claude-haiku-4-5  # cross-provider
mmh summary runs/2026-04-23_14-30-00            # print a past run
mmh cost runs/2026-04-23_14-30-00               # cost breakdown
```

## How it works

Iteration 0 evaluates a fixed **zero-shot baseline harness**
(`src/mini_meta_harness/harnesses/baseline_zero_shot.py`) over 100 held-out
Symptom2Disease examples and writes the trace to
`runs/<id>/iterations/000/`.

Iterations 1..N invoke a **tool-using proposer agent**
(`src/mini_meta_harness/proposer.py`). It has seven tools: `list_iterations`,
`read_harness`, `read_reasoning`, `read_eval_trace(only_failures=True)`,
`read_score`, `scoreboard`, and `write_harness(code, reasoning)`. The agent
reads the scoreboard, inspects raw failure traces from the best prior
iteration, then submits a new single-file Python harness. The evaluator runs
it, records a new trace, and the loop continues.

Each iteration writes `harness.py`, `reasoning.md`,
`filesystem_reads.jsonl`, `eval_trace.jsonl`, and `score.json`. That on-disk
layout is a **public contract** — bump `schema_version` on `RunConfig`
before changing field names.

## On-disk layout

```
runs/<run_id>/
├── config.json
├── cost.json
├── summary.json
└── iterations/
    ├── 000/
    │   ├── harness.py
    │   ├── reasoning.md
    │   ├── filesystem_reads.jsonl
    │   ├── eval_trace.jsonl
    │   └── score.json
    └── 001/ ...
```

## Differences from the paper

| | paper | this repo |
|---|---|---|
| Proposer | Claude Opus 4.6 + Claude Code | Kimi K2.6 (`kimi-k2.6` via Moonshot) + our tool-use loop |
| Target | Claude | Kimi K2.6 by default; `--target-model claude-haiku-4-5` for cross-provider |
| Context budget | ~50× larger | single-node, 100 eval examples |
| Benchmarks | Terminal-Bench 2.0 + text classification | Symptom2Disease only |
| Harness shape | single-file Python | same |
| Memory | filesystem of past iterations + raw traces | same |

## Mock mode

`--mock` replaces both clients with deterministic stand-ins
(`src/mini_meta_harness/mock.py`) so the whole pipeline runs in seconds
without a Moonshot key. CI uses it as an end-to-end smoke test.

## Tests

```bash
uv run pytest          # unit tests
uv run ruff check .    # lint
```

## Citation

```bibtex
@misc{meta-harness-2026,
  title  = {Meta-Harness: Searching over LLM Harness Code with Raw Execution Traces},
  author = {Stanford IRIS Lab},
  year   = {2026},
  eprint = {2603.28052},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

Reference implementation: <https://github.com/stanford-iris-lab/meta-harness>.

## License

MIT — see [LICENSE](LICENSE).
