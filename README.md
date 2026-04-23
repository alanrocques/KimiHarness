# MiniMetaHarness — Build Specification

**You are building a budget-friendly reproduction of the Meta-Harness paper ([arXiv:2603.28052](https://arxiv.org/abs/2603.28052)) using Kimi K2.6 (via Moonshot's Anthropic-compatible API) as the proposer, instead of the paper's Claude Opus 4.6.**

Read this entire document before writing any code. The priority order of concerns is: correctness → faithful-to-paper semantics → clean trace export → cost control → polish.

---

## 0. Context: what the paper does

Meta-Harness is an outer loop that **searches over "harness" code** — the Python code that wraps an LLM and decides what to store, retrieve, and show. A coding-agent proposer has filesystem access to every prior candidate harness, its score, and its raw execution traces, and iteratively rewrites the harness. The paper's headline empirical claim: using raw traces (not summaries) as memory, the loop discovers harnesses that beat hand-tuned baselines by meaningful margins on text classification and Terminal-Bench 2.0.

The reference implementation is at [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness), specifically `reference_examples/text_classification/`. **Read that directory before starting.** We are building a sibling project, not a fork — closer in spirit to "rewrote the reference for clarity, swapped in an open-weight proposer, added a clean trace export format."

## 1. Goals and non-goals

### Goals
1. Run 12 iterations of Meta-Harness on Symptom2Disease (100-example held-out slice) for under $15.
2. Use Kimi K2.6 (model id `kimi-k2-0711-preview` or latest available on Moonshot) as the proposer.
3. Use Kimi K2.6 again (or Claude Haiku 4.5 if the user configures it) as the *target* model being harnessed.
4. Export every iteration as structured JSON on disk, in a schema designed to be consumed by a downstream visualizer.
5. Ship a Jupyter notebook that loads a completed run and makes the before/after story obvious.
6. Provide a `--mock` mode that runs the full pipeline without any API calls, using canned responses, so contributors can exercise the code path for free.

### Non-goals
- Match the paper's absolute accuracy numbers (their context budget is ~50x ours).
- Run on Terminal-Bench 2.0 or any agentic coding benchmark.
- Multi-file harness programs — keep harnesses as single-file Python programs, same as the paper.
- Web UI (that's a sibling project, #4 in the portfolio plan).

## 2. Environment

- macOS, Python 3.11+, `uv` for dependency management, Claude Code as the IDE agent running against this spec.
- Moonshot API key in `.env` as `MOONSHOT_API_KEY`.
- Moonshot's Anthropic-compatible endpoint: `https://api.moonshot.ai/anthropic`.
- **Important**: use the official `anthropic` Python SDK and point its `base_url` at Moonshot. Do not roll a custom HTTP client. Example:
  ```python
  from anthropic import Anthropic
  client = Anthropic(api_key=os.environ["MOONSHOT_API_KEY"], base_url="https://api.moonshot.ai/anthropic")
  ```

## 3. Repository layout

```
mini-meta-harness/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE                       # MIT
├── src/mini_meta_harness/
│   ├── __init__.py
│   ├── types.py                  # Pydantic schemas (see §5)
│   ├── config.py                 # Env loading, model pricing
│   ├── dataset.py                # Symptom2Disease loader
│   ├── evaluator.py              # Runs a candidate harness on the eval set
│   ├── proposer.py               # The coding-agent proposer loop
│   ├── memory.py                 # Filesystem-backed memory interface
│   ├── outer_loop.py             # The Meta-Harness outer loop
│   ├── cost.py                   # Token/dollar accounting
│   ├── harnesses/                # Baseline + any human-written harnesses
│   │   ├── __init__.py
│   │   └── baseline_zero_shot.py
│   └── run.py                    # Typer CLI entry point
├── notebooks/
│   └── explore_run.ipynb         # Load a run, chart it
├── runs/                         # Gitignored; one subdir per run
│   └── .gitkeep
├── data/                         # Gitignored; dataset cache
│   └── .gitkeep
├── tests/
│   ├── test_evaluator.py
│   ├── test_memory.py
│   └── test_mock_proposer.py
└── scripts/
    └── print_run_summary.py      # CLI helper to print a run's summary.json
```

## 4. Dependencies (pyproject.toml)

Runtime:
- `anthropic>=0.40.0` — used against Moonshot's compatible endpoint
- `datasets>=3.0.0` — load Symptom2Disease
- `pydantic>=2.9.0`
- `python-dotenv>=1.0.1`
- `rich>=13.9.0` — pretty CLI
- `tenacity>=9.0.0` — retries
- `tiktoken>=0.8.0` — rough token counting
- `typer>=0.12.0`

Optional `[notebook]` extra: `jupyter`, `matplotlib`, `pandas`, `seaborn`.

Optional `[dev]` extra: `pytest`, `pytest-asyncio`, `ruff`.

Entry point: `mmh = "mini_meta_harness.run:app"`.

## 5. Data model (`src/mini_meta_harness/types.py`)

All Pydantic v2. Include `schema_version: str = "0.1.0"` on `RunConfig` — downstream tools will key off this.

```python
class RunConfig(BaseModel):
    run_id: str                    # e.g. "2026-04-23_14-30-00"
    iterations: int
    dataset: str = "symptom2disease"
    eval_split_size: int = 100
    proposer_model: str
    target_model: str
    mock: bool = False
    started_at: datetime
    schema_version: str = "0.1.0"

class FilesystemRead(BaseModel):
    path: str
    bytes_read: int
    at: datetime

class EvalExample(BaseModel):
    example_id: int
    input: str
    gold_label: str
    predicted_label: str | None
    correct: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None = None

class IterationScore(BaseModel):
    accuracy: float
    n_examples: int
    n_correct: int
    n_errors: int
    mean_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int

class CostBreakdown(BaseModel):
    proposer_input_tokens: int = 0
    proposer_output_tokens: int = 0
    target_input_tokens: int = 0
    target_output_tokens: int = 0
    estimated_usd: float = 0.0

class Iteration(BaseModel):
    index: int                        # 1-indexed
    harness_code: str                 # Full Python source
    reasoning: str                    # Proposer's free-text rationale
    filesystem_reads: list[FilesystemRead]
    score: IterationScore
    cost: CostBreakdown
    started_at: datetime
    finished_at: datetime

class RunSummary(BaseModel):
    config: RunConfig
    iterations: list[Iteration]
    total_cost: CostBreakdown
    best_iteration_index: int
    finished_at: datetime
```

## 6. On-disk trace format (the contract)

Every run creates `runs/<run_id>/`:

```
runs/2026-04-23_14-30-00/
├── config.json              # serialized RunConfig
├── summary.json             # serialized RunSummary (written at end)
├── cost.json                # running CostBreakdown, updated every iteration
├── iterations/
│   ├── 001/
│   │   ├── harness.py           # the proposed harness, executable as a module
│   │   ├── reasoning.md         # the proposer's reasoning, markdown
│   │   ├── filesystem_reads.jsonl   # one FilesystemRead per line
│   │   ├── eval_trace.jsonl         # one EvalExample per line
│   │   └── score.json               # IterationScore
│   └── 002/... (same structure)
```

This layout is the **public contract**. A downstream visualizer project is going to consume it. Do not change field names without bumping `schema_version`.

## 7. Dataset (`dataset.py`)

- Load `gretelai/symptom_to_diagnosis` from HuggingFace `datasets`.
- Deterministic train/eval split using `random.Random(seed=42)`: 100 eval examples, rest for few-shot pool.
- Expose `load_eval_split(n: int = 100) -> list[EvalExample]` returning `EvalExample` instances with `predicted_label=None`, `correct=False`, etc. (pre-fill fields the harness will overwrite).
- Cache the parquet under `data/`.

## 8. Harness interface (critical)

A harness is a single Python file that must define exactly one callable:

```python
def classify(text: str, target_client, target_model: str) -> str:
    """Return the predicted label (one of the dataset's class strings).

    You may call target_client.messages.create(...) any number of times.
    You may read/write files in /tmp. You may import stdlib and anthropic.
    You may NOT install packages, make network calls other than via target_client,
    or spawn subprocesses.
    """
```

The evaluator imports each candidate harness as a module via `importlib.util.spec_from_file_location`, runs `classify()` on each eval example, and records the trace. Failures (exceptions, timeout, invalid label) are caught and recorded as `error` in `EvalExample`, counted against accuracy but not crashing the loop.

**Provide a baseline harness** at `src/mini_meta_harness/harnesses/baseline_zero_shot.py`:

```python
# baseline_zero_shot.py
# Score: TBD (will be run as iteration 0)
# The simplest possible harness: zero-shot prompt, one LLM call, parse the label.

from anthropic import Anthropic

CLASS_LABELS = [...]  # filled in from dataset.py

def classify(text: str, target_client: Anthropic, target_model: str) -> str:
    prompt = f"Classify the following patient symptom description into exactly one of these diagnoses: {', '.join(CLASS_LABELS)}.\n\nSymptoms: {text}\n\nRespond with just the diagnosis name, nothing else."
    response = target_client.messages.create(
        model=target_model,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
```

This baseline is copied into `runs/<id>/iterations/000/harness.py` and scored first. The proposer sees it starting iteration 1.

## 9. Evaluator (`evaluator.py`)

`def evaluate(harness_path: Path, target_client, target_model: str, eval_set: list[EvalExample]) -> tuple[IterationScore, list[EvalExample]]`.

- Dynamically imports the harness via `importlib.util`.
- Runs each example through `classify()` with a 60-second per-example timeout (use `concurrent.futures.ThreadPoolExecutor` with `timeout=`).
- Normalize labels for comparison: lowercase, strip whitespace.
- Counts token usage from the Anthropic SDK response objects (`response.usage.input_tokens`, `response.usage.output_tokens`).
- Records one `EvalExample` per example to a JSONL.
- Returns aggregate `IterationScore` and the list of per-example traces.
- **Do not run examples in parallel across the eval set.** Sequential is fine for 100 examples and makes traces interpretable.

## 10. Memory (`memory.py`)

This is the filesystem-backed memory that the proposer has access to. The paper's big insight is that a filesystem is a sufficient external memory. Build it simply:

```python
class Memory:
    def __init__(self, run_dir: Path): ...
    def list_iterations(self) -> list[int]: ...
    def read_harness(self, iteration: int) -> str: ...
    def read_reasoning(self, iteration: int) -> str: ...
    def read_eval_trace(self, iteration: int) -> list[EvalExample]: ...
    def read_score(self, iteration: int) -> IterationScore: ...
    def scoreboard(self) -> list[tuple[int, float]]: ...  # (iteration, accuracy)
```

The proposer is given this object and is expected to use it. See §11.

## 11. Proposer (`proposer.py`)

This is the coding agent that writes new harnesses. The paper uses Claude Code with filesystem access; we approximate with a **tool-calling agent loop** using Anthropic's tool-use API against Moonshot's endpoint.

### Tools exposed to the proposer

1. `list_iterations() -> list[int]` — every past iteration index
2. `read_harness(iteration: int) -> str` — full source of a past harness
3. `read_reasoning(iteration: int) -> str` — proposer's own past reasoning
4. `read_eval_trace(iteration: int, only_failures: bool = False) -> list[dict]` — per-example results from a past iteration. When `only_failures=True`, returns only incorrect examples. This is the key affordance — the paper shows that access to raw failed-example traces is what drives gains.
5. `read_score(iteration: int) -> dict` — score summary
6. `scoreboard() -> list` — every past `(iteration, accuracy)` pair
7. `write_harness(code: str, reasoning: str) -> str` — submit the new harness. Terminates the agent loop and returns a confirmation string.

### Loop

```python
def propose_next_harness(memory: Memory, proposer_client, proposer_model: str, iteration_index: int) -> ProposerResult:
    system_prompt = PROPOSER_SYSTEM_PROMPT  # see §11.1
    messages = [{"role": "user", "content": f"Propose harness iteration {iteration_index}."}]
    fs_reads: list[FilesystemRead] = []
    proposer_tokens_in = 0
    proposer_tokens_out = 0

    for step in range(MAX_PROPOSER_STEPS):  # cap at 30
        response = proposer_client.messages.create(
            model=proposer_model,
            max_tokens=8192,
            system=system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        proposer_tokens_in += response.usage.input_tokens
        proposer_tokens_out += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            # Proposer gave up without submitting — rare, handle gracefully
            break

        # Process tool uses, record filesystem reads, append tool_result messages
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "write_harness":
                    return ProposerResult(
                        harness_code=block.input["code"],
                        reasoning=block.input["reasoning"],
                        filesystem_reads=fs_reads,
                        proposer_input_tokens=proposer_tokens_in,
                        proposer_output_tokens=proposer_tokens_out,
                    )
                # else: execute tool, record fs_read, append tool_result to messages
                ...

    raise ProposerFailure("Proposer exceeded step limit without submitting a harness")
```

### 11.1 Proposer system prompt

This is important enough to fix inline — copy it into the code:

```
You are a harness-engineering agent. Your job is to improve a Python classification
harness over many iterations, using full access to the filesystem of past attempts.

You have these tools:
- list_iterations(): see what exists
- read_harness(i), read_reasoning(i), read_score(i): inspect past attempts
- read_eval_trace(i, only_failures=True): read per-example results — USE THIS HEAVILY,
  especially on failures. Do not summarize traces in your head; read the raw content.
- scoreboard(): quick view of accuracy over iterations
- write_harness(code, reasoning): submit a new harness. This ends your turn.

Guidelines:
1. Read the scoreboard first. Focus on the best past iteration and the most recent one.
2. Read failure traces from the best iteration to find patterns.
3. Propose a concrete, targeted edit — do not rewrite from scratch unless the best
   harness is clearly broken.
4. Your harness must define def classify(text: str, target_client, target_model: str) -> str.
5. You may use few-shot examples, chain-of-thought prompting, multi-call verification,
   retrieval from an in-memory bank, or anything else expressible in a single Python file.
6. Do not attempt to install packages or make non-LLM network calls.
7. When you call write_harness, include reasoning that explains WHY this edit should help,
   grounded in specific failures you observed.
```

### 11.2 Tool schemas

Define proper Anthropic tool-use schemas for each of the 7 tools. Use JSON Schema for `input_schema`. Do not skip this — Moonshot's Anthropic-compatible endpoint requires valid tool schemas.

## 12. Outer loop (`outer_loop.py`)

```python
def run(config: RunConfig) -> RunSummary:
    run_dir = Path("runs") / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    proposer_client = make_client(...)
    target_client = make_client(...)
    memory = Memory(run_dir)
    eval_set = load_eval_split(config.eval_split_size)

    # Iteration 0: the baseline
    iter0 = evaluate_and_record(
        iteration_index=0,
        harness_source=read_baseline_source(),
        reasoning="Starting baseline: zero-shot single-call harness.",
        ...
    )

    # Iterations 1..N: proposer loop
    for i in range(1, config.iterations + 1):
        proposed = propose_next_harness(memory, proposer_client, config.proposer_model, i)
        record_iteration(i, proposed, eval_set, target_client, config.target_model, run_dir)

    summary = build_summary(run_dir, config)
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    return summary
```

Write each iteration's artifacts (harness.py, reasoning.md, filesystem_reads.jsonl, eval_trace.jsonl, score.json) eagerly after each iteration. If the run crashes at iteration 7, the first 6 must be on disk and valid.

## 13. CLI (`run.py`)

Typer-based:

```bash
mmh run --iterations 12 --dataset symptom2disease --eval-size 100
mmh run --iterations 3 --mock              # no API calls
mmh summary runs/2026-04-23_14-30-00       # print a nice summary of a past run
mmh cost runs/2026-04-23_14-30-00          # cost breakdown
```

Use `rich` for pretty output. Show a live progress bar for the evaluator (100 examples per iteration).

## 14. Mock mode

When `mock=True`:
- Replace proposer client with a `MockProposer` that returns canned `ProposerResult` objects sampled from `tests/fixtures/mock_proposer_results.json`.
- Replace target client with a `MockTargetClient` that returns a deterministic "predicted label" based on hashing the input (so some examples "correct" and some not, in a reproducible way).
- No API calls made. Entire pipeline runs in under 10 seconds for 3 iterations.

This is critical — recruiters skimming your GitHub can `uv run mmh run --iterations 3 --mock` and see the system work without a Moonshot key.

## 15. Cost tracking (`cost.py`)

Hardcode Moonshot Kimi K2.6 pricing (update these from the current Moonshot pricing page — rough ballpark: ~$0.60/M input, ~$2.50/M output; verify before shipping):

```python
PRICING = {
    "kimi-k2-0711-preview": {"input_per_mtok": 0.60, "output_per_mtok": 2.50},
    # Claude Haiku 4.5 pricing if user opts in as target
    "claude-haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
}
```

After every iteration, update `runs/<id>/cost.json` with running totals. Print cumulative cost to stderr at end of each iteration so the user can `Ctrl+C` if they're burning more than expected.

## 16. Notebook (`notebooks/explore_run.ipynb`)

Cells:
1. Imports, load the most recent run (or a given `run_id`).
2. Print the config nicely.
3. Accuracy-over-iterations line chart (matplotlib).
4. Cost-over-iterations stacked bar (proposer cost vs. target cost).
5. For the best iteration: print the harness code, print the reasoning.
6. Confusion matrix on the eval set for iteration 0 (baseline) vs. the best iteration — this is the "headline" visual.
7. Table of per-iteration proposer behavior: files read, steps taken, tokens used.

Keep it under ~25 cells. This notebook is the recruiter-facing narrative of the whole repo.

## 17. Tests

- `test_memory.py`: create a fake run dir with 2 iterations, verify `Memory` reads them correctly.
- `test_evaluator.py`: evaluate the baseline harness against a 3-example mock dataset with a mock target client, assert the trace is well-formed.
- `test_mock_proposer.py`: run 3 full iterations in mock mode, assert that the expected files exist on disk with valid schemas.

No need for exhaustive coverage — these three test files are enough to make CI green and demonstrate rigor.

## 18. GitHub Actions (`.github/workflows/ci.yml`)

Single workflow: on push, `uv sync`, `ruff check`, `pytest`. Run mock-mode end-to-end as a test. No Moonshot API calls in CI.

## 19. README

After the code works, write a real README. The current spec's README section is placeholder. The real one must have:
- TL;DR with the accuracy chart as a PNG at the top.
- Quickstart (4 commands).
- A "Differences from the paper" table.
- A "How it works" section explaining the outer loop in ~150 words.
- Citation block for the paper.
- Link back to `stanford-iris-lab/meta-harness`.

## 20. Build order (recommended for Claude Code)

1. `types.py` — everything else depends on these.
2. `config.py`, `.env.example`, `pyproject.toml`.
3. `dataset.py` + a quick smoke test that loading works.
4. `harnesses/baseline_zero_shot.py`.
5. `evaluator.py` + `test_evaluator.py` — get this green before anything else.
6. `memory.py` + `test_memory.py`.
7. Mock proposer + `test_mock_proposer.py` — full pipeline in mock mode working.
8. Real proposer (`proposer.py`) with tool-use against Moonshot.
9. `outer_loop.py` + `run.py` CLI.
10. `cost.py` integration.
11. Notebook.
12. Final README pass.

After step 7, do a `uv run mmh run --iterations 3 --mock` and confirm the full `runs/<id>/` directory structure is valid. Only then spend real API credits.

## 21. First real run checklist

Before spending money:
- [ ] Mock mode completes cleanly for 3 iterations.
- [ ] `eval_trace.jsonl` files validate against `EvalExample` schema.
- [ ] Cost estimator has been sanity-checked on a 1-iteration real run (expect <$1.50).
- [ ] Baseline iteration 0 accuracy looks plausible (should be ~0.4–0.6 for Symptom2Disease zero-shot).

Then: `uv run mmh run --iterations 12` with a budget alert at $20.

## 22. Open questions to flag to the user, not decide yourself

- Exact current Moonshot model ID (verify on platform.moonshot.ai — `kimi-k2-0711-preview` was current as of early 2026 but may have been superseded).
- Whether the user wants Haiku 4.5 as the target model instead of Kimi (different portfolio story — "cross-provider harness optimization").
- Whether to add a second dataset (AG News) after Symptom2Disease works, to demonstrate generality.

Do not make these decisions; surface them in a `QUESTIONS.md` file when you're done.

---

**End of spec.** Hand this file to Claude Code and start with step 20.1.