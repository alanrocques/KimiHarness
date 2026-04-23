"""Meta-Harness outer loop: evaluate baseline, then iterate the proposer."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import cost as cost_mod
from .dataset import load as load_dataset
from .evaluator import evaluate
from .harnesses import baseline_zero_shot
from .memory import Memory
from .mock import MockProposer, MockTargetClient
from .types import (
    CostBreakdown,
    EvalExample,
    FilesystemRead,
    Iteration,
    IterationScore,
    ProposerResult,
    RunConfig,
    RunSummary,
)


def _iter_dir(run_dir: Path, index: int) -> Path:
    d = run_dir / "iterations" / f"{index:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_iteration_artifacts(
    run_dir: Path,
    index: int,
    harness_code: str,
    reasoning: str,
    filesystem_reads: list[FilesystemRead],
    traces: list[EvalExample],
    score: IterationScore,
) -> None:
    d = _iter_dir(run_dir, index)
    (d / "harness.py").write_text(harness_code)
    (d / "reasoning.md").write_text(reasoning + "\n" if not reasoning.endswith("\n") else reasoning)
    with (d / "filesystem_reads.jsonl").open("w") as f:
        for r in filesystem_reads:
            f.write(json.dumps(r.model_dump(), default=str) + "\n")
    with (d / "eval_trace.jsonl").open("w") as f:
        for t in traces:
            f.write(json.dumps(t.model_dump(), default=str) + "\n")
    (d / "score.json").write_text(json.dumps(score.model_dump(), indent=2))


def _baseline_source() -> str:
    return Path(baseline_zero_shot.__file__).read_text()


def run(config: RunConfig, *, verbose: bool = True) -> RunSummary:
    """Execute the full outer loop and return the final RunSummary.

    Each iteration's artifacts are written to disk eagerly, so a crash at
    iteration N still leaves iterations 0..N-1 as a valid partial run.
    """
    run_dir = Path("runs") / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    dataset = load_dataset(config.eval_split_size)
    eval_set = dataset.eval_examples

    # Clients
    if config.mock:
        target_client = MockTargetClient()
        target_client.register_gold(eval_set)
        proposer_fn = None  # resolved below using MockProposer
    else:
        from .config import make_moonshot_client, make_target_client

        proposer_client = make_moonshot_client()
        target_client = make_target_client(config.target_model)
        proposer_fn = proposer_client  # actual client; proposer module uses it

    memory = Memory(run_dir)
    running_total = CostBreakdown()

    # --- Iteration 0: baseline ---------------------------------------------
    _run_iteration(
        config=config,
        run_dir=run_dir,
        memory=memory,
        index=0,
        harness_code=_baseline_source(),
        reasoning="Baseline: zero-shot single-call harness from src/mini_meta_harness/harnesses/baseline_zero_shot.py.",
        filesystem_reads=[],
        proposer_tokens=(0, 0),
        target_client=target_client,
        eval_set=eval_set,
        running_total=running_total,
        verbose=verbose,
    )

    # --- Iterations 1..N: proposer loop ------------------------------------
    mock_proposer = MockProposer(memory=memory) if config.mock else None
    for i in range(1, config.iterations + 1):
        proposed = _invoke_proposer(
            config=config,
            memory=memory,
            iteration_index=i,
            proposer_client=proposer_fn,
            mock_proposer=mock_proposer,
        )
        harness_path_code = _persist_harness_for_eval(run_dir, i, proposed.harness_code)
        _run_iteration(
            config=config,
            run_dir=run_dir,
            memory=memory,
            index=i,
            harness_code=harness_path_code,
            reasoning=proposed.reasoning,
            filesystem_reads=proposed.filesystem_reads,
            proposer_tokens=(proposed.proposer_input_tokens, proposed.proposer_output_tokens),
            target_client=target_client,
            eval_set=eval_set,
            running_total=running_total,
            verbose=verbose,
            already_written=True,
        )

    summary = _build_summary(run_dir, config, running_total)
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    return summary


def _invoke_proposer(
    *,
    config: RunConfig,
    memory: Memory,
    iteration_index: int,
    proposer_client,
    mock_proposer: MockProposer | None,
) -> ProposerResult:
    if config.mock:
        assert mock_proposer is not None
        return mock_proposer.propose(iteration_index)
    from .proposer import propose_next_harness

    return propose_next_harness(
        memory=memory,
        proposer_client=proposer_client,
        proposer_model=config.proposer_model,
        iteration_index=iteration_index,
    )


def _persist_harness_for_eval(run_dir: Path, index: int, code: str) -> str:
    """Write the harness source to its iteration dir early so the evaluator
    can import it by path. Returns the (possibly unchanged) source string."""
    d = _iter_dir(run_dir, index)
    (d / "harness.py").write_text(code)
    return code


def _run_iteration(
    *,
    config: RunConfig,
    run_dir: Path,
    memory: Memory,
    index: int,
    harness_code: str,
    reasoning: str,
    filesystem_reads: list[FilesystemRead],
    proposer_tokens: tuple[int, int],
    target_client,
    eval_set: list[EvalExample],
    running_total: CostBreakdown,
    verbose: bool,
    already_written: bool = False,
) -> None:
    d = _iter_dir(run_dir, index)
    harness_path = d / "harness.py"
    if not already_written:
        harness_path.write_text(harness_code)

    trace_path = d / "eval_trace.jsonl"

    def _cb(done: int, total: int, _ex: EvalExample) -> None:
        if verbose and (done == total or done % 10 == 0):
            print(f"  iter {index:03d}: {done}/{total} examples", file=sys.stderr)

    score, traces = evaluate(
        harness_path=harness_path,
        target_client=target_client,
        target_model=config.target_model,
        eval_set=eval_set,
        trace_path=trace_path,
        progress_callback=_cb,
    )

    proposer_in, proposer_out = proposer_tokens
    cost = cost_mod.breakdown_for_iteration(
        proposer_model=config.proposer_model,
        target_model=config.target_model,
        proposer_input_tokens=proposer_in,
        proposer_output_tokens=proposer_out,
        target_input_tokens=score.total_input_tokens,
        target_output_tokens=score.total_output_tokens,
    )

    # Merge into running total and write running cost.json
    new_total = running_total.add(cost)
    # Mutate in place so the caller's reference updates.
    running_total.proposer_input_tokens = new_total.proposer_input_tokens
    running_total.proposer_output_tokens = new_total.proposer_output_tokens
    running_total.target_input_tokens = new_total.target_input_tokens
    running_total.target_output_tokens = new_total.target_output_tokens
    running_total.estimated_usd = new_total.estimated_usd
    cost_mod.write_running_total(run_dir, running_total)

    _write_iteration_artifacts(
        run_dir=run_dir,
        index=index,
        harness_code=harness_code,
        reasoning=reasoning,
        filesystem_reads=filesystem_reads,
        traces=traces,
        score=score,
    )

    if verbose:
        print(
            f"[iter {index:03d}] accuracy={score.accuracy:.3f} "
            f"errors={score.n_errors} "
            f"cost=${running_total.estimated_usd:.4f}",
            file=sys.stderr,
        )


def _build_summary(run_dir: Path, config: RunConfig, total_cost: CostBreakdown) -> RunSummary:
    iters_dir = run_dir / "iterations"
    iteration_indices = sorted(
        int(p.name) for p in iters_dir.iterdir() if p.is_dir() and p.name.isdigit()
    )
    iterations: list[Iteration] = []
    best_idx = iteration_indices[0] if iteration_indices else 0
    best_acc = -1.0
    for i in iteration_indices:
        d = iters_dir / f"{i:03d}"
        score = IterationScore.model_validate(json.loads((d / "score.json").read_text()))
        harness_code = (d / "harness.py").read_text()
        reasoning = (d / "reasoning.md").read_text() if (d / "reasoning.md").exists() else ""
        reads: list[FilesystemRead] = []
        fr_path = d / "filesystem_reads.jsonl"
        if fr_path.exists():
            for line in fr_path.read_text().splitlines():
                line = line.strip()
                if line:
                    reads.append(FilesystemRead.model_validate(json.loads(line)))
        # We don't persist started/finished_at per iteration separately; use
        # file mtime as an approximation for the summary.
        finished = datetime.fromtimestamp((d / "score.json").stat().st_mtime)
        started = datetime.fromtimestamp((d / "harness.py").stat().st_mtime)
        iterations.append(
            Iteration(
                index=i,
                harness_code=harness_code,
                reasoning=reasoning,
                filesystem_reads=reads,
                score=score,
                # Cost per iteration isn't stored separately in this simplified
                # layout; callers wanting that granularity can re-derive from
                # eval_trace tokens plus proposer_tokens in filesystem_reads.
                cost=CostBreakdown(),
                started_at=started,
                finished_at=finished,
            )
        )
        if score.accuracy > best_acc:
            best_acc = score.accuracy
            best_idx = i

    return RunSummary(
        config=config,
        iterations=iterations,
        total_cost=total_cost,
        best_iteration_index=best_idx,
        finished_at=datetime.now(UTC),
    )
