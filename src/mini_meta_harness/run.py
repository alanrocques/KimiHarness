"""Typer CLI: ``mmh run``, ``mmh summary``, ``mmh cost``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import outer_loop
from .config import DEFAULT_PROPOSER_MODEL, DEFAULT_TARGET_MODEL
from .cost import read_running_total
from .types import RunConfig, RunSummary

app = typer.Typer(
    no_args_is_help=True,
    help="mini-meta-harness — reproduce Meta-Harness on a budget with Kimi K2.6.",
)
console = Console(stderr=True)


@app.command()
def run(
    iterations: int = typer.Option(12, help="Number of proposer iterations (iteration 0 is always baseline)."),
    dataset: str = typer.Option("symptom2disease", help="Dataset name."),
    eval_size: int = typer.Option(100, help="Number of held-out eval examples per iteration."),
    proposer_model: str = typer.Option(DEFAULT_PROPOSER_MODEL, help="Moonshot model id used for proposing harnesses."),
    target_model: str = typer.Option(DEFAULT_TARGET_MODEL, help="Model id used inside the harness (target)."),
    mock: bool = typer.Option(False, "--mock", help="Run the full pipeline with canned responses; no API calls."),
    run_id: Optional[str] = typer.Option(None, help="Override the auto-generated run id."),
):
    """Execute the Meta-Harness outer loop."""
    rid = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    config = RunConfig(
        run_id=rid,
        iterations=iterations,
        dataset=dataset,
        eval_split_size=eval_size,
        proposer_model=proposer_model,
        target_model=target_model,
        mock=mock,
        started_at=datetime.now(timezone.utc),
    )
    console.log(f"[bold]Starting run[/bold] {rid} (mock={mock}, iterations={iterations})")
    summary = outer_loop.run(config)
    _print_summary(summary)
    console.log(f"Run written to runs/{rid}")


@app.command()
def summary(run_dir: Path = typer.Argument(..., help="Path to a run directory (e.g. runs/2026-04-23_14-30-00).")):
    """Pretty-print a past run's summary.json."""
    path = run_dir / "summary.json"
    if not path.exists():
        console.print(f"[red]No summary.json at {path}[/red]")
        raise typer.Exit(code=1)
    s = RunSummary.model_validate(json.loads(path.read_text()))
    _print_summary(s)


@app.command()
def cost(run_dir: Path = typer.Argument(..., help="Path to a run directory.")):
    """Print cost breakdown for a past run."""
    total = read_running_total(run_dir)
    table = Table(title=f"Cost — {run_dir.name}")
    table.add_column("field")
    table.add_column("value", justify="right")
    for k, v in total.model_dump().items():
        if k == "estimated_usd":
            table.add_row(k, f"${v:,.4f}")
        else:
            table.add_row(k, f"{v:,}")
    Console().print(table)


def _print_summary(summary: RunSummary) -> None:
    cfg = summary.config
    header = Table(title=f"Run {cfg.run_id}", show_header=False)
    header.add_column("k")
    header.add_column("v")
    header.add_row("proposer_model", cfg.proposer_model)
    header.add_row("target_model", cfg.target_model)
    header.add_row("iterations", str(cfg.iterations))
    header.add_row("dataset", cfg.dataset)
    header.add_row("mock", str(cfg.mock))
    header.add_row("best_iteration", str(summary.best_iteration_index))
    header.add_row("total_usd", f"${summary.total_cost.estimated_usd:,.4f}")
    Console().print(header)

    iters = Table(title="Scoreboard")
    iters.add_column("iter", justify="right")
    iters.add_column("accuracy", justify="right")
    iters.add_column("errors", justify="right")
    iters.add_column("mean_latency_ms", justify="right")
    for it in summary.iterations:
        iters.add_row(
            str(it.index),
            f"{it.score.accuracy:.3f}",
            str(it.score.n_errors),
            f"{it.score.mean_latency_ms:.1f}",
        )
    Console().print(iters)


if __name__ == "__main__":
    app()
