#!/usr/bin/env python3
"""Print a run's summary.json in a human-readable table.

Usage:
    uv run python scripts/print_run_summary.py runs/2026-04-23_14-30-00
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_meta_harness.types import RunSummary  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    path = run_dir / "summary.json"
    if not path.exists():
        print(f"no summary.json at {path}", file=sys.stderr)
        return 1
    s = RunSummary.model_validate(json.loads(path.read_text()))
    print(f"run_id:           {s.config.run_id}")
    print(f"proposer_model:   {s.config.proposer_model}")
    print(f"target_model:     {s.config.target_model}")
    print(f"iterations:       {s.config.iterations}")
    print(f"dataset:          {s.config.dataset}")
    print(f"eval_split_size:  {s.config.eval_split_size}")
    print(f"mock:             {s.config.mock}")
    print(f"schema_version:   {s.config.schema_version}")
    print(f"best_iteration:   {s.best_iteration_index}")
    print(f"total_usd:        ${s.total_cost.estimated_usd:,.4f}")
    print()
    print(f"{'iter':>4} {'accuracy':>9} {'errors':>7} {'mean_ms':>9} {'in_tok':>8} {'out_tok':>8}")
    for it in s.iterations:
        sc = it.score
        print(
            f"{it.index:>4} {sc.accuracy:>9.3f} {sc.n_errors:>7d} "
            f"{sc.mean_latency_ms:>9.1f} {sc.total_input_tokens:>8d} {sc.total_output_tokens:>8d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
