from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from mini_meta_harness import outer_loop
from mini_meta_harness.types import EvalExample, IterationScore, RunConfig, RunSummary


def test_mock_run_three_iterations_produces_valid_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = RunConfig(
        run_id="test-mock-run",
        iterations=3,
        eval_split_size=10,
        proposer_model="kimi-k2.6",
        target_model="kimi-k2.6",
        mock=True,
        started_at=datetime.now(UTC),
    )
    outer_loop.run(cfg, verbose=False)

    run_dir = tmp_path / "runs" / "test-mock-run"
    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "cost.json").exists()

    for i in range(0, 4):  # baseline + 3 iterations
        d = run_dir / "iterations" / f"{i:03d}"
        assert d.exists(), f"missing iteration dir {d}"
        assert (d / "harness.py").exists()
        assert (d / "reasoning.md").exists()
        assert (d / "filesystem_reads.jsonl").exists()
        assert (d / "eval_trace.jsonl").exists()
        assert (d / "score.json").exists()
        # trace validates against EvalExample
        lines = [ln for ln in (d / "eval_trace.jsonl").read_text().splitlines() if ln.strip()]
        assert lines, f"empty eval_trace in iter {i}"
        for ln in lines:
            EvalExample.model_validate(json.loads(ln))
        # score validates
        IterationScore.model_validate(json.loads((d / "score.json").read_text()))

    # Summary validates and references the best iteration.
    parsed = RunSummary.model_validate(json.loads((run_dir / "summary.json").read_text()))
    assert parsed.best_iteration_index in {it.index for it in parsed.iterations}
    assert len(parsed.iterations) == 4
