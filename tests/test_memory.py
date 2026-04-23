from __future__ import annotations

import json
from pathlib import Path

from mini_meta_harness.memory import Memory
from mini_meta_harness.types import EvalExample, IterationScore


def _seed_iteration(root: Path, i: int, code: str, traces: list[EvalExample], score: IterationScore, reasoning: str = "") -> None:
    d = root / "iterations" / f"{i:03d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "harness.py").write_text(code)
    (d / "reasoning.md").write_text(reasoning)
    with (d / "eval_trace.jsonl").open("w") as f:
        for t in traces:
            f.write(json.dumps(t.model_dump(), default=str) + "\n")
    (d / "score.json").write_text(json.dumps(score.model_dump()))


def test_memory_reads_two_iterations(tmp_path: Path):
    traces_a = [
        EvalExample(example_id=0, input="x", gold_label="a", predicted_label="a", correct=True),
        EvalExample(example_id=1, input="y", gold_label="b", predicted_label="z", correct=False),
    ]
    traces_b = [
        EvalExample(example_id=0, input="x", gold_label="a", predicted_label="a", correct=True),
        EvalExample(example_id=1, input="y", gold_label="b", predicted_label="b", correct=True),
    ]
    score_a = IterationScore(accuracy=0.5, n_examples=2, n_correct=1, n_errors=0, mean_latency_ms=10.0, total_input_tokens=20, total_output_tokens=5)
    score_b = IterationScore(accuracy=1.0, n_examples=2, n_correct=2, n_errors=0, mean_latency_ms=12.0, total_input_tokens=24, total_output_tokens=6)

    _seed_iteration(tmp_path, 0, "def classify(*a, **k): return 'a'\n", traces_a, score_a, reasoning="baseline")
    _seed_iteration(tmp_path, 1, "def classify(*a, **k): return 'b'\n", traces_b, score_b, reasoning="iter 1")

    mem = Memory(tmp_path)
    assert mem.list_iterations() == [0, 1]
    assert "classify" in mem.read_harness(0)
    assert mem.read_reasoning(1) == "iter 1"
    assert mem.read_score(0).accuracy == 0.5
    assert mem.scoreboard() == [(0, 0.5), (1, 1.0)]

    all_traces = mem.read_eval_trace(0)
    assert len(all_traces) == 2
    failures = mem.read_eval_trace(0, only_failures=True)
    assert len(failures) == 1
    assert failures[0].example_id == 1

    reads = mem.pop_reads()
    assert reads, "memory should record reads"
    # Second pop returns nothing (reads are consumed).
    assert mem.pop_reads() == []
