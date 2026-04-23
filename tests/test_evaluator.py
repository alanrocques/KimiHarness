from __future__ import annotations

from pathlib import Path

from mini_meta_harness.evaluator import evaluate
from mini_meta_harness.mock import MockTargetClient
from mini_meta_harness.types import EvalExample

BASELINE = Path(__file__).resolve().parents[1] / "src/mini_meta_harness/harnesses/baseline_zero_shot.py"


def test_evaluate_baseline_on_three_examples(tmp_path: Path):
    examples = [
        EvalExample(example_id=0, input="Sneezing and runny nose for two days", gold_label="common cold"),
        EvalExample(example_id=1, input="Burning urination and frequent urges", gold_label="urinary tract infection"),
        EvalExample(example_id=2, input="Throbbing one-sided headache with nausea", gold_label="migraine"),
    ]
    client = MockTargetClient()
    client.register_gold(examples)

    trace_path = tmp_path / "eval_trace.jsonl"
    score, traces = evaluate(
        harness_path=BASELINE,
        target_client=client,
        target_model="kimi-k2.6",
        eval_set=examples,
        trace_path=trace_path,
    )
    assert score.n_examples == 3
    assert len(traces) == 3
    assert all(t.latency_ms >= 0 for t in traces)
    assert trace_path.exists()
    lines = [ln for ln in trace_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    # Every entry should have a prediction or an error recorded.
    for t in traces:
        assert (t.predicted_label is not None) or (t.error is not None)


def test_evaluator_catches_harness_errors(tmp_path: Path):
    bad = tmp_path / "bad_harness.py"
    bad.write_text("def classify(text, target_client, target_model):\n    raise RuntimeError('boom')\n")
    examples = [EvalExample(example_id=0, input="x", gold_label="common cold")]
    score, traces = evaluate(
        harness_path=bad,
        target_client=MockTargetClient(),
        target_model="kimi-k2.6",
        eval_set=examples,
    )
    assert score.n_errors == 1
    assert traces[0].error is not None
