"""Run a candidate harness over the eval set and record a trace."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .types import EvalExample, IterationScore

PER_EXAMPLE_TIMEOUT_S = 60


class _UsageTrackingClient:
    """Thin wrapper that records token usage for each ``messages.create`` call.

    Harnesses are handed this instead of the raw Anthropic client so that we
    can attribute per-example token counts without asking harness authors to
    thread that bookkeeping themselves.
    """

    def __init__(self, inner):
        self._inner = inner
        self.messages = _UsageTrackingMessages(self)
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


class _UsageTrackingMessages:
    def __init__(self, parent: _UsageTrackingClient):
        self._parent = parent

    def create(self, **kwargs):
        resp = self._parent._inner.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._parent.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self._parent.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return resp


@contextmanager
def _no_bytecode():
    """Skip __pycache__ writes while dynamically loading candidate harnesses."""
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = prev


def _load_harness_module(harness_path: Path):
    module_name = f"harness_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load harness at {harness_path}")
    mod = importlib.util.module_from_spec(spec)
    with _no_bytecode():
        spec.loader.exec_module(mod)
    if not hasattr(mod, "classify"):
        raise AttributeError(
            f"Harness at {harness_path} does not define a `classify` function"
        )
    return mod


def _normalize_label(s: str | None) -> str:
    return (s or "").strip().lower()


def evaluate(
    harness_path: Path,
    target_client,
    target_model: str,
    eval_set: list[EvalExample],
    *,
    trace_path: Path | None = None,
    progress_callback=None,
) -> tuple[IterationScore, list[EvalExample]]:
    """Run the harness over ``eval_set`` and return ``(score, per-example traces)``.

    Failures are swallowed into the ``error`` field on each ``EvalExample`` —
    they count against accuracy but never crash the loop.
    """
    mod = _load_harness_module(harness_path)
    classify_fn = mod.classify

    tracked = _UsageTrackingClient(target_client)
    traces: list[EvalExample] = []

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_path.exists():
            trace_path.unlink()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        for i, example in enumerate(eval_set):
            entry = deepcopy(example)
            tracked.reset()
            t0 = time.perf_counter()
            try:
                fut = executor.submit(classify_fn, example.input, tracked, target_model)
                prediction = fut.result(timeout=PER_EXAMPLE_TIMEOUT_S)
                entry.predicted_label = str(prediction)
                entry.correct = _normalize_label(prediction) == _normalize_label(example.gold_label)
            except FuturesTimeout:
                entry.error = f"timeout>{PER_EXAMPLE_TIMEOUT_S}s"
                entry.predicted_label = None
                entry.correct = False
            except Exception as e:
                entry.error = f"{type(e).__name__}: {e}"
                entry.predicted_label = None
                entry.correct = False
            finally:
                entry.latency_ms = (time.perf_counter() - t0) * 1000.0
                entry.input_tokens = tracked.input_tokens
                entry.output_tokens = tracked.output_tokens

            traces.append(entry)
            if trace_path is not None:
                with trace_path.open("a") as f:
                    f.write(json.dumps(entry.model_dump(), default=str) + "\n")
            if progress_callback is not None:
                progress_callback(i + 1, len(eval_set), entry)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    n = len(traces)
    n_correct = sum(1 for t in traces if t.correct)
    n_errors = sum(1 for t in traces if t.error is not None)
    total_in = sum(t.input_tokens for t in traces)
    total_out = sum(t.output_tokens for t in traces)
    mean_latency = (sum(t.latency_ms for t in traces) / n) if n else 0.0
    score = IterationScore(
        accuracy=(n_correct / n) if n else 0.0,
        n_examples=n,
        n_correct=n_correct,
        n_errors=n_errors,
        mean_latency_ms=mean_latency,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
    )
    return score, traces
