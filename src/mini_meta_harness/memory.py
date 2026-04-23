"""Filesystem-backed memory the proposer reads from.

The paper's core claim: a filesystem of past harnesses + raw execution traces
is a sufficient external memory for the proposer. This module is just the
thin read API; writes go through the outer loop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .types import EvalExample, FilesystemRead, IterationScore


class Memory:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.iterations_dir = self.run_dir / "iterations"
        self._reads: list[FilesystemRead] = []

    # -- reads ---------------------------------------------------------------

    def _iter_dir(self, iteration: int) -> Path:
        return self.iterations_dir / f"{iteration:03d}"

    def _record_read(self, path: Path, content: bytes | str) -> None:
        size = len(content) if isinstance(content, bytes) else len(content.encode("utf-8"))
        self._reads.append(
            FilesystemRead(path=str(path), bytes_read=size, at=datetime.now(timezone.utc))
        )

    def list_iterations(self) -> list[int]:
        if not self.iterations_dir.exists():
            return []
        out: list[int] = []
        for p in sorted(self.iterations_dir.iterdir()):
            if p.is_dir() and p.name.isdigit():
                out.append(int(p.name))
        return out

    def read_harness(self, iteration: int) -> str:
        path = self._iter_dir(iteration) / "harness.py"
        text = path.read_text()
        self._record_read(path, text)
        return text

    def read_reasoning(self, iteration: int) -> str:
        path = self._iter_dir(iteration) / "reasoning.md"
        text = path.read_text() if path.exists() else ""
        self._record_read(path, text)
        return text

    def read_eval_trace(
        self, iteration: int, only_failures: bool = False
    ) -> list[EvalExample]:
        path = self._iter_dir(iteration) / "eval_trace.jsonl"
        if not path.exists():
            return []
        raw = path.read_text()
        self._record_read(path, raw)
        out: list[EvalExample] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            ex = EvalExample.model_validate(json.loads(line))
            if only_failures and ex.correct and ex.error is None:
                continue
            out.append(ex)
        return out

    def read_score(self, iteration: int) -> IterationScore:
        path = self._iter_dir(iteration) / "score.json"
        raw = path.read_text()
        self._record_read(path, raw)
        return IterationScore.model_validate(json.loads(raw))

    def scoreboard(self) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for i in self.list_iterations():
            try:
                s = self.read_score(i)
                out.append((i, s.accuracy))
            except Exception:
                continue
        return out

    # -- read tracking ------------------------------------------------------

    def pop_reads(self) -> list[FilesystemRead]:
        reads = self._reads
        self._reads = []
        return reads
