"""Pydantic schemas shared across the pipeline.

The on-disk layout described in README §6 is a public contract consumed by a
downstream visualizer. Bump ``schema_version`` on RunConfig if field names
change.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


class RunConfig(BaseModel):
    run_id: str
    iterations: int
    dataset: str = "symptom2disease"
    eval_split_size: int = 100
    proposer_model: str
    target_model: str
    mock: bool = False
    started_at: datetime
    schema_version: str = SCHEMA_VERSION


class FilesystemRead(BaseModel):
    path: str
    bytes_read: int
    at: datetime


class EvalExample(BaseModel):
    example_id: int
    input: str
    gold_label: str
    predicted_label: str | None = None
    correct: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
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

    def add(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            proposer_input_tokens=self.proposer_input_tokens + other.proposer_input_tokens,
            proposer_output_tokens=self.proposer_output_tokens + other.proposer_output_tokens,
            target_input_tokens=self.target_input_tokens + other.target_input_tokens,
            target_output_tokens=self.target_output_tokens + other.target_output_tokens,
            estimated_usd=self.estimated_usd + other.estimated_usd,
        )


class Iteration(BaseModel):
    index: int
    harness_code: str
    reasoning: str
    filesystem_reads: list[FilesystemRead] = Field(default_factory=list)
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


class ProposerResult(BaseModel):
    """Structured result returned by a proposer after one round of tool use."""

    harness_code: str
    reasoning: str
    filesystem_reads: list[FilesystemRead] = Field(default_factory=list)
    proposer_input_tokens: int = 0
    proposer_output_tokens: int = 0
    steps_taken: int = 0
