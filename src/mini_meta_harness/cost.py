"""Dollar-cost accounting.

Pricing numbers are approximate — verify against the current Moonshot /
Anthropic pricing pages before running expensive jobs. See README §15.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import CostBreakdown

PRICING: dict[str, dict[str, float]] = {
    # Moonshot Kimi K2.6 — flagship (https://platform.kimi.ai/docs/pricing/chat-k26).
    # Cache-miss input is used as the conservative default; if the SDK starts
    # surfacing cache_read tokens, split this into hit/miss branches.
    "kimi-k2.6": {"input_per_mtok": 0.95, "output_per_mtok": 4.00},
    # Prior-generation Kimi, kept for back-compat with older run directories.
    "kimi-k2-0711-preview": {"input_per_mtok": 0.60, "output_per_mtok": 2.50},
    # Anthropic Claude Haiku 4.5 (if user opts in as target)
    "claude-haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    "claude-haiku-4-5-20251001": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
}


def _price_for(model: str) -> dict[str, float]:
    if model in PRICING:
        return PRICING[model]
    # Unknown model: default to Kimi-style pricing rather than crashing.
    return PRICING["kimi-k2.6"]


def usd_for(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _price_for(model)
    return (
        input_tokens / 1_000_000 * p["input_per_mtok"]
        + output_tokens / 1_000_000 * p["output_per_mtok"]
    )


def breakdown_for_iteration(
    proposer_model: str,
    target_model: str,
    proposer_input_tokens: int,
    proposer_output_tokens: int,
    target_input_tokens: int,
    target_output_tokens: int,
) -> CostBreakdown:
    usd = usd_for(proposer_model, proposer_input_tokens, proposer_output_tokens) + usd_for(
        target_model, target_input_tokens, target_output_tokens
    )
    return CostBreakdown(
        proposer_input_tokens=proposer_input_tokens,
        proposer_output_tokens=proposer_output_tokens,
        target_input_tokens=target_input_tokens,
        target_output_tokens=target_output_tokens,
        estimated_usd=round(usd, 6),
    )


def write_running_total(run_dir: Path, total: CostBreakdown) -> None:
    (run_dir / "cost.json").write_text(json.dumps(total.model_dump(), indent=2))


def read_running_total(run_dir: Path) -> CostBreakdown:
    path = run_dir / "cost.json"
    if not path.exists():
        return CostBreakdown()
    return CostBreakdown.model_validate(json.loads(path.read_text()))
