"""Real proposer: an Anthropic tool-use loop against Moonshot's endpoint.

The proposer is a coding-agent that reads past harnesses and their eval
traces, then submits a new harness via the ``write_harness`` tool. Submitting
ends the loop. See README §11.
"""

from __future__ import annotations

import json
from typing import Any

from .memory import Memory
from .types import FilesystemRead, ProposerResult

MAX_PROPOSER_STEPS = 30
MAX_TOKENS_PER_STEP = 8192

PROPOSER_SYSTEM_PROMPT = """You are a harness-engineering agent. Your job is to improve a Python classification
harness over many iterations, using full access to the filesystem of past attempts.

You have these tools:
- list_iterations(): see what exists
- read_harness(i), read_reasoning(i), read_score(i): inspect past attempts
- read_eval_trace(i, only_failures=True): read per-example results — USE THIS HEAVILY,
  especially on failures. Do not summarize traces in your head; read the raw content.
- scoreboard(): quick view of accuracy over iterations
- write_harness(code, reasoning): submit a new harness. This ends your turn.

Guidelines:
1. Read the scoreboard first. Focus on the best past iteration and the most recent one.
2. Read failure traces from the best iteration to find patterns.
3. Propose a concrete, targeted edit — do not rewrite from scratch unless the best
   harness is clearly broken.
4. Your harness must define def classify(text: str, target_client, target_model: str) -> str.
5. You may use few-shot examples, chain-of-thought prompting, multi-call verification,
   retrieval from an in-memory bank, or anything else expressible in a single Python file.
6. Do not attempt to install packages or make non-LLM network calls.
7. When you call write_harness, include reasoning that explains WHY this edit should help,
   grounded in specific failures you observed.
"""


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_iterations",
        "description": "Return every past iteration index (ints, ascending).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_harness",
        "description": "Return the full Python source of a past iteration's harness.",
        "input_schema": {
            "type": "object",
            "properties": {"iteration": {"type": "integer", "minimum": 0}},
            "required": ["iteration"],
        },
    },
    {
        "name": "read_reasoning",
        "description": "Return the proposer's reasoning markdown for a past iteration.",
        "input_schema": {
            "type": "object",
            "properties": {"iteration": {"type": "integer", "minimum": 0}},
            "required": ["iteration"],
        },
    },
    {
        "name": "read_eval_trace",
        "description": (
            "Return per-example eval results for a past iteration. "
            "Set only_failures=true to restrict to examples the harness got wrong or errored on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iteration": {"type": "integer", "minimum": 0},
                "only_failures": {"type": "boolean", "default": False},
            },
            "required": ["iteration"],
        },
    },
    {
        "name": "read_score",
        "description": "Return the score summary for a past iteration.",
        "input_schema": {
            "type": "object",
            "properties": {"iteration": {"type": "integer", "minimum": 0}},
            "required": ["iteration"],
        },
    },
    {
        "name": "scoreboard",
        "description": "Return (iteration, accuracy) pairs for every past iteration.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "write_harness",
        "description": (
            "Submit the new harness as Python source. The code must define "
            "`def classify(text, target_client, target_model) -> str`. This ends your turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full Python source for the new harness."},
                "reasoning": {
                    "type": "string",
                    "description": "Why this edit should help, grounded in specific failures you observed.",
                },
            },
            "required": ["code", "reasoning"],
        },
    },
]


class ProposerFailure(RuntimeError):
    pass


def _run_tool(memory: Memory, name: str, args: dict[str, Any]) -> str:
    if name == "list_iterations":
        return json.dumps(memory.list_iterations())
    if name == "read_harness":
        return memory.read_harness(int(args["iteration"]))
    if name == "read_reasoning":
        return memory.read_reasoning(int(args["iteration"]))
    if name == "read_eval_trace":
        only = bool(args.get("only_failures", False))
        traces = memory.read_eval_trace(int(args["iteration"]), only_failures=only)
        return json.dumps([t.model_dump(mode="json") for t in traces])
    if name == "read_score":
        return json.dumps(memory.read_score(int(args["iteration"])).model_dump())
    if name == "scoreboard":
        return json.dumps(memory.scoreboard())
    raise ValueError(f"Unknown tool: {name}")


def propose_next_harness(
    memory: Memory,
    proposer_client,
    proposer_model: str,
    iteration_index: int,
) -> ProposerResult:
    messages: list[dict] = [
        {"role": "user", "content": f"Propose harness iteration {iteration_index}."}
    ]
    proposer_in = 0
    proposer_out = 0

    for step in range(MAX_PROPOSER_STEPS):
        response = proposer_client.messages.create(
            model=proposer_model,
            max_tokens=MAX_TOKENS_PER_STEP,
            system=PROPOSER_SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        if getattr(response, "usage", None):
            proposer_in += response.usage.input_tokens or 0
            proposer_out += response.usage.output_tokens or 0

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # No tool call, no submission — proposer gave up.
            break

        # Check for a terminal write_harness submission first.
        for block in tool_uses:
            if block.name == "write_harness":
                code = block.input.get("code", "")
                reasoning = block.input.get("reasoning", "")
                reads = memory.pop_reads()
                return ProposerResult(
                    harness_code=code,
                    reasoning=reasoning,
                    filesystem_reads=reads,
                    proposer_input_tokens=proposer_in,
                    proposer_output_tokens=proposer_out,
                    steps_taken=step + 1,
                )

        # Append the assistant turn, then run tools and append tool_result turn.
        assistant_content = [
            b.model_dump() if hasattr(b, "model_dump") else _block_to_dict(b)
            for b in response.content
        ]
        messages.append({"role": "assistant", "content": assistant_content})
        tool_results: list[dict] = []
        for block in tool_uses:
            try:
                result = _run_tool(memory, block.name, dict(block.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            except Exception as e:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "is_error": True,
                        "content": f"{type(e).__name__}: {e}",
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    raise ProposerFailure(
        f"Proposer exceeded step limit ({MAX_PROPOSER_STEPS}) without submitting a harness"
    )


def _block_to_dict(block: Any) -> dict:
    """Fallback serializer for content blocks returned by the Anthropic SDK."""
    t = getattr(block, "type", None)
    if t == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if t == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}),
        }
    return {"type": t or "unknown"}


__all__ = [
    "propose_next_harness",
    "ProposerFailure",
    "PROPOSER_SYSTEM_PROMPT",
    "TOOL_SCHEMAS",
    "MAX_PROPOSER_STEPS",
    "FilesystemRead",
]
